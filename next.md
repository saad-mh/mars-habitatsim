# Manual Mesh Segmentation → Synthetic Dataset Pipeline

## Context

Part of the `sam_vla` Mars rover autonomy pipeline. Goal: generate a labeled
segmentation dataset (RGB frame ↔ per-pixel category mask) directly from
simulation ground-truth geometry, instead of manual pixel annotation.

Terrain meshes are manually divided into category groups by a human (mentor's
requirement — auto-segmentation of natural rock/terrain is unreliable). Each
divided sub-mesh gets a convex hull, which **defines the mask boundary**
directly (not just a physics/collision proxy). Hull overshoot on jagged
rocks is accepted as small label noise, to be handled at a later cleanup
stage — not solved at generation time.

Categories (up to 4, TBD which are actually used):

- small_rock
- big_rock
- bedrock
- hole_in_ground

---

## Pipeline Steps

### 1. Manual mesh division + convex hull authoring

- Cut the terrain/scene mesh into disjoint sub-meshes along category
  boundaries (Blender or in-engine editor).
- Compute a convex hull per sub-mesh. This hull **is** the mask boundary —
  not just used for raycast/collision simplification.
- Keep each category's pieces as separate objects, not one merged mesh —
  per-object mesh IDs are required downstream.
- Accept that hulls overshoot jagged/irregular rocks slightly. This is
  fine — treat as label noise, not a blocker.

### 2. Mesh ID → category registry

- Assign a stable unique `mesh_id` to every sub-mesh at import/build time.
  Do not rely on engine-runtime instance IDs alone if they aren't stable
  across reloads — assign your own persistent ID and store it as
  metadata/custom property on the mesh object if the engine supports it.
- Maintain a standalone JSON registry, decoupled from the sim scene file,
  so it survives re-exports:

```json
{
  "mesh_id_map": {
    "1024": { "category": "small_rock", "name": "rock_cluster_A_03" },
    "1025": { "category": "bedrock", "name": "bedrock_slab_01" },
    "1026": { "category": "hole_in_ground", "name": "crater_02" }
  }
}
```

### 3. Per-frame ground truth extraction

Two approaches — pick based on engine support (Habitat-Sim likely supports
approach B natively via its semantic sensor; confirm before committing):

**A. Raycast-based**

- Cast a ray per pixel (or sampled grid) from camera through scene per
  captured frame.
- Each hit resolves to a `mesh_id` → look up category → paint mask.
- Cost: expensive at full pixel density; best used offline per recorded
  frame, or for sparse validation (e.g. "what did the rover's forward
  ray hit"), not as the primary dense-mask method.

**B. Instance/semantic buffer rendering (preferred for dense masks)**

- Render an instance-ID buffer alongside RGB in the same pass (rasterized,
  not per-pixel raycast) — each pixel encodes the mesh ID directly.
- Vectorized lookup through `mesh_id_map` → category mask. Much cheaper
  than raycasting per pixel, exact edges.
- **Action item:** check whether Habitat-Sim's semantic sensor already
  outputs this directly — likely yes, since it's built for this exact
  use case. If so, skip building custom raycast-per-pixel infra entirely.

**Suggested split:** use B for dense per-frame masks, A only for sparse
runtime validation/interaction checks (reuse whatever raycast infra
already exists for the CBF safety layer / obstacle detection, rather than
building a second raycast system just for this).

### 4. Frame ↔ mesh ↔ category record

Emit one record per captured frame, structurally similar to the existing
`EpisodeLogger` JSONL pattern — reuse that infra instead of building a
parallel logging system:

```json
{
  "frame_id": "ep003_00042",
  "rgb_path": "frames/ep003_00042.png",
  "instance_mask_path": "masks/ep003_00042_instance.png",
  "objects": [
    {"mesh_id": 1024, "category": "small_rock", "pixel_count": 812, "bbox": [x, y, w, h]},
    {"mesh_id": 1026, "category": "hole_in_ground", "pixel_count": 3021, "bbox": [x, y, w, h]}
  ],
  "camera_pose": {"...": "..."}
}
```

### 5. Mask / dataset export

- Collapse instance mask → category mask (N classes + background) via the
  `mesh_id_map` lookup. This is the actual training target.
- Export in a standard format rather than a bespoke loader:
  - COCO-style instance/panoptic JSON, **or**
  - Plain PNG category masks (simplest; compatible with torchvision,
    mmsegmentation, etc.)
- Keep the instance-level mask in addition to the category mask — cheap to
  store now, useful later if instance segmentation (not just semantic) is
  ever needed. Don't discard it after collapsing to category.

### 6. Sanity checks before scaling up

- Manually spot-check ~20 frames: does the projected mask actually align
  with the RGB rock silhouette?
- Check class imbalance early — bedrock/ground will dominate pixel count
  vs. small rocks. This matters for loss weighting at training time, so
  it's worth knowing before generating the full dataset, not after.
- Decide a concrete tolerance for hull-overshoot noise (e.g. "boundary
  drift under N px is fine") so "small negligible noise" has an actual
  definition before it needs debugging later.

### 7. LoRA finetuning (segmentation model)

`sam_vla.perception.finetune_sam2_lora` closes the loop: it finetunes the
SAM2-backed segmentation model that `sam_vla.perception.sam_segmenter`
actually runs at inference time (`sam2_custom_head.SimpleSAM2Seg` -- SAM2.1
Hiera-L's image encoder as a feature backbone + a small from-scratch conv
head predicting dense per-pixel class logits directly, no
points/boxes/mask prompts, no video/memory-bank state -- run_segmentation_sweep
samples independent camera poses, not sequences). There's no promptable
mask decoder anywhere in this pipeline's SAM2 usage to LoRA-adapt, so LoRA
targets the one pretrained component actually in play: the Hiera trunk's
attention projections (`attn.qkv` / `attn.proj`). The task head is
random-init and always trained in full regardless of encoder mode.

Reads `masks_category/<frame_id>.png` + `segmentation_frames.jsonl` +
`summary.json` straight out of a `run_segmentation_sweep` run directory --
the same dense per-class-index masks Step 5 already writes, before
`export_annotations` ever runs -- rather than round-tripping through
`export_annotations`' COCO/YOLO output (built for box/polygon consumers,
not dense-mask training). A COCO-polygon-rasterizing dataset path exists
as a fallback for when only a portable export (no raw run_dir) is
available; YOLO/YOLO-seg aren't supported as inputs here (no mask info /
redundant with COCO).

Needs `torch`/`peft`/`cv2` only -- no `habitat_sim` -- so it runs in the
`sam2` conda env (already used for SAM2 work here, see `sam2.yml`),
decoupled from the `habitat` env the sweep stage needs, same split as
`run_dataset_pipeline.py` shelling the sweep out to a separate
interpreter.

```
python -m sam_vla.perception.finetune_sam2_lora \
    --run-dir output/<run_id> --out-dir sam_lora_runs/exp1 \
    --encoder-mode lora --lora-rank 8 --lora-alpha 16 \
    --epochs 20 --batch-size 8
```

Saves LoRA adapter weights (`peft`'s own `save_pretrained` output) and the
task head separately under `<out-dir>/best/` and `<out-dir>/final/` --
never merged into the base checkpoint, so the base SAM2 checkpoint stays
swappable. Loss is dice + focal (multi-class generalization of SAM2's own
reference training loss, `sam2/training/loss_fns.py`), weighted 20:1
focal:dice matching Meta's own `sam2.1_hiera_b+_MOSE_finetune.yaml`.

**Manual / TODO, not covered by this module:**
- Wiring a trained LoRA adapter into `sam_weights_loader.load_sam_model` /
  `sam_segmenter` for actual rollout use -- currently a separate,
  deliberate step (`finetune_sam2_lora.load_finetuned_model` is the
  loading-side reference, not yet called from the production path).
- Hyperparameter sweep (rank/alpha/lr) and a held-out eval harness beyond
  per-epoch val mIoU -- only a train/val split from one run is done here,
  no cross-run generalization check.
- `--encoder-mode full` (full encoder finetune, no LoRA) and `frozen`
  (decoder-only, matching `sam/train_sam2_simple_fast.py`'s original
  default) exist as CLI options for comparison but weren't benchmarked
  against `lora` here.

---

## Suggestions / Open Items

- **Engine choice for the ID buffer:** confirm Habitat-Sim's semantic
  sensor output format (per-pixel instance ID vs. per-pixel semantic
  class) before writing any raycast code — this single check could remove
  an entire pipeline branch (Step 3A as primary method).
- **Mesh ID stability:** verify IDs survive scene reload/re-export before
  building the registry around them. If not stable, assign IDs manually
  via custom mesh properties at authoring time instead of trusting
  runtime-assigned IDs.
- **Reuse over rebuild:** both the per-frame record format (Step 4) and
  the raycast infra (Step 3A) have existing counterparts in the `sam_vla`
  codebase (`EpisodeLogger`, CBF safety layer raycasts) — extend those
  rather than writing parallel systems.
- **Storage:** keep both instance and category masks per frame. Instance
  masks are cheap to store and expensive to regenerate later if only
  category masks were kept.
- **Noise tolerance:** define the acceptable hull-overshoot threshold
  explicitly (even a rough pixel/percentage number) so it's a known
  parameter rather than an undefined "handle it later."
