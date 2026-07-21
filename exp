The Belief System in navdp/
"Belief" here means a persistent, probabilistic estimate of where a navigation goal is, expressed in the robot's local body frame, that survives even when the goal is not currently visible. It's a classic Bayesian filter (mean + covariance) wrapped in machine-learning infrastructure so a diffusion policy can consume it as conditioning. Everything else in this system — occupancy forecasting, safety CBFs, route management, semantic priors, VLM grounding — is built to feed into or read out of this one belief representation.

I'll go layer by layer: state → refinement → encoding → policy integration → runtime control → data/training.

1. The core state: SubgoalBeliefBank
belief_bank.py

This is a plain NumPy Kalman-filter-like tracker, one BeliefSlot per named goal ("beanbag", "cabinet", etc.):


BeliefSlot: goal_id, mu[2], Sigma[2,2], visible, initialized, last_seen_step, time_since_seen, confidence
Each call to bank.update(observations, odom_delta, step) does one of three things per slot (belief_bank.py:99-134):

Seen → snap mu to the measured position, Sigma to a tight sigma_visible (0.05), reset time_since_seen=0, confidence=1.
Not seen, but previously initialized → dead-reckon: mu is transformed by the inverse SE(2) odometry (ego_motion_update, belief_bank.py:193-203), Sigma grows by +odom_noise each step, confidence decays geometrically, time_since_seen increments.
Never seen → stays at a huge, uninformative Sigma (large_uncertainty=1000).
This means: as the robot moves and the goal drops out of the camera FOV, the belief mean keeps pointing at the goal's last-known location by literally rotating/translating it against the robot's own motion — so the robot can still turn back toward it.

bank.as_tensor(...) (belief_bank.py:136-175) flattens all slots into the raw 11-dim feature layout used everywhere downstream:


[mu_x, mu_y, Sigma_xx, Sigma_xy, Sigma_yy,
 visible, initialized, time_since_seen, confidence,
 is_active, route_index_normalized]
This 11-dim vector-per-goal is the belief interchange format between every module below.

2. Neural refinement: RelationalBelief
relational_belief.py

The raw Kalman belief above is purely geometric/hand-designed. RelationalBelief is a small permutation-equivariant Transformer that takes the [B, N, 11] bank tensor for all goals at once and outputs a refined 13-dim tensor:


[mu_x, mu_y, Sigma_xx, Sigma_xy, Sigma_yy,
 visible, initialized, time_since_seen, confidence,
 is_active, route_index_normalized,
 sigma_ale_mean, sigma_epi_mean]
Key design points:

Zero-init correction: delta_head weights start at zero, so at initialization the module is an exact identity on mu (relational_belief.py:91-92). The correction is gated by correction_gate = initialized * (1 - visible*confidence) * (1 - exp(-recency/5)) (relational_belief.py:141) — i.e., it's only allowed to nudge the mean when the belief is stale/occluded, never when a fresh sensor reading is trusted.
Two separate uncertainty heads: sigma_ale (aleatoric — sensor/dynamics noise, initialized to equal the Kalman Sigma) and sigma_epi (epistemic — "how much should the network itself distrust this," near-zero at init). This split is what later feeds the EpistemicGate (section 6).
Self-supervised training via occlusion_dropout/reconstruction_loss (relational_belief.py:176-238): randomly hides visible slots, trains the network to reconstruct mu (Gaussian NLL, supervising sigma_ale) and to have sigma_epi regress the actual reconstruction error — i.e., epistemic uncertainty is calibrated against real mistakes, not just guessed.
This is the "smart" layer on top of the "dumb" Kalman filter — it lets goals reason about each other (permutation-equivariant self-attention across the N goal slots) and learn when the Kalman assumption (constant-velocity dead reckoning) is breaking down.

3. Turning belief into policy conditioning: belief_encoder.py
belief_encoder.py

Three pieces:

normalize_belief_features (belief_encoder.py:11-46): squashes the unbounded raw belief entries (mu in metres, Sigma growing unboundedly during long occlusions, time_since_seen in raw frame counts) into roughly [-1,1] via /5, log1p, tanh. This matters because a long occlusion makes Sigma/time_since_seen blow up; feeding that raw into a Linear layer saturates it and produces near-random conditioning for exactly the long-occlusion case that matters most.
BeliefEncoder: MLP + learned per-goal-slot positional embedding + optional Transformer encoder → per-goal condition tokens [B, N, embed_dim], plus one active_belief_token gathered via active_goal_index (or inferred from the is_active flag, feature index 9).
ObstacleMapEncoder / RouteTokenEncoder: siblings that encode the local occupancy grid and the current route-pointer index into tokens the same way.
NavDPConditionAdapter: the glue class actually embedded in the policies. It optionally encodes belief/obstacle-map/route-index and appends those tokens after the base VLM/image condition tokens (torch.cat), never replacing them — so a pretrained image-conditioned policy keeps working even with the belief branch bolted on.
4. Policy integration
There are three different policy architectures that consume belief, at increasing sophistication:

a) model.py (flow-matching NavDP) and model_s2_dit.py (S2DiT)
Both just own a NavDPConditionAdapter and a BeliefConditionedCocosSource and pass belief_tensor/obstacle_map/route_index/active_goal_index straight through encode() → the adapter → concatenated condition tokens (model.py:295-344). This is the "basic" integration: belief is just more tokens for cross-attention.

b) cocos_source.py — BeliefConditionedCocosSource
cocos_source.py
Rather than diffusion always starting from pure Gaussian noise x_source = N(0,1), this predicts a belief-informed source mean from the pooled condition tokens: x_source = alpha*mean + beta*epsilon. Diffusion is cheaper/more accurate when you start closer to the answer — this lets the belief bias where the diffusion sampling starts from.

c) belief_augmented_traj_dit.py — BeliefAugmentedTrajectoryDiT (the most complete integration)
belief_augmented_traj_dit.py

This is a full trajectory-diffusion DiT designed to wrap a frozen, pretrained image-conditioned policy and add belief on top without disturbing the original weights:

DiTCrossAttnBlock (belief_augmented_traj_dit.py:34-99) has two cross-attention paths into the trajectory tokens: the original cross_attn (frozen image/VLM condition) and a new memory_cross_attn gated by a learned tanh(memory_gate) starting at 0 — so belief/route/obstacle tokens start as a complete no-op and only get weight as training justifies it.
refine_belief() runs the raw 11-dim bank tensor through RelationalBelief (section 2) before encoding it.
BeliefAnchoredSource (belief_augmented_traj_dit.py:102-172): predicts both a trajectory source mean and a noise-scale beta from the active belief slot. beta is trained to shrink when the belief is reliable (high confidence, low Sigma) — i.e., diffusion sampling gets a tighter, more confident prior exactly when the belief says it should trust itself.
set_train_stage("adapter"/"top"/"all") (belief_augmented_traj_dit.py:648-679): freeze schedule for warm-start training — you can train only the belief/memory branch while keeping the pretrained backbone frozen, exactly matching the "adapter" scripts described below.
5. Dual-diffusion novelty layer: dual_belief_diffusion.py
dual_belief_diffusion.py

This is a two-stage diffusion stack for when the belief history itself (not just the current slot) needs to be inferred:

Backward/context diffusion (DualHeadConditionedDiT): given the current condition tokens, diffusion-reconstructs a short window of the recent belief trajectory (belief_history_steps) it didn't directly observe, and predicts a log-variance per dimension in the same pass (dual heads: eps_head, logvar_head).
Forward waypoint diffusion: the reconstructed belief-history + its uncertainty are encoded into extra tokens (history_encoder) and appended to conditioning for the actual action-chunk diffusion, whose noise schedule is modulated by AdaptiveNoiseSchedule — higher predicted uncertainty in the belief history → higher injected noise scale in the forward diffusion, so the policy naturally hedges when its own belief-reconstruction is unsure.
This is a research/ablation-grade component for when you want the diffusion policy to explicitly reason about uncertain belief histories, not just the current mean/covariance.

6. Runtime control extensions
These live in navdp/navdp/extensions/ and are the pieces that turn belief numbers into actual robot decisions.

belief_control.py
belief_control.py

EpistemicGate: decides "should the robot stop and scan instead of committing to a waypoint chunk?" It fires if either sigma_epi (from RelationalBelief, index 12 of the refined 13-dim tensor) is above threshold or u_occ (occupancy-forecast uncertainty, see §7) is high — two independent uncertainty sources, either can trigger a scan-in-place. High u_occ alone (without triggering a full scan) also scales down commanded speed via speed_scale_from_u_occ.
strength_from_sigma_ale: maps aleatoric uncertainty to an SDEdit "strength" — used to decide how much noise to re-inject when warm-starting diffusion sampling from a previous trajectory.
build_warm_start_path: builds a straight-line path toward the belief mean, used to warm-start diffusion sampling (warm_x in BeliefAugmentedTrajectoryDiT.sample).
refine_bank_with_model: convenience wrapper calling RelationalBelief at inference time.
route_manager.py
route_manager.py
RouteManager.update() reads the belief mean of the currently active goal and advances the route pointer once the robot is within success_radius of it (route_manager.py:60-78). So the belief bank isn't just conditioning — it's literally what decides "we've arrived, move to the next subgoal."

foresight_gate.py
foresight_gate.py
Ranks diffusion-sampled action candidates using OccupancyForesightHead (§7) as a secondary signal, strictly subordinate to a hard depth-based collision gate — rejected candidates can never be re-admitted (enforced by an assertion, foresight_gate.py:114-116). This is a geometric analogue of the epistemic gate: "predicted-free space is a reason to prefer a path, never permission to ignore real depth."

semantic_prior.py
semantic_prior.py
Before the goal has ever been seen, SemanticPrior.predict() turns currently-visible anchor objects (e.g. seeing a stove implies a kitchen implies a fridge nearby) into a broad Gaussian prior over goal position, using an AffinityTable of co-occurrence weights. seed_belief_bank() injects this into a BeliefSlot — but only if the slot has never been initialized, so a real metric sighting always overrides the coarser semantic guess. This is literally seeding the Kalman filter's prior from vision-language semantics instead of leaving it at "totally uninformative."

system2_pixel_goal.py
system2_pixel_goal.py
A frozen VLM (QwenVLPixelGoal) grounds a language instruction to a pixel goal every every steps (System2Scheduler). That pixel is rendered into a goal-mask blob fed into the same pipeline slot a ground-truth segmentation mask would fill. Between VLM calls, the goal mask is empty and the belief bank propagates the last grounded pixel by odometry — i.e., the belief/occlusion machinery is explicitly reused as the async bridge between a slow VLM and a fast control loop.

ghost_geometry.py
ghost_geometry.py
Pure math (no torch) to project a belief's bearing/range estimate into pixel space so it can be rendered as a "ghost" overlay circle on the current image — a visualization/debug tool for occluded-goal belief.

safe_diffusion.py
safe_diffusion.py
CBF (control-barrier-function) safety guidance for diffusion sampling. Belief connects here via mu_goal in cbf_horizon_cost/build_cbf_guidance: the goal-attraction term steers the sampled trajectory back toward the belief-predicted goal position, independent of any currently-visible obstacle — so even after an obstacle has been avoided and left the mask, the guidance keeps pulling the robot back toward what the belief says is the goal (this is called out explicitly as compensating for the "bare policy tends to forget the belief" failure mode, safe_diffusion.py:442-447).

7. The occupancy analogue: occupancy_foresight.py
occupancy_foresight.py

Not "belief" about a goal, but the same pattern applied to geometry: predicting the next egocentric occupancy grid, with an aleatoric uncertainty (u_occ) that plays the same functional role as sigma_ale in the goal belief. Key mechanism: egomotion_warp() analytically warps the current occupancy grid by the candidate action's SE(2) motion (the same transform SubgoalBeliefBank.ego_motion_update uses) — so "propagate what you know by how you moved" is shared physics between goal-belief and occupancy-belief. A small residual conv net only has to learn newly-revealed geometry. u_occ's floor is literally the fraction of the robot's footprint that came from warped-but-unobserved (out-of-bounds) cells — i.e., "how much of my forecast for the space I'm about to drive into is a hallucination." This feeds ForesightGate and EpistemicGate above.

8. Data layer
belief_bank.py is used directly during dataset construction: habitat_route_dataset.py builds a per-episode SubgoalBeliefBank, replays every frame through bank.update() (feeding SAM-mask+depth-derived observations and pose-derived odometry), and caches the resulting [T, 1, 11] belief sequence (_episode_belief_sequence, habitat_route_dataset.py:448-476). couple_belief_dropout deliberately synchronizes goal-mask dropout (simulated occlusion) with the belief that's fed to the policy, so training sees consistent "goal hidden → belief must coast on odometry" episodes rather than a mismatched signal.
route_belief_dataset.py (route_belief_dataset.py) is a generic loader for precomputed .npz/.pkl route-belief training samples (belief_tensor is one of the required fields) — the format the model-conditioning code above expects at training time.
9. Training/eval scripts (in navdp/scripts/)
These represent a deliberate ladder of increasing dependence on the frozen backbone:

gen_belief_propagation_data.py — synthesizes pure belief-space episodes (dead-reckoning + re-observation dynamics, zero rendering/Habitat dependency) with a scripted P-controller as the target-generating policy.
train_belief_only_policy.py / test_belief_only_policy.py — trains a small, freshly-initialized diffusion policy conditioned purely on the propagated belief sequence, with zero dependence on any pretrained image backbone. Purpose: isolate and study belief-propagation dynamics as the sole driver of actions.
train_belief_adapter.py — teaches a frozen pretrained S2DiT policy the "belief-driven return" behavior: appends a small zero-init adapter conditioned on belief bearing, trained against P-controller counterfactual targets, specifically for out-of-view frames where belief is the only cue that can pull the robot back on-screen.
train_vla_adapter.py — the sibling recipe conditioned on language embeddings instead of belief bearing (shares precompute_cond/diffusion_loss with the belief adapter script).
10. Deployment loop: policy_runner.py
policy_runner.py

PolicyRunner.step() (policy_runner.py:120-186) is the real-time loop tying everything together, once per frame:

Compute odometry delta from consecutive poses.
make_belief_observation() (from mask+depth via SAMDepthTargetExtractor) → bank.update(obs, odom, step).
bank.as_tensor(...) → belief_t tensor.
Build the obstacle occupancy grid (DepthObstacleMap).
Optionally build CBF guidance (nearest_obstacle_point, estimate_obstacle_velocity, build_cbf_guidance) — safety, largely independent of belief except via mu_goal if enabled elsewhere.
self.model.sample(..., belief_tensor=belief_t, obstacle_map=obstacle_t, ..., guidance_fn=guidance) — belief flows into the policy exactly through the NavDPConditionAdapter/BeliefAugmentedTrajectoryDiT path described in §3–4.
Mental model summary

SubgoalBeliefBank (Kalman filter, numpy)      -- "where is the goal, and how sure am I"
        │  raw [N,11] tensor
        ▼
RelationalBelief (transformer)                -- refines mu, splits uncertainty into
        │  refined [N,13] tensor                 aleatoric (sensor) vs epistemic (model)
        ▼
BeliefEncoder / NavDPConditionAdapter         -- belief -> extra condition tokens
        │
        ▼
Policy (model.py / model_s2_dit.py /          -- diffusion/flow-matching conditioned on
BeliefAugmentedTrajectoryDiT)                    belief tokens + belief-anchored source/noise
        │
        ▼
Runtime control (EpistemicGate, ForesightGate,-- belief uncertainty -> scan-vs-commit,
RouteManager, safe_diffusion goal-attraction)    route advancement, safety steering
Everything else (occupancy_foresight.py, semantic_prior.py, system2_pixel_goal.py, ghost_geometry.py) either feeds this belief pipeline a better prior/observation, consumes its uncertainty output for a control decision, or visualizes it — but the Kalman-filter-plus-transformer-refinement core described in §1–2 is the single source of truth every other file reads from or writes into.

belief_encoder.py
I am supposed to be playing around, experimenting with different values, mean, covariance, and other stuff, to see which set of params work the best for this belief. All of the belief knowledge should be from within @navdp/ . I do not want any writeovers inside the same directory. if need be, create a new directory named 'belief_exp', which would contain scripts to test parameters and variables. My mentor asked to test specifically about mean, covariance, and others. Plan the implementation before working it out.
Bash Check scripts dir, runs dir, and semantic prior eval script existence
IN
ls -la /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/scripts/
echo "---runs/belief_only_policy---"
find /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/runs -maxdepth 3
echo "---eval_semantic_prior exists?---"
find /home/gpu/Desktop/pineapple/mars-habitatsim -iname "eval_semantic_prior*"

OUT
total 144
drwxrwxr-x 3 gpu gpu  4096 Jul 17 20:23 .
drwxrwxr-x 6 gpu gpu  4096 Jul 17 18:36 ..
-rw-rw-r-- 1 gpu gpu  7608 Jul 17 17:03 gen_belief_propagation_data.py
drwxrwxr-x 2 gpu gpu  4096 Jul 17 20:23 __pycache__
-rw-rw-r-- 1 gpu gpu 75157 Jul 17 14:44 rollout_habitat_policy.py
-rw-rw-r-- 1 gpu gpu  5448 Jul 17 20:11 test_belief_only_policy.py
-rw-rw-r-- 1 gpu gpu  6852 Jul 17 16:05 train_belief_adapter.py
-rw-rw-r-- 1 gpu gpu 16763 Jul 17 17:06 train_belief_only_policy.py
-rw-rw-r-- 1 gpu gpu 11333 Jul 17 14:44 train_vla_adapter.py
---runs/belief_only_policy---
/home/gpu/Desktop/pineapple/mars-habitatsim/navdp/runs
/home/gpu/Desktop/pineapple/mars-habitatsim/navdp/runs/belief_only_policy
/home/gpu/Desktop/pineapple/mars-habitatsim/navdp/runs/belief_only_policy/belief_only_policy.pt
---eval_semantic_prior exists?---
Bash Check line counts of relevant scripts
IN
wc -l /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/scripts/gen_belief_propagation_data.py /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/scripts/train_belief_only_policy.py /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/scripts/test_belief_only_policy.py /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/scripts/rollout_habitat_policy.py

OUT
   150 /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/scripts/gen_belief_propagation_data.py
   348 /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/scripts/train_belief_only_policy.py
   124 /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/scripts/test_belief_only_policy.py
  1697 /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/scripts/rollout_habitat_policy.py
  2319 total
Read gen_belief_propagation_data.py (lines 51-150)
Read train_belief_only_policy.py (lines 61-348)
Read test_belief_only_policy.py (lines 51-124)
Bash Find belief bank param usage/defaults in rollout script
IN
grep -n "SubgoalBeliefBank(\|sigma_init\|sigma_visible\|odom_noise\|decay_factor\|large_uncertainty" /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/scripts/rollout_habitat_policy.py

OUT
864:    bank = SubgoalBeliefBank([category], sigma_visible=0.05, odom_noise=0.02)
Bash Inspect belief_adapter.pt checkpoint contents and RelationalBelief usage
IN
cd /home/gpu/Desktop/pineapple/mars-habitatsim/navdp && python3 -c "
import torch
ckpt = torch.load('belief_adapter.pt', map_location='cpu', weights_only=False)
if isinstance(ckpt, dict):
    for k,v in ckpt.items():
        if hasattr(v,'keys'):
            print(k, '-> dict with', len(v), 'keys; sample:', list(v.keys())[:5])
        else:
            print(k, '=', v if not hasattr(v,'shape') else v.shape)
"
echo "---grep for RelationalBelief training/usage in scripts---"
grep -rn "RelationalBelief\|relational_belief" /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/scripts/ 2>/dev/null
echo "---grep for EpistemicGate usage in scripts---"
grep -rln "EpistemicGate" /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/scripts/ 2>/dev/null

OUT
Traceback (most recent call last):
  File "<string>", line 2, in <module>
ModuleNotFoundError: No module named 'torch'
---grep for RelationalBelief training/usage in scripts---
---grep for EpistemicGate usage in scripts---
Bash Check python env and search full repo for EpistemicGate/RelationalBelief usage outside core modules
IN
which python3; python3 -c "import sys; print(sys.executable)"
find / -maxdepth 4 -iname "venv" -o -iname ".venv" 2>/dev/null | grep -v proc | head -5
grep -rln "EpistemicGate\|RelationalBelief" /home/gpu/Desktop/pineapple/mars-habitatsim/navdp/ --include="*.py" | grep -v "/models/\|/extensions/"

OUT
/usr/bin/python3
/usr/bin/python3
/usr/lib/python3.12/venv
Bash Locate a Python environment with torch installed
IN
find / -maxdepth 6 -iname "site-packages" 2>/dev/null | grep -v proc
echo "---conda---"
find / -maxdepth 4 -iname "conda*" 2>/dev/null | grep -v proc
echo "---pip3 show torch---"
pip3 show torch 2>&1 | head -5
echo "---check for other python versions---"
ls /usr/bin/python*
compgen -c python 2>/dev/null | sort -u

OUT
/home/gpu/miniconda3/lib/python3.14/site-packages
/home/gpu/miniconda3/pkgs/ply-3.11-pyhd8ed1ab_3/site-packages
/home/gpu/miniconda3/pkgs/smmap-3.0.5-pyh44b312d_0/site-packages
/home/gpu/miniconda3/pkgs/mypy_extensions-1.1.0-pyha770c72_0/site-packages
/home/gpu/miniconda3/pkgs/zipp-3.23.0-pyhd8ed1ab_0/site-packages
/home/gpu/miniconda3/pkgs/conda-anaconda-telemetry-0.3.0-pyhd3eb1b0_1/site-packages
/home/gpu/miniconda3/pkgs/jeepney-0.7.1-pyhd3eb1b0_0/site-packages
/home/gpu/miniconda3/pkgs/colorama-0.4.6-pyhd8ed1ab_1/site-packages
/home/gpu/miniconda3/pkgs/pip-26.1.2-pyh8b19718_0/site-packages
/home/gpu/miniconda3/pkgs/conda-libmamba-solver-26.6.0-pyh3785b3c_0/site-packages
/home/gpu/miniconda3/pkgs/deprecated-1.2.18-pyhd8ed1ab_0/site-packages
/home/gpu/miniconda3/pkgs/trimesh-4.1.8-pyhd8ed1ab_0/site-packages
/home/gpu/miniconda3/pkgs/gitpython-3.1.45-pyhff2d567_0/site-packages
/home/gpu/miniconda3/pkgs/dataclasses-json-0.6.7-pyhd8ed1ab_1/site-packages
/home/gpu/miniconda3/pkgs/cycler-0.12.1-pyhd8ed1ab_1/site-packages
/home/gpu/miniconda3/pkgs/typing_extensions-4.14.1-pyhe01879c_0/site-packages
/home/gpu/miniconda3/pkgs/packaging-26.2-pyhc364b38_0/site-packages
/home/gpu/miniconda3/pkgs/imageio-2.37.0-pyhfb79c49_0/site-packages
/home/gpu/miniconda3/pkgs/wheel-0.45.1-pyhd8ed1ab_1/site-packages
/home/gpu/miniconda3/pkgs/gitdb-4.0.12-pyhd8ed1ab_0/site-packages
/home/gpu/miniconda3/pkgs/attrs-25.3.0-pyh71513ae_0/site-packages
/home/gpu/miniconda3/pkgs/certifi-2025.8.3-pyhd8ed1ab_0/site-packages
/home/gpu/miniconda3/pkgs/imageio-ffmpeg-0.6.0-pyhd8ed1ab_0/site-packages
/home/gpu/miniconda3/pkgs/setuptools-80.9.0-pyhff2d567_0/site-packages
/home/gpu/miniconda3/pkgs/setuptools-75.3.0-pyhd8ed1ab_0/site-packages
/home/gpu/miniconda3/pkgs/python-installer-1.0.0-pyhd3eb1b0_0/site-packages
/home/gpu/miniconda3/pkgs/pip-26.0.1-pyhc872135_1/site-packages
/home/gpu/miniconda3/pkgs/pip-26.1.2-pyh0d26453_0/site-packages
/home/gpu/miniconda3/pkgs/python-dateutil-2.9.0.post0-pyhe01879c_2/site-packages
/home/gpu/miniconda3/pkgs/marshmallow-3.20.2-pyhd8ed1ab_0/site-packages
/home/gpu/miniconda3/pkgs/wheel-0.47.0-pyhd8ed1ab_0/site-packages
/home/gpu/miniconda3/pkgs/importlib_resources-6.5.2-pyhd8ed1ab_0/site-packages
/home/gpu/miniconda3/pkgs/anaconda-anon-usage-0.8.1-pyhb46e38b_100/site-packages
/home/gpu/miniconda3/pkgs/toml-0.10.2-pyhd8ed1ab_1/site-packages
/home/gpu/miniconda3/pkgs/pip-24.3.1-pyh8b19718_0/site-packages
/home/gpu/miniconda3/pkgs/typing_inspect-0.9.0-pyhd8ed1ab_1/site-packages
/home/gpu/miniconda3/pkgs/wheel-0.45.1-pyhd8ed1ab_0/site-packages
/home/gpu/miniconda3/pkgs/setuptools-83.0.0-pyh332efcf_0/site-packages
/home/gpu/miniconda3/pkgs/six-1.17.0-pyhe01879c_1/site-packages
/home/gpu/miniconda3/pkgs/munkres-1.1.4-pyhd8ed1ab_1/site-packages
/home/gpu/miniconda3/pkgs/pip-25.2-pyh8b19718_0/site-packages
/home/gpu/miniconda3/pkgs/pyparsing-3.2.3-pyhe01879c_2/site-packages
/home/gpu/miniconda3/pkgs/tomli-2.2.1-pyhe01879c_2/site-packages
/home/gpu/miniconda3/pkgs/tqdm-4.67.1-pyhd8ed1ab_1/site-packages
/home/gpu/miniconda3/pkgs/tomli-w-1.2.0-pyhd3eb1b0_0/site-packages
---conda---
/home/gpu/miniconda3/condarc.d
/home/gpu/miniconda3/condabin
/home/gpu/miniconda3/conda-meta
/home/gpu/.cache/conda
/home/gpu/.cache/conda-anaconda-tos
---pip3 show torch---
/bin/bash: line 12: pip3: command not found
---check for other python versions---
/usr/bin/python3
/usr/bin/python3.12
python3
python3.12
pythoncalls-bpfcc
pythonflow-bpfcc
pythongc-bpfcc
pythonstat-bpfcc
Bash List conda environments and check which has torch
IN
source /home/gpu/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda env list 2>/dev/null
echo "---"
for e in $(conda env list 2>/dev/null | grep -v "^#" | awk '{print $1}'); do
  echo "env: $e"
  conda run -n "$e" python -c "import torch; print('torch', torch.__version__)" 2>&1 | tail -1
done

OUT
# conda environments:
#
# * -> active
# + -> frozen
base                     /home/gpu/miniconda3
eloftr                   /home/gpu/miniconda3/envs/eloftr
habitat                  /home/gpu/miniconda3/envs/habitat
sam2                     /home/gpu/miniconda3/envs/sam2
sam3                     /home/gpu/miniconda3/envs/sam3

---
env: base
ERROR conda.cli.main_run:execute(148): `conda run python -c import torch; print('torch', torch.__version__)` failed. (See above for error)
env: eloftr
ERROR conda.cli.main_run:execute(148): `conda run python -c import torch; print('torch', torch.__version__)` failed. (See above for error)
env: habitat
ERROR conda.cli.main_run:execute(148): `conda run python -c import torch; print('torch', torch.__version__)` failed. (See above for error)
env: sam2
torch 2.13.0+cu130
env: sam3
torch 2.10.0+cu128
Bash Inspect belief_adapter.pt checkpoint using sam2 conda env with torch
IN
source /home/gpu/miniconda3/etc/profile.d/conda.sh
conda run -n sam2 python -c "
import torch
ckpt = torch.load('/home/gpu/Desktop/pineapple/mars-habitatsim/navdp/belief_adapter.pt', map_location='cpu', weights_only=False)
if isinstance(ckpt, dict):
    for k,v in ckpt.items():
        if hasattr(v,'keys'):
            print(k, '-> dict with', len(v), 'keys; sample:', list(v.keys())[:8])
        else:
            print(k, '=', v if not hasattr(v,'shape') else v.shape)
"

OUT
adapter -> dict with 5 keys; sample: ['alpha', 'proj.0.weight', 'proj.0.bias', 'proj.2.weight', 'proj.2.bias']
belief_feat_dim = 3
dim = 512
num_tokens = 4
r_scale = 10.0
AskUserQuestion
OUT
Your questions have been answered: "The mean (mu) in SubgoalBeliefBank has no tunable knobs of its own — it's either snapped to a fresh measurement or deterministically dead-reckoned by the reported odometry (verified in belief_bank.py: no constructor param touches mu directly). So 'testing the mean' really means testing how well mu tracks the true goal under different measurement/odometry noise conditions, while 'testing covariance' means testing sigma_init/sigma_visible/odom_noise/decay_factor/large_uncertainty. Should the experiment harness scope include the learned RelationalBelief refinement layer too?"="Kalman bank only (Recommended)", "What should decide which parameter set is 'best'?"="Calibration + task performance (Recommended)", "What scenario data should the sweep run against?"="Synthetic only (Recommended)". You can now continue with these answers in mind.
Write i-am-supposed-to-rosy-metcalfe.md
196 lines
Claude’s Plan i-am-supposed-to-rosy-metcalfe.md
User approved the plan
Update Todos

Create belief_exp/ directory scaffold

Write common.py (sys.path bootstrap, p_controller, ego_motion_true)

Write scenario.py (closed-loop episode simulator using SubgoalBeliefBank/RouteManager)

Write metrics.py (calibration + task performance scoring)

Write sweep.py (paired random-search CLI driver)

Write inspect_one.py (single-config trace CLI)

Write README.md

Run verification steps 1-4
Bash Create belief_exp directory scaffold
IN
mkdir -p /home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/results
ls /home/gpu/Desktop/pineapple/mars-habitatsim/

OUT
annotations
assets
belief_exp
command.txt
conda_env.py
generate_rock_env.py
hm2obj.py
ht_vel_client.py
ht_vel_server.py
kb_teleop.py
labels.txt
log_reader.py
marsyard2022.mtl
marsyard2022_terrain_hm.png
marsyard2022_terrain_texture.png
navdp
obj2glb.py
pix2vid.py
qwen_vlm_client.py
qwen_vlm_server.py
qwen_vlm_smoke_test.py
README.md
rgbd_drive.py
rgbd_test.py
rock_envs
rollout_navdp2.py
rollout_navdp_policy.py
rollout_navdp.py
round1_walk_rgbd.py
run_navdp_rollout.sh
run_qwen_vla_rollout.sh
run_vlm_nav_interactive.sh
sam
sam2_auto_masks_folder.py
sam_annotation_adapter.py
sam_vla
test_goal_pixel_ratio.py
test_qwen_vlm_persistent.py
test_resolve_vlm_selection.py
usage
verify_vlm_nav_setup.py
vlm_nav_demo.py
vlm_nav_interactive.py
vlm_query.py
Good, directory created. Now let's write the modules.

Write common.py
85 lines
Write scenario.py
203 lines
Write metrics.py
79 lines
Write sweep.py
235 lines
Write inspect_one.py
98 lines
Write README.md
110 lines
Now let's run the verification steps.

Update Todos

Create belief_exp/ directory scaffold

Write common.py (sys.path bootstrap, p_controller, ego_motion_true)

Write scenario.py (closed-loop episode simulator using SubgoalBeliefBank/RouteManager)

Write metrics.py (calibration + task performance scoring)

Write sweep.py (paired random-search CLI driver)

Write inspect_one.py (single-config trace CLI)

Write README.md

Run verification steps 1-4
Bash Run inspect_one.py smoke test with default params
IN
source /home/gpu/miniconda3/etc/profile.d/conda.sh
conda run -n sam2 python /home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/inspect_one.py --seed 0 2>&1 | head -80

OUT
  t  vis  pause   true_x   true_y     mu_x     mu_y     err   sig_x   sig_y  conf
---------------------------------------------------------------------------------
  0    Y           3.988    1.177    4.119    1.271   0.161   0.224   0.224  1.00
  1    .           3.950    0.363    4.136    0.547   0.263   0.265   0.265  0.95
  2    .           3.765    0.017    4.077    0.311   0.429   0.300   0.300  0.90
  3    .           3.564   -0.173    3.920    0.078   0.436   0.332   0.332  0.86
  4    .           3.368   -0.220    3.733    0.020   0.437   0.361   0.361  0.81
  5    .           3.179   -0.232    3.499    0.067   0.438   0.387   0.387  0.77
  6    .           2.991   -0.272    3.337    0.096   0.505   0.412   0.412  0.74
  7    .           2.805   -0.328    3.165   -0.012   0.480   0.436   0.436  0.70
  8    .           2.629   -0.321    3.021   -0.057   0.473   0.458   0.458  0.66
  9    .           2.460   -0.289    2.907   -0.192   0.458   0.480   0.480  0.63
 10    .           2.302   -0.183    2.726   -0.150   0.426   0.500   0.500  0.60
 11    .           2.141   -0.101    2.475   -0.143   0.337   0.520   0.520  0.57
 12    Y           1.980   -0.021    2.023    0.049   0.082   0.224   0.224  1.00
 13    Y           1.770   -0.051    1.944   -0.100   0.181   0.224   0.224  1.00
 14    Y           1.561    0.006    1.624   -0.215   0.229   0.224   0.224  1.00
 15    Y           1.346    0.130    1.528   -0.002   0.225   0.224   0.224  1.00
 16    Y           1.136    0.131    1.155    0.068   0.066   0.224   0.224  1.00
 17    Y           0.931    0.093    0.989    0.222   0.142   0.224   0.224  1.00
 18    Y           0.731   -0.020    0.688   -0.094   0.085   0.224   0.224  1.00
 19    Y           0.523    0.029    0.389   -0.111   0.194   0.224   0.224  1.00

RouteManager advanced at step 19; final true dist = 0.524

metrics over 1 episode(s):
        mean_err_visible = 0.1517
       mean_err_occluded = 0.4255
         calibration_nll = -1.9143
         coverage_1sigma = 0.9250
         coverage_2sigma = 1.0000
      coverage_deviation = 0.2878
         mean_final_dist = 0.5236
            advance_rate = 1.0000
   mean_steps_to_advance = 19.0000
      false_advance_rate = 0.0000
              n_episodes = 1.0000
Bash Compare calibration metrics for overconfident vs well-matched sigma_visible
IN
source /home/gpu/miniconda3/etc/profile.d/conda.sh
echo "=== overconfident sigma_visible (1e-4) with real obs noise 0.3 ==="
conda run -n sam2 python /home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/inspect_one.py --sigma-visible 1e-4 --env-obs-noise 0.3 --episodes 30 --seed 1 2>&1 | tail -15
echo
echo "=== well-matched sigma_visible (~0.09, matches obs_noise variance) ==="
conda run -n sam2 python /home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/inspect_one.py --sigma-visible 0.09 --env-obs-noise 0.3 --episodes 30 --seed 1 2>&1 | tail -15

OUT
=== overconfident sigma_visible (1e-4) with real obs noise 0.3 ===

never advanced within max_steps; final true dist = 0.713

metrics over 30 episode(s):
        mean_err_visible = 0.3708
       mean_err_occluded = 0.8504
         calibration_nll = 402.6870
         coverage_1sigma = 0.3880
         coverage_2sigma = 0.5067
      coverage_deviation = 0.7426
         mean_final_dist = 0.6093
            advance_rate = 0.7333
   mean_steps_to_advance = 25.2273
      false_advance_rate = 0.0455
              n_episodes = 30.0000

=== well-matched sigma_visible (~0.09, matches obs_noise variance) ===

never advanced within max_steps; final true dist = 2.009

metrics over 30 episode(s):
        mean_err_visible = 0.3711
       mean_err_occluded = 0.8088
         calibration_nll = -0.9173
         coverage_1sigma = 0.7753
         coverage_2sigma = 0.9742
      coverage_deviation = 0.1123
         mean_final_dist = 1.0962
            advance_rate = 0.5333
   mean_steps_to_advance = 25.7500
      false_advance_rate = 0.0000
              n_episodes = 30.0000
Bash Run small smoke-test sweep (20 configs x 20 episodes)
IN
source /home/gpu/miniconda3/etc/profile.d/conda.sh
time conda run -n sam2 python /home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/sweep.py --configs-n 20 --episodes-per-config 20 --out /home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/results/smoke_test.csv 2>&1 | tail -40

OUT
[1/20] configs evaluated
[2/20] configs evaluated
[3/20] configs evaluated
[4/20] configs evaluated
[5/20] configs evaluated
[6/20] configs evaluated
[7/20] configs evaluated
[8/20] configs evaluated
[9/20] configs evaluated
[10/20] configs evaluated
[11/20] configs evaluated
[12/20] configs evaluated
[13/20] configs evaluated
[14/20] configs evaluated
[15/20] configs evaluated
[16/20] configs evaluated
[17/20] configs evaluated
[18/20] configs evaluated
[19/20] configs evaluated
[20/20] configs evaluated

wrote 20 configs to /home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/results/smoke_test.csv

combined_score calibration_sc     task_score     sigma_init  sigma_visible     odom_noise   decay_factor large_uncertai success_radius sigma_ale_thre
-----------------------------------------------------------------------------------------------------------------------------------------------------
        0.8569         1.5718         0.1419         4.2603         0.0424         0.0187         0.9182      2486.2937         0.3164         0.8724
        0.7982         1.2606         0.3358         0.1262         0.0444         0.0097         0.8973      4486.3655         0.8206         0.6868
        0.7168         0.9653         0.4683         0.2753         0.0257         0.1106         0.8449        88.7318         0.4307         1.2136
        0.5951         0.4049         0.7852         1.1061         0.0112         0.0452         0.8032      1640.1063         0.6102         1.8653
        0.5066         0.6465         0.3666         4.6590         0.0935         0.0010         0.9538      4523.7795         0.6719         0.7074
        0.4099         0.4492         0.3706         3.5155         0.0118         0.0782         0.8827      2283.3789         0.2080         0.7936
        0.3768         0.6199         0.1337         0.1360         0.1285         0.0043         0.9398      3859.8716         0.3015         1.7431
        0.3001         0.5229         0.0773         0.1333         0.0132         0.0998         0.9262       492.1680         0.3308         1.3801
        0.1651        -0.3172         0.6474         0.2163         0.2360         0.0133         0.9968       116.1065         0.9704         1.6217
        0.1139        -0.4215         0.6494         0.1295         0.2688         0.0014         0.8685       362.7166         0.9728         1.1682

Best config, ready to paste:
SubgoalBeliefBank([goal_id], sigma_init=4.26, sigma_visible=0.04242, odom_noise=0.01869, decay_factor=0.9182, large_uncertainty=2486)
RouteManager(route, success_radius=0.3164)
gate.sigma_ale_threshold = 0.8724

real	0m3.845s
user	0m5.125s
sys	0m0.390s
Bash Inspect CSV header and confirm navdp/ untouched via git status
IN
head -3 /home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/results/smoke_test.csv
echo "---"
git status --short

OUT
sigma_init,sigma_visible,odom_noise,decay_factor,large_uncertainty,success_radius,sigma_ale_threshold,mean_err_visible,mean_err_occluded,calibration_nll,coverage_1sigma,coverage_2sigma,coverage_deviation,mean_final_dist,advance_rate,mean_steps_to_advance,false_advance_rate,n_episodes,calibration_score,task_score,combined_score
4.2603388920012355,0.04242355063279966,0.018688552360099586,0.9181909161210206,2486.293650589535,0.31637883054922544,0.8723696398214406,0.21578624201613628,0.8081662122635321,-1.143288657273864,0.7178387650085764,0.9245283018867925,0.06511046312178392,0.598249039798975,0.75,24.533333333333335,0.06666666666666667,20.0,1.5718464993899335,0.14185527435404663,0.8568508868719901
0.12619083378829712,0.044352455840107094,0.009748168020621348,0.8972810598198594,4486.365458211075,0.8205529504814628,0.6868289891665958,0.21317715106063231,0.8591658424293092,-0.9536311707447201,0.6857976653696498,0.8998054474708171,0.05779221789883271,0.9498060330748558,0.9,23.11111111111111,0.05555555555555555,20.0,1.2606081694016376,0.3358173878802586,0.7982127786409481
---
?? belief_exp/
All verification checks pass. Let me remove the smoke-test artifact (keep results/ clean) and run the full sweep as the last verification step.

Bash Run full sweep (200 configs x 60 episodes) as final verification
IN
rm /home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/results/smoke_test.csv
source /home/gpu/miniconda3/etc/profile.d/conda.sh
time conda run -n sam2 python /home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/sweep.py --configs-n 200 --episodes-per-config 60 2>&1 | tail -35

OUT
[60/200] configs evaluated
[70/200] configs evaluated
[80/200] configs evaluated
[90/200] configs evaluated
[100/200] configs evaluated
[110/200] configs evaluated
[120/200] configs evaluated
[130/200] configs evaluated
[140/200] configs evaluated
[150/200] configs evaluated
[160/200] configs evaluated
[170/200] configs evaluated
[180/200] configs evaluated
[190/200] configs evaluated
[200/200] configs evaluated

wrote 200 configs to /home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/results/sweep_20260720_153509.csv

combined_score calibration_sc     task_score     sigma_init  sigma_visible     odom_noise   decay_factor large_uncertai success_radius sigma_ale_thre
-----------------------------------------------------------------------------------------------------------------------------------------------------
        0.9720         1.3299         0.6142         0.2049         0.0318         0.0210         0.8285        53.2655         0.5471         1.5482
        0.9392         1.2693         0.6092         0.4413         0.0359         0.0205         0.9300       115.2778         0.5757         1.9851
        0.9113         1.2309         0.5917         0.1822         0.0277         0.0307         0.9489        63.0438         0.4151         0.8012
        0.9105         1.2270         0.5939         0.1592         0.0437         0.0148         0.9298       235.2048         0.3990         1.8249
        0.8922         1.1866         0.5977         0.2976         0.0490         0.0148         0.9930       169.8616         0.7284         1.5436
        0.8570         1.1257         0.5883         1.6199         0.0382         0.0322         0.9861      1171.0895         0.4938         1.8304
        0.8170         1.1508         0.4833         2.6935         0.0513         0.0080         0.8269        84.2189         0.6178         1.1806
        0.8098         1.0477         0.5720         0.2250         0.0577         0.0160         0.8249       414.0410         0.8244         1.4410
        0.7516         0.9827         0.5205         0.8447         0.0559         0.0324         0.8027       105.6469         0.4351         1.3931
        0.7444         0.9497         0.5391         1.1052         0.0355         0.0447         0.8964      4988.5314         0.8208         1.6782

Best config, ready to paste:
SubgoalBeliefBank([goal_id], sigma_init=0.2049, sigma_visible=0.03183, odom_noise=0.02098, decay_factor=0.8285, large_uncertainty=53.27)
RouteManager(route, success_radius=0.5471)
gate.sigma_ale_threshold = 1.548

real	0m23.767s
user	0m25.067s
sys	0m0.370s
Bash Final check: navdp/ untouched, list all belief_exp files
IN
git status --short
echo "---"
find /home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp -type f | sort

OUT
?? belief_exp/
---
/home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/common.py
/home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/inspect_one.py
/home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/metrics.py
/home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/__pycache__/common.cpython-310.pyc
/home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/__pycache__/metrics.cpython-310.pyc
/home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/__pycache__/scenario.cpython-310.pyc
/home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/README.md
/home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/results/sweep_20260720_153509.csv
/home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/scenario.py
/home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/sweep.py
Bash Clean up bytecode cache
IN
rm -rf /home/gpu/Desktop/pineapple/mars-habitatsim/belief_exp/__pycache__

OUT
(Bash completed with no output)
Update Todos

Create belief_exp/ directory scaffold

Write common.py (sys.path bootstrap, p_controller, ego_motion_true)

Write scenario.py (closed-loop episode simulator using SubgoalBeliefBank/RouteManager)

Write metrics.py (calibration + task performance scoring)

Write sweep.py (paired random-search CLI driver)

Write inspect_one.py (single-config trace CLI)

Write README.md

Run verification steps 1-4
Built and verified belief_exp/ (sibling to navdp/, nothing under navdp/ touched — git status confirms only belief_exp/ is new).

Files: common.py (imports the real SubgoalBeliefBank/RouteManager, tiny P-controller), scenario.py (closed-loop episode simulator driving the real classes), metrics.py (calibration + task-performance scoring), sweep.py (paired random-search CLI → leaderboard CSV), inspect_one.py (single-config step trace), README.md (full glossary).

All verification steps passed:

inspect_one.py default run — mu tracks the goal when visible, drifts smoothly and Sigma grows monotonically while occluded, resets to sigma_visible on re-sighting. Matches SubgoalBeliefBank internals exactly.
Calibration sanity check worked as designed: forcing sigma_visible=1e-4 against real observation noise 0.3 gave NLL=402.7 and only 39%/51% coverage (nominal 68%/95%) — badly overconfident. A matched sigma_visible=0.09 gave NLL=-0.92 and 78%/97% coverage — well calibrated. Interesting side-finding worth flagging to your mentor: the overconfident config actually advanced the route faster (task metrics looked better) precisely because it was wrong in a way that looked confident — exactly the failure mode coverage_deviation/false_advance_rate exist to catch, and why the leaderboard reports calibration and task-performance separately rather than one blind number.
Smoke sweep (20×20) and full sweep (200 configs × 60 paired episodes, ~24s) both ran cleanly, produced non-degenerate leaderboards, and wrote complete CSVs to belief_exp/results/.
Run it yourself with:


conda run -n sam2 python belief_exp/inspect_one.py
conda run -n sam2 python belief_exp/sweep.py --configs-n 200 --episodes-per-config 60
(needs a torch-enabled env — sam2/sam3 — because navdp.extensions transitively imports torch; plain python3 won't have it. Details and full param/metric glossary are in belief_exp/README.md.)