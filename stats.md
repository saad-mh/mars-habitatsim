# SAM2 Training

## Run 1

### command used:

```
python -m sam_vla.perception.finetune_sam2_lora --run-dir output/2026-07-30_160329_segmentation-sweep-goal_grid_obsna_seed0/ --out-dir sam_lora_runs/exp1 --encoder-mode lora --epochs 20
```

### Stats

```
classes: ['background', 'small_rock', 'big_rock', 'bedrock', 'hole_in_ground'] (5)
inputs ->train: 1576 frames, val: 175 frames
encoder_mode=lora, trainable params: 2,291,869 / 214,995,149 (1.07%)

```

### start

```
epoch 1/20 train_loss=1.6588 val_loss=1.0078 val_mIoU=0.5354
```

### end

```
epoch 20/20 train_loss=0.6541 val_loss=0.6616 val_mIoU=0.7995
```

### best

```
epoch 19/20 train_loss=0.6552 val_loss=0.6607 val_mIoU=0.8027
```

## Run 2

### command used:

```
python -m sam_vla.perception.finetune_sam2_lora --run-dir output/2026-07-31_180313_segmentation-sweep-goal_grid_obsna_seed10/ --out-dir sam_lora_runs/exp2 --encoder-mode lora --epochs 20
```

### Stats

```
classes: ['background', 'small_rock', 'big_rock', 'bedrock', 'hole_in_ground'] (5)
inputs -> train: 12238 frames, val: 1359 frames
encoder_mode=lora, trainable params: 2,291,869 / 214,995,149 (1.07%)

```

### start

```
epoch 1/20 train_loss=1.6588 val_loss=1.0078 val_mIoU=0.5354
```

### end

```
epoch 20/20 train_loss=0.6541 val_loss=0.6616 val_mIoU=0.7995
```

### best

```
epoch 19/20 train_loss=0.6552 val_loss=0.6607 val_mIoU=0.8027
```
