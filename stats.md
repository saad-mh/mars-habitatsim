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

<!-- epoch 1/20 train_loss=1.6588 val_loss=1.0078 val_mIoU=0.5354 -->

```
epoch 1/20 train_loss=1.9283 val_loss=1.1120 val_mIoU=0.5601
```

### end

<!-- epoch 20/20 train_loss=0.6541 val_loss=0.6616 val_mIoU=0.7995 -->

```
epoch 20/20 train_loss=0.6550 val_loss=0.6599 val_mIoU=0.8055
```

### best

<!-- epoch 19/20 train_loss=0.6552 val_loss=0.6607 val_mIoU=0.8027 -->

```
epoch 20/20 train_loss=0.6550 val_loss=0.6599 val_mIoU=0.8055
```

### Confusion Matrix & Eval

```
training took 6670.51 seconds
[eval] confusion matrix on validation set (175 frames):
                  background  small_rock    big_rock     bedrock  hole_in_gr
background       183,309,518           0      37,864           0           0
small_rock                 0           0           0           0           0
big_rock              34,447           0     118,971           0           0
bedrock                    0           0           0           0           0
hole_in_ground             0           0           0           0           0
[eval] overall pixel accuracy (val set): 0.9996

[eval] pixel accuracy on 10 random image(s) from the dataset:
    idx=   275  accuracy=0.9997
    idx=  1165  accuracy=0.9999
    idx=  1735  accuracy=0.9996
    idx=  1643  accuracy=0.9996
    idx=  1564  accuracy=0.9996
    idx=   129  accuracy=0.9999
    idx=   522  accuracy=0.9997
    idx=   241  accuracy=0.9998
    idx=  1014  accuracy=0.9999
    idx=  1558  accuracy=0.9999
[eval] mean accuracy over 10 random image(s): 0.9997
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
epoch 1/20 train_loss=1.0274 val_loss=0.7834 val_mIoU=0.6817
```

### end

```
epoch 20/20 train_loss=0.6541 val_loss=0.6616 val_mIoU=0.7995
```

### best

```
epoch 19/20 train_loss=0.6552 val_loss=0.6607 val_mIoU=0.8027
```
