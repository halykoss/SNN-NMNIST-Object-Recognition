# SNN-NMNIST-Object-Detection
An object detection model for NMNIST larger video frame

```
$ python3 train.py --img-dim 128 --input-layer 10080 --resize-max 58 --threshold 0.5 --lr 0.0002 \
    --dataset-aug 2.0 --batch-size 256
```

```
$ python3 test.py --img-dim 128 --resize-max 58 
```

```
$ python3 main.py --img-dim 128 --resize-max 58
```