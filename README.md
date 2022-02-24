# SNN-NMNIST-Object-Recognition

An object recognition model for NMNIST larger video frame.

![Image example](/img/home.gif "Results example")

## Installation

Use the package manager [conda](https://docs.conda.io/en/latest/) to install this project.

```bash
conda install --file environment.yml
```

## Usage
If you want to generate dataset examples you can run

```bash
$ python3 main.py --img-dim 128 --resize-max 58
```

If you want to train a model you can run 

```bash
$ python3 train.py --img-dim 128 --input-layer 10080 --no-resize --threshold 0.5 --lr 0.002 --dataset-aug 2.0 --batch-size 256 --random-noise --epochs 20
```

If you want to visualize model prediction you can run 

```bash
$ python3 test.py --img-dim 128 --resize-max 58 
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.



