Per-Sample Gradient Computation Experiments
======
This repo contains experiments for differentially private stochastic gradient descent via per-sample gradient
computations. For more details, please refer to the report.
## Scripts
**run_vision_experiments.py** is a script for for running various per-gradient methods on the MNIST and CIFAR-10
classification tasks. The models include convolutional and linear layers and contain about 1.8 million parameters.
The script can be run like this:
```
$ python run_vision_experiments.py --dp-mode oprod --batch_size 64 --dataset CIFAR --lr 0.01 --epochs 5
```

**run_language_experiments.py** is a script for for running various per-gradient methods on the IMDB sentiment
classification task. The model includes recurrent and linear layers and contain about 2.7 million parameters.
The script can be run like this:
```
$ python run_language_experiments.py --dp-mode naive-sm --batch_size 32 --lr 0.01 --epochs 3
```

The per-sample gradient methods available are:
* "no-dp" : no differential privacy
* "naive" : naive gradient computation (sequential)
* "naive-sm" : naive memory efficient gradient computation (sequential)
* "multi" : multiple model gradient computation
* "oprod" : outer produced gradient computation (vision only)
* "single-fwd-sm" : single forward memory efficient (vision only)
* "single-fwd-lg" : single forward storing batch gradient (vision only)

## Requirements
* numpy==1.16.2
* torch==1.2.0
* pytorch_memlab==0.0.4
* torchtext==0.5.0
* torchvision==0.4.0

## Acknowledgements
The multiple model and batch gradient computations are from the repo here: https://github.com/owkin/grad-cnns

## Contact
* Judy Hanwen Shen
* e-mail: heyyjudes@outlook.com