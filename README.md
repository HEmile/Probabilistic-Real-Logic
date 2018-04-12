# PROBABILISTIC REAL LOGIC

- This repository contains the implementation of PRL. The code is adapted from https://gitlab.fbk.eu/donadello/LTN_IJCAI17 which implemented the paper *Learning and Reasoning in Logic Tensor Networks: Theory and Application for Semantic Image Interpretation* by Serafini Luciano, Donadello Ivan, d'Avila Garcez Artur.
- This code implements the paper *Semi-supervised Classification using Differentiable Reasoning* by Emile van Krieken.
- The code is written using Python 3.6 and Tensorflow 1.3.0, though it should work with newer versions.
- This repository is missing the data. You can download it from https://www.dropbox.com/sh/502aq9u537lrmbv/AAAy00eEtQkIIK5Ytt3Bc2Iha/code/data?dl=0 and then extract it to the `code` folder.


## How to train

```sh
$ python train.py
```
- `code/config.py` contains the parameters and experimental setup, see that file for more information. Multiple models can be trained simultaneously. 
- During training, summaries are written to the `code/logging` folder. From that folder, run ```tensorboard --logdir .``` to follow the progress.
- After training, the models are written to `code/models`.

