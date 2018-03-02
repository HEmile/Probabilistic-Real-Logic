# LOGIC TENSOR NETWORK

- This repository contains the implementation of LTN, the generated grounded theories, python scripts for baseline and grounded theories evaluation and the PascalPart dataset.
- All the material in the repository is the implementation of the paper *Learning and Reasoning in Logic Tensor Networks: Theory and Application for Semantic Image Interpretation* by Serafini Luciano, Donadello Ivan, d'Avila Garcez Artur.
- Download the repository with, for example, `git clone git@gitlab.fbk.eu:donadello/LTN_ACM_SAC17.git`.
- Move (or before unzip `LTN.zip`) into the `LTN/code` folder.
- Before execute LTN install TensorFlow library https://www.tensorflow.org/.
- You can use/test the trained grounded theories or train a new grounded theory, see how-tos below.

## Structure of LTN folder

- `pascalpart_dataset`: it contains the refined images and annotations (e.g., small specific parts are merged into bigger parts) of pascalpart dataset in pascalvoc style. This folder is necessary if you want to train Fast-RCNN (https://github.com/rbgirshick/fast-rcnn) on this dataset for computing the grounding/features vector of each bounding box.
    - `Annotations`: the annotations in `.xml` format. To see bounding boxes in the images use the pascalvoc devkit http://host.robots.ox.ac.uk/pascal/VOC/index.html.
    - `ImageSets`: the split of the dataset into train and test set according to every unary predicate/class. For further information See pascalvoc format at devkit http://host.robots.ox.ac.uk/pascal/VOC/index.html.
    - `JPEGImages`: the images in `.jpg` format.

- `code`: it contains the data, the output folder and the source code of LTN.
    - `data`: the training set, the test set, a light version of the dataset and the ontology that defines the mereological axioms.
    - `reports`: the output of the evaluation of the baseline and of the grounded theories
    - `training_with_constraints`: the trained grounded theory with mereological constraints.
    - `training_without_constraints`: the trained grounded theory without mereological constraints.

## How to train a grounded theory

```sh
$ python train_with_constraints.py
$ python train_without_constraints.py
```
- Trained grounded theories are in `training_with_constraints` and `training_with_constraints` folders.

## How to test the grounded theories

```sh
$ python test.py
```
- Results are in `reports/model_evaluation/types_partOf_models_evaluation.csv`
- More detailed results are in `reports/model_evaluation/training_without_constraints` and in `reports/model_evaluation/training_with_constraints`

## How to evaluate the baseline
```sh
$ python baseline.py
```
- Results are in `reports/baseline/types_partOf_baseline_evaluation.csv`