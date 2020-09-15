# Unsupervised Domain Adaptation by Uncertain Feature Alignment

Official implementation for our BMVC 2020 paper ["Unsupervised Domain Adaptation by Uncertain Feature Alignment"](https://www.bmvc2020-conference.com/assets/papers/0221.pdf).

## Reproduce results
We use conda for all the library, interpreter and dependency management. You can load the exact library versions with `conda env create --file=environment.yaml --name UFAL`.
Afterwards, modify the `src/globals.py` file to provide logging and dataset paths.
After that, activate the environment with `conda activate UFAL` and run the scripts in the `scripts` folder, e.g. `bash src/scripts/run_visda.sh`.
PyTorch should automatically load the pretrained Imagenet weights from the internet when using an architecture for the first time.

For reproduction of results, we recommend 4x 1080Ti GPUs (NVIDIA driver version 440.100) on Ubuntu 18.04. Please note that the results may vary depending on your setup.


## Datasets

The VisDA test set labels can be downloaded from [here](https://raw.githubusercontent.com/VisionLearningGroup/taskcv-2017-public/master/classification/data/image_list.txt). 
You can then use the script provided in `src/misc/prepare_visda_test.py` to transform the trunk version into a normal dataset of form "domain/class/image.jpg".

For other datasets, just implement a new dataset provider given the templates in `src/datasets/providers.py`.

## Paper

The published version of the paper is available [here](https://www.bmvc2020-conference.com/assets/papers/0221.pdf). Please consider citing us when you use our code.

```
@inproceedings{ringwald2020unsupervised,
    title={{Unsupervised Domain Adaptation by Uncertain Feature Alignment}},
    author={Ringwald, Tobias and Stiefelhagen, Rainer},
    booktitle={{The British Machine Vision Conference (BMVC)}},
    year={2020}
}
```

## Repo structure

* `src/`: contains the Python source code for UFAL.
* `data/`: place for raw data, we use symlinks to the real datasets in this folder. Can be changed in `src/globals.py`.
* `environment.yaml`: Contains library versions to reproduce the results.

