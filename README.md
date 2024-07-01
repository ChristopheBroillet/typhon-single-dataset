# Single-Dataset Applications of Typhon and Parallel Transfer to Mitigate Overfitting and Improve Sample Efficiency Across Deep Learning
This repository contains the code of the work done during my Master thesis at the University of Fribourg. This is a direct following-up and extension of the original [**Typhon** framework](https://github.com/eXascaleInfolab/typhon). The details of the algorithms and the implementation can be read in the full thesis, *2024_MSc_Christophe_Broillet.pdf*. In its current state, Typhon can be applied to the following tasks:
1. Classification (in standard Typhon, Two Levels Typhon, or Ultra Typhon mode, see *2024_MSc_Christophe_Broillet.pdf* for more details)
2. Segmentation
3. Autoencoding

## Description of the files and folders
The folder *experiments* contains all the files for the different desired experiments. An example file *basefile_0.py* is given. These are the files that need to be called in order to launch the experiments (see the *Running experiments* section below).

The folder *architectures* contains all blocks composing the different architectures for the different tasks. The user can of course add customized ones. See more details on the architectures in the file *architectures/README.md*

The file *experiment.py* is the generic experiment script that launches the main experiment.

The file *architecture_loader.py* handles the loading of the neural network architecture given as parameters.

The file *loop_loader.py* takes care of loading the data in various ways depending on the tasks, and implements the loop loader which continuously gives batch for all datasets.

The file *metrics.py* implements the different metrics and custom loss functions for the experiments.

The file *results.ipynb* is used to check the results of the experiments.

The file *typhon.py* is the core file containing the algorithmic parts of Typhon.

The file *typhon_model.py* handles the Typhon model, and the relations between the feature extractor and the decision makers.

The file *utils.py* contains some utilitary and miscellaneous functions.

## Running experiments
To run the code and the experiments, a *Conda enviroment* must first be setup using
```
conda env create -f environment.yml
```
after cloning this repository.

One can then activate the environment using
```
conda activate typhon
```

The example experiment can finally be run using the following command:
```
python experiments/basefile_0.py
```
Hyperparameters and many other options can be modified in the aformentioned file.

**N.B.** that you can do a shorter run by simply adding your OS name by going into the Experiment class implementation in the *experiment.py* file (line 32).

Finally one can check the results of the experiments by running the *results.ipynb* Jupyter Notebook using
```
jupyter notebook
```