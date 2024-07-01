###########################################
## THIS IS THE GENERIC EXPERIMENT SCRIPT ##
###########################################
# Check folder 'experiments/' to find the actual exps
if __name__ == '__main__':
    print("You should not call this directly. Check folder `experiments`.")
    import sys
    sys.exit()


import os
import datetime
import time
from pathlib import Path
import shutil
import torch
from brutelogger import BruteLogger
import typhon
import utils
import copy
import math
import warnings
import numpy as np


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg

        # Local level/debug config: shorter runs
        # Simply add your `os.uname().nodename` to the list.
        self.is_local_run = os.uname().nodename in ['example_os_name']
        if self.is_local_run:
            self.cfg.update({
                'nb_batches_per_epoch': 1,
                'n_samples': {
                    'train': 10000,
                    'spec': 0,
                },
                'bootstrap_size': 10,
                'n_points': 2,
            })

        assert (not self.cfg['resume']) or (not self.cfg['timestamp']), "Cannot resume experiment with timestamp activated"
        assert (not self.cfg['transfer'] == 'sequential') or (not self.cfg['resume']), "Cannot resume training on sequential"
        assert (self.cfg['k_fold'] == 0) or (self.cfg['k_fold'] >= 3), f"k_fold should be either == 0 or >= 3"
        assert self.cfg['training_task'] in ['classification', 'segmentation', 'autoencoding']
        assert not self.cfg['ultra_typhon'] or (self.cfg['training_task'] == 'classification'), f"ultra_typhon can only be used with classification"
        assert not self.cfg['ultra_typhon'] or (len(self.cfg['dsets']) == 1), f"ultra_typhon can only be used with one dataset"
        assert not self.cfg['ultra_typhon'] or (self.cfg['n_samples']['spec'] == 0), f"ultra_typhon can only be used with Typhon training (no specialization)"
        assert not self.cfg['ultra_typhon'] or not self.cfg['twolevels'], f"Cannot use ultra typhon and the 2 level version"
        assert (not self.cfg['k_fold'] > 0) or (not self.cfg['ultra_typhon']), f"Currently could not use ultra_typhon with cross validation"
        assert self.cfg['ultra_typhon'] or not self.cfg['eval_within_epoch'], f"Cannot eval_within_epoch without ultra_typhon"
        assert (self.cfg['k_fold'] == 0) or (self.cfg['n_samples']['train'] == 0) or (self.cfg['n_samples']['spec'] == 0), \
            f"Cannot do cross validation on train AND spec at the same time"

        # Resolve CPU threads and cuda device
        torch.set_num_threads(self.cfg['trg_n_cpu'])
        if torch.cuda.is_available():
            # Need to have a GPU and to precise it at the end of the experiment file name
            # Or in the terminal after the file name
            # Assertion blocks if we cannot cast to int, i.e. last part of experiment file name is not an int
            assert isinstance(int(self.cfg['trg_gpu']), int), "Please precise your GPU at the end of the experiment file name"
            # Will anyway stop if the index is not available or wrong
            self.cuda_device = f"cuda:{int(self.cfg['trg_gpu'])}"
        # Otherwise just go with CPU
        else:
            self.cuda_device = 'cpu'
            torch.set_num_threads(self.cfg['trg_n_cpu'])

        # To have a new reference in memory
        # It is used for the splitting in some subsets
        # Contains the dataset names without splitting
        self.orig_dsets = copy.deepcopy(self.cfg['dsets'])

        # Number of splits per datasets
        self.dset_splits = {name:n for name, n in zip(self.orig_dsets, self.cfg['dset_splits'])}

        # Get the number of classes (only used in classification)
        self.cfg['n_classes'] = []
        for dset_name in self.orig_dsets:
            self.cfg['n_classes'].append(len(list((Path(self.cfg['dsets_path']) / f"{dset_name}" / 'train').iterdir())))

        # Create sub-datasets name if there is a splitting
        # Save everything in self.cfg['dsets']
        self.cfg['dsets'] = []
        for dset_name in self.orig_dsets:
            assert (self.dset_splits[dset_name] == 0) or (self.dset_splits[dset_name] >= 2), f"dset_splits should be == 0 or >= 2 for dataset {dset_name}"
            assert (not self.cfg['k_fold'] > 0) or (not self.dset_splits[dset_name] > 0), f"Currently could not use k-fold cross validation with splitting at the same time"
            assert (not self.dset_splits[dset_name] > 0) or (not self.cfg['ultra_typhon']), f"Currently could not use ultra_typhon with splitting"
            if self.cfg['twolevels']:
                # n_classes is the number of split for the dataset
                self.dset_splits[dset_name] = self.cfg['n_classes'][0]
                self.cfg['n_classes'] = []
                for path in (Path(self.cfg['dsets_path']) / f"{dset_name}" / 'train').iterdir():
                    self.cfg['dsets'].append(f"{dset_name}_{path.stem}")
                    self.cfg['n_classes'].append(len(list(path.iterdir())))
                # Also set loss function to be BCELoss
                self.cfg['loss_functions'] = [torch.nn.BCEWithLogitsLoss()]
            elif self.cfg['ultra_typhon']:
                assert len(self.cfg['n_classes']) == 1, f"Runtime error"
                # n_classes is the number of split for the dataset
                self.dset_splits[dset_name] = self.cfg['n_classes'][0]
                for i in range(self.dset_splits[dset_name]):
                    self.cfg['dsets'].append(f"{dset_name}_{i}")
                # Then n_classes is set as 1 for ultra_typhon
                self.cfg['n_classes'][0] = 1
                # Also set loss function to be BCELoss
                self.cfg['loss_functions'] = [torch.nn.BCEWithLogitsLoss()]
            else:
                # One head for the full dataset (and if no splitting)
                self.cfg['dsets'].append(f"{dset_name}")
                if self.dset_splits[dset_name] != 0:
                    # We will have 1 head per each subsets if splitting
                    for i in range(self.dset_splits[dset_name]):
                        self.cfg['dsets'].append(f"{dset_name}_{i}")

        if self.cfg['n_negative_heads'] == 'all': self.cfg['n_negative_heads'] = len(self.cfg['dsets']) - 1

        # Create all paths, stored in self.paths
        self.make_paths()

        # Setup logger
        BruteLogger.save_stdout_to_file(path=self.paths['logs'])

        # Duplicate all hyperparameters according the dataset splitting
        if len(self.orig_dsets) != len(self.cfg['dsets']):
            self.duplicate_hyperparams()

        # Set dropout specific to each DMs, and first to FE
        self.dropouts = {}
        for type in self.cfg['dropouts'].keys():
            self.dropouts[type] = [self.cfg['dropouts'][type][0], {name:dropout for name, dropout in zip(self.cfg['dsets'], self.cfg['dropouts'][type][1:])}]

        # Set learning rates specific to each DMs
        self.lrates = {}
        for type in self.cfg['lrates'].keys():
            self.lrates[type] = {name:lrate for name, lrate in zip(self.cfg['dsets'], self.cfg['lrates'][type])}

        # Set loss functions specific to each DMs
        # Check if loss is specified for each dset, otherwise copy it
        assert (len(self.cfg['loss_functions']) == 1 or len(self.cfg['loss_functions']) == len(self.cfg['dsets'])), f"'loss_functions' must be a list with either 1 element, or len(dsets) == {len(self.cfg['dsets'])} elements"
        if len(self.cfg['loss_functions']) != len(self.cfg['dsets']):
            print('> Copying loss functions')
            self.cfg['loss_functions'] = [copy.deepcopy(self.cfg['loss_functions'][0]) for _ in self.cfg['dsets']]
        self.loss_functions = {name:fct for name, fct in zip(self.cfg['dsets'], self.cfg['loss_functions'])}

        # Set optimizers (= optimization algorithm) specific to each DMs
        # Check if optimizer is specified for each dset, otherwise copy it
        assert (len(self.cfg['optimizers']) == 1 or len(self.cfg['optimizers']) == len(self.cfg['dsets'])), f"'optimizers' must be a list with either 1 element, or len(dsets) == {len(self.cfg['dsets'])} elements"
        if len(self.cfg['optimizers']) != len(self.cfg['dsets']):
            print('> Copying optimizers')
            self.cfg['optimizers'] = [copy.deepcopy(self.cfg['optimizers'][0]) for _ in self.cfg['dsets']]
        self.optimizers = {name:optim for name, optim in zip(self.cfg['dsets'], self.cfg['optimizers'])}

        # Number of classes for classification
        self.n_classes = {}
        for dset_name, n in zip(self.cfg['dsets'], self.cfg['n_classes']):
            assert n >= 1, f"n_classes should be >= 1, not the case for dataset {dset_name}"
            self.n_classes[dset_name] = n

        # Compute the number of epochs based on n_samples
        self.cfg['epochs'] = {}
        for type, n_samples in self.cfg['n_samples'].items():
            if type == 'train':
                # In training, n_samples is n_epochs * batches_per_epoch * batch_size(train) * n_datasets
                n_epochs = int(n_samples / (self.cfg['nb_batches_per_epoch'] * self.cfg['batch_size']['train'] * len(self.cfg['dsets'])))
                # Because in ultra typhon or two levels we train one positive and n negative examples per batch per dataset per epoch
                if self.cfg['ultra_typhon'] or self.cfg['twolevels']:
                    n_epochs = int(n_samples / (self.cfg['nb_batches_per_epoch'] * self.cfg['batch_size']['train'] * (self.cfg['n_negative_heads'] + 1)))
            if type == 'spec':
                # In specialization, we still take as the number of epochs
                n_epochs = n_samples
            self.cfg['epochs'][type] = n_epochs

        # Will compute the metrics every x epochs, based on the 'n_points'
        assert self.cfg['n_points'] >= 2, "n_points should be minimum 2"
        self.epochs_to_evaluate = {}
        for type, epochs in self.cfg['epochs'].items():
            if epochs != 0:
                assert self.cfg['n_points'] <= epochs, f"'n_points' should be <= 'epochs' for {type}"
            self.epochs_to_evaluate[type] = np.linspace(0, epochs, num=self.cfg['n_points'], dtype='int')

        # Set mu_var_loss as False if not specified (since it is only for autoencoding)
        if self.cfg.get('mu_var_loss', None) is None:
            self.cfg['mu_var_loss'] = False

        self.train_args = {
            'paths': self.paths,
            'dsets_names': self.cfg['dsets'],
            'n_classes': self.n_classes,
            'ultra_typhon': self.cfg['ultra_typhon'],
            'twolevels': self.cfg['twolevels'],
            'dset_splits': self.dset_splits,
            'k_fold': self.cfg['k_fold'],
            'architecture': self.cfg['architecture'],
            'initialization': self.cfg['initialization'],
            # Ensure at least 1 for initialization
            'bootstrap_size': max(self.cfg['bootstrap_size'], 1),
            'nb_batches_per_epoch': self.cfg['nb_batches_per_epoch'],
            'n_negative_heads': self.cfg['n_negative_heads'],
            'nb_epochs': self.cfg['epochs'],
            'lrates': self.lrates,
            'dropouts': self.dropouts,
            'batch_size': self.cfg['batch_size'],
            'epochs_to_multiply_bs': self.cfg['epochs_to_multiply_bs'],
            'loss_functions': self.loss_functions,
            'optim_class': self.optimizers,
            'opt_metric': self.cfg['opt_metric'],
            'epochs_to_evaluate': self.epochs_to_evaluate,
            'eval_within_epoch': self.cfg['eval_within_epoch'],
            'training_task': self.cfg['training_task'],
            'mu_var_loss': self.cfg['mu_var_loss'],
            'cuda_device': self.cuda_device,
            'resume': self.cfg['resume'],
            # Convert from hours to seconds
            'time_threshold': self.cfg['time_threshold']*3600,
        }

        print(f"> Config loaded successfully for {self.cfg['transfer']} training:")
        # Print the config so it is written in the log file as well
        for key, value in self.train_args.items():
            if key == 'paths': continue
            print(f">> {key}: {value}")


    def make_paths(self):
        # Make Path objects
        self.cfg.update({
            'dsets_path' : Path(self.cfg['dsets_path']),
            'ramdir' : Path(self.cfg['ramdir']),
            'out_path' : Path(self.cfg['out_path']),
            'exp_file' : Path(self.cfg['exp_file']),
        })

        # Copy dataset to ram for optimization
        # The slash operator '/' in the pathlib module is similar to os.path.join()
        dsets_path_ram = self.cfg['ramdir'] / self.cfg['dsets_path']
        if not self.is_local_run:
            import shutil
            # Copy only dsets used!
            for dset_name in self.orig_dsets:
                if not (dsets_path_ram / dset_name).is_dir():
                    print(f"> Copying {dset_name} on to RAM")
                    shutil.copytree(self.cfg['dsets_path'] / dset_name, dsets_path_ram / dset_name)

        # All paths in one place
        if self.cfg['timestamp']:
            # Add timestamp in folder name to avoid duplicates
            experiment_path = self.cfg['out_path'] / f"{self.cfg['exp_file'].stem}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        else:
            experiment_path = self.cfg['out_path'] / f"{self.cfg['exp_file'].stem}"

        assert (not self.cfg['resume']) or experiment_path.is_dir(), ("Folder experiment does not exist, "
            "either run experiment from the beginning or remove timestamp from folder name")

        models_path = experiment_path / 'models'
        self.paths = {
            'experiment' : experiment_path,
            # Brutelogger logs
            'logs' : experiment_path / 'run_logs',
            'dsets' : {d: self.cfg['dsets_path'] / f"{d}" for d in self.orig_dsets}
                            if self.is_local_run else {d: dsets_path_ram / f"{d}" for d in self.orig_dsets},
            # Trained model (no specialization)
            # p for parallel and s for sequential
            'train_model_p' : models_path / 'train_model_p.pth',
            # Best score on a single dataset during parallel training
            'best_models_p' : {d: models_path / f"best_model_{d}_p.pth" for d in self.cfg['dsets']},
            'train_model_s' : models_path / 'train_model_s.pth',
            # Model saved after the "normal training" in sequential training
            'gen_model_s' : models_path / 'gen_model_s.pth',
            # Specialized models
            'spec_models_p' : {d: models_path / f"spec_model_{d}_p.pth" for d in self.cfg['dsets']},
            'spec_models_s' : {d: models_path / f"spec_model_{d}_s.pth" for d in self.cfg['dsets']},
            # bootstrap model
            'bootstrap_model' : models_path / 'bootstrap_model.pth',
            # Plots
            'metrics' : experiment_path / 'run_plot',
            'samples_training' : experiment_path / 'run_plot' / 'samples_training',
            'samples_spec' : experiment_path / 'run_plot' / 'samples_spec'
        }

        # Add path in ultra_typhon to save the model
        if self.cfg['ultra_typhon'] or self.cfg['twolevels']: self.paths['best_models_p'][self.orig_dsets[0]] = models_path / f"best_model_{self.orig_dsets[0]}_p.pth"

        # Create directories
        self.paths['metrics'].mkdir(parents=True, exist_ok=True)
        self.paths['samples_training'].mkdir(parents=True, exist_ok=True)
        self.paths['samples_spec'].mkdir(parents=True, exist_ok=True)
        self.paths['logs'].mkdir(parents=True, exist_ok=True)
        models_path.mkdir(parents=True, exist_ok=True)


    def duplicate_hyperparams(self):
        print('> Duplicating hyperparameters')

        dropouts = copy.deepcopy(self.cfg['dropouts'])
        self.cfg['dropouts'] = {}
        for type in dropouts.keys():
            # First element is untouched (dropout for FE)
            new_dropouts = [dropouts[type][0]]
            # Duplicate dropout for each split
            for idx, n_split in enumerate(self.dset_splits.values()):
                # idx+1 since first element is for FE (untouched)
                # n_split+1 because we have one more head for the full entire dataset
                new_dropouts = new_dropouts + [dropouts[type][idx+1]]*(n_split+1)
            self.cfg['dropouts'][type] = new_dropouts

        lrates = copy.deepcopy(self.cfg['lrates'])
        self.cfg['lrates'] = {}
        for type in lrates.keys():
            new_lrates = []
            for idx, n_split in enumerate(self.dset_splits.values()):
                new_lrates = new_lrates + [lrates[type][idx]]*(n_split+1)
            self.cfg['lrates'][type] = new_lrates

        n_classes = copy.deepcopy(self.cfg['n_classes'])
        self.cfg['n_classes'] = []
        for idx, (n_class, n_split) in enumerate(zip(n_classes, self.dset_splits.values())):
            self.cfg['n_classes'] = self.cfg['n_classes'] + [n_class]*(n_split+1)


    def main_run(self):
        start = time.perf_counter()
        assert self.cfg['transfer'] in ['sequential', 'parallel'], "Please transfer argument must be 'sequential' or 'parallel'"
        # Need this for sequential learning
        if self.cfg['transfer'] == 'sequential':
            assert self.cfg['trg_dset'] == self.cfg['dsets'][0], "Target dataset must be in first position"
        # Copy the experiment.py and exp cfg file in the experiment dir
        shutil.copy2(self.cfg['exp_file'], self.paths['experiment'])
        shutil.copy2('experiment.py', self.paths['experiment'])
        # Copy architectures as well
        shutil.copy2('architectures/' + self.cfg['architecture'] + '_fe.py', self.paths['experiment'])
        shutil.copy2('architectures/' + self.cfg['architecture'] + '_dm.py', self.paths['experiment'])

        self.typhon = typhon.Typhon(**self.train_args)
        # Bootstrap initialization
        if self.cfg['initialization'] == 'bootstrap':
            # Remove old bootstrap file
            if self.paths['bootstrap_model'].is_file():
                print("> Removing old bootstrap model:", self.paths['bootstrap_model'])
                os.remove(self.paths['bootstrap_model'])
            # Initialize new bootstrap model
            self.typhon.bootstrap()
        # Random initialization
        if self.cfg['initialization'] == 'random':
            # Remove old bootstrap file
            if self.paths['bootstrap_model'].is_file():
                print("> Removing old bootstrap model:", self.paths['bootstrap_model'])
                os.remove(self.paths['bootstrap_model'])
            # Initialize new random model
            self.typhon.random_initialization()
        # Security check
        if self.cfg['initialization'] == 'load':
            assert self.paths['bootstrap_model'].is_file(), f"Bootstrap initialization file missing ({self.paths['bootstrap_model']}), please choose another initialization"
            print("> Loading Bootstrap initialization from ", self.paths['bootstrap_model'])
        # Compute time for bootstrap
        checkpoint_bootstrap = time.perf_counter()

        if self.cfg['transfer'] == 'sequential':
            if self.cfg['epochs']['train'] > 0:
                if self.cfg['k_fold'] > 0:
                    self.typhon.cross_validation(self.paths['bootstrap_model'], 'sequential', 'train')
                else:
                    self.typhon.s_train(self.paths['bootstrap_model'])
            # If no train is made, take bootstrap as starting model for spec
            else: self.paths['train_model_s'] = self.paths['bootstrap_model']
            checkpoint_train = time.perf_counter()
            if self.cfg['epochs']['spec'] > 0:
                if self.cfg['k_fold'] > 0:
                    self.typhon.cross_validation(self.paths['train_model_s'], 'sequential', 'spec')
                else:
                    self.typhon.s_specialization(self.paths['train_model_s'])

        if self.cfg['transfer'] == 'parallel':
            if self.cfg['epochs']['train'] > 0:
                if self.cfg['resume']:
                    self.typhon.p_train(self.paths['train_model_p'])
                else:
                    if self.cfg['k_fold'] > 0:
                        self.typhon.cross_validation(self.paths['bootstrap_model'], 'parallel', 'train')
                    else:
                        self.typhon.p_train(self.paths['bootstrap_model'])
            # If no train is made, take bootstrap as starting model for spec
            else: self.paths['train_model_p'] = self.paths['bootstrap_model']
            checkpoint_train = time.perf_counter()
            if self.cfg['epochs']['spec'] > 0:
                if self.cfg['k_fold'] > 0:
                    self.typhon.cross_validation(self.paths['train_model_p'], 'parallel', 'spec')
                else:
                    self.typhon.p_specialization(self.paths['train_model_p'])

        stop = time.perf_counter()
        total_time = stop - start
        bootstrat_time = checkpoint_bootstrap - start
        training_time = checkpoint_train - checkpoint_bootstrap
        spec_time = stop - checkpoint_train
        print()
        print(f"> Bootstrap took {int(bootstrat_time / 3600)} hours {int((bootstrat_time % 3600) / 60)} minutes {bootstrat_time % 60:.1f} seconds")
        print(f"> Training took {int(training_time / 3600)} hours {int((training_time % 3600) / 60)} minutes {training_time % 60:.1f} seconds")
        print(f"> Specialization took {int(spec_time / 3600)} hours {int((spec_time % 3600) / 60)} minutes {spec_time % 60:.1f} seconds")
        print(f"> Experiment ended in {int(total_time / 3600)} hours {int((total_time % 3600) / 60)} minutes {total_time % 60:.1f} seconds")
        utils.print_time('END EXPERIMENT')
