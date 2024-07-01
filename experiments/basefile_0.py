#######################
## EXPERIMENT CONFIG ##
#######################
import torch
import sys, os
sys.path.insert(0, os.getcwd())
from experiment import Experiment
from pathlib import Path
import metrics


cfg = {
    # Harware setup
    # Get GPU number either from the terminal or from the file name
    'trg_gpu': sys.argv[-1] if not (sys.argv[-1].endswith('.py') or sys.argv[-1].startswith('-')) else Path(__file__).stem.split('_')[-1],
    # How many CPU threads to use
    'trg_n_cpu': 8,

    # General setup and modes
    # Training task (classification / segmentation / autoencoding)
    'training_task': 'classification',
    # Transfer learning type (sequential / parallel)
    'transfer': 'parallel',
    # Datasets
    'dsets': ['Cifar10', 'Cifar100'],
    # Only used in sequential transfer
    'trg_dset': 'Cifar10',
    # Mode where the dataset is split into class-subsets, then train one head per class (single-class classification) (Only used with classification)
    'ultra_typhon': False,
    # Mode where the dataset is split into super-classes, each head has one super-class that does multi-class classification (Only used with classification)
    'twolevels': False,
    # Split on dataset on to multiples subdatasets, and have one head per each subset
    # Additionally there will be one head for the entire dataset
    # Put 0 if no splitting
    'dset_splits': [0]*3,
    # For k-fold cross validation
    # 0 for no cross validation
    'k_fold': 0,

    # Hyperparameters
    'lrates': {
        # One for each dataset
        'train': [5e-6]*3,
        'spec': [5e-6]*3,
        # Frozen is for sequential train only, when training with frozen feature extractor
        'frozen': [5e-6]*3,
    },
    'dropouts': {
        # First one for the FE, following for the DMs
        'train': [0]*4,
        'spec': [0]*4,
        'frozen': [0]*4,
    },
    'batch_size': {
        'train': 8,
        'spec': 32,
        # Used when evaluate the model, to speed up
        'evaluation': 256,
    },
    'architecture': 'caffe',
    # Fractions (0 to 1) of the total epochs number on when to mutliply batch size as key
    # Multiplicator as value
    # Put nothing for a constant batch size
    # e.g. 'epochs_to_multiply_bs': {1/3: 2, 2/3: 4} multiplies at 1/3 of the epochs the batch size by 2, and at 2/3 again by 4
    'epochs_to_multiply_bs': {},
    # Only for training, since in specialization it trains on all batches
    'nb_batches_per_epoch': 1,
    # Only in ultra typhon, number of "negative" heads on which a batch is trained
    # Put 'all' to train on all heads for one batch
    'n_negative_heads': 1,

    # Only for autoencoding. Some loss functions requires mu and logvar as well (in particular for VAEs)
    # In those cases, make sure the dm returns 3 objects (output, mu, logvar)
    'mu_var_loss': False,

    # Bootstrap parameters
    # Type of initialization. Either 'bootstrap', 'random' or 'load'
    'initialization': 'random',
    # Number of models to test the bootstrap. Ignored if 'initialization' is not 'bootstrap'
    'bootstrap_size': 3,

    # Experiment setup
    # Number of samples we want to train on
    'n_samples': {
        # 0 if we do not want one of these steps
        'train': 10000,
        # BE CAREFUL HERE IT IS STILL THE NUMBER OF EPOCHS
        'spec': 0,
    },
    # Number of points we want on the final graph/plot
    # This corresponds to the number of times we compute metrics on all sets during the run
    'n_points': 4,
    # If evaluation within the epoch is activated
    'eval_within_epoch': False,
    # One per each dataset, or just one to be copied
    # 'loss_functions': [metrics.DiceLoss()],
    'loss_functions': [torch.nn.CrossEntropyLoss()],
    # 'loss_functions': [torch.nn.MSELoss()],
    # 'loss_functions': [torch.nn.BCEWithLogitsLoss()],
    # One per each dataset, or just one to be copied
    'optimizers': [torch.optim.Adam],
    # Metric used to compare models, i.e. which one is the best (also used in bootstrap)
    'opt_metric': 'accuracy',

    # Paths and filenames
    'dsets_path': 'datasets',
    # 'dsets_path' : '../../../preproc',
    # Copying data to RAM once to speed it up
    'ramdir': '/dev/shm',
    # Path of output, where results will be stored
    'out_path': 'results',

    # Max timer for the experiment, once it is above, the experiment will end after x hours
    'time_threshold': 24,
    # Add timestamp to avoid overwriting on the folder (e.g. if we want to repeat the same exp)
    'timestamp': False,
    # If we want to resume the current exp (False for a new experiment)
    # Makes only sense for typhon training
    'resume': False,
    # Experiment name, do not change
    'exp_file': __file__,
}
#######################
if __name__ == '__main__':
    exp = Experiment(cfg)
    # DEBUG: uncaught exceptions drop you into ipdb for postmortem debugging
    import sys, IPython; sys.excepthook = IPython.core.ultratb.ColorTB(call_pdb=True)
    exp.main_run()
