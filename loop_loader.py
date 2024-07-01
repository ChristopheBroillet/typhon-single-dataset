import numpy as np
import torch
import torchvision
import glob
import copy
from pathlib import Path


class AutoencodingDatasetFolder(torchvision.datasets.DatasetFolder):
    def __init__(self, loader, path):
        self.loader = loader
        self.imgs_path = path
        self.data = list(glob.glob(self.imgs_path + "*[!_mask].npy")) # All non-mask
        self.num_samples = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = self.loader(img_path)
        return img, img


class SegmentationDatasetFolder(torchvision.datasets.DatasetFolder):
    def __init__(self, loader, path):
        self.loader = loader
        self.imgs_path = path
        file_list = glob.glob(self.imgs_path + "*[!_mask].npy") # All non-mask
        self.data = []
        for img in file_list:
            mask = img[:-4] + "_mask.npy"
            self.data.append([img, mask])
        self.num_samples = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, img_mask_path = self.data[idx]
        img = self.loader(img_path)
        img_mask = self.loader(img_mask_path)
        img_mask = img_mask.bool()
        return img, img_mask


class LoopLoader():
    def __init__(self,
            dset_path,
            which, # '['train', 'test', 'val']'
            batch_size,
            training_task,
        ):

        self.dset_path = dset_path
        self.which = which
        self.batch_size = batch_size
        self.training_task = training_task

        if self.training_task == 'classification':
            # For a list of which, we concatenate
            self.ds_folder = torch.utils.data.ConcatDataset([torchvision.datasets.DatasetFolder(
                root=f"{self.dset_path}/{split}",
                extensions="npy",
                loader=image_loader())
                for split in self.which])
        elif self.training_task == 'segmentation':
            self.ds_folder = torch.utils.data.ConcatDataset([SegmentationDatasetFolder(
                path=f"{self.dset_path}/{split}/",
                loader=image_loader())
                for split in self.which])
        elif self.training_task == 'autoencoding':
            self.ds_folder = torch.utils.data.ConcatDataset([AutoencodingDatasetFolder(
                path=f"{self.dset_path}/{split}/",
                loader=image_loader())
                for split in self.which])
        else:
            raise UnrecognizedTaskError("Task is not valid, should be in ['classification', 'segmentation', 'autoencoding']")

        # Used when training, drop_last=True so we always have batches of the wanted size
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.ds_folder,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True)
        self.reload_iter()

        # Used when testing, drop_last=False so we do not miss some samples
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.ds_folder,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True)

    def modify_dataset(self, new_ds_folder):
        self.ds_folder = copy.deepcopy(new_ds_folder)
        # To actually apply the new dataset to the loader
        self.reload_iter()

    def split_loaders_twolevels(self):
        assert len(self.which) == 1, f"To use split loaders, 'which' needs to contain only one element"
        assert self.which[0] == 'train', "Cannot call without train as which"

        # Place loaders from [0, n_superclasses)
        loaders = {}
        data_path = Path(self.dset_path) / self.which[0]
        for superclass_path in data_path.iterdir():
            class_dataset = torchvision.datasets.folder.DatasetFolder(
                root=f"{superclass_path}",
                extensions="npy",
                loader=image_loader(),
                # Transform the label to one-hot (since only used in training)
                target_transform=lambda x: torch.nn.functional.one_hot(torch.tensor(x), num_classes=int(len(list(superclass_path.iterdir())))).float()
            )

            # Use copy/deepcopy to have a new object with new reference
            class_loader = copy.copy(self)
            class_loader.modify_dataset(class_dataset)
            idx_to_class = {}
            class_to_dset = {}
            for class_name, idx in class_dataset.class_to_idx.items():
                idx_to_class[int(idx)] = int(class_name)
                # Used only in "cheating"
                class_to_dset[int(class_name)] = f"{self.dset_path.stem}_{superclass_path.stem}"
            loaders[f"{self.dset_path.stem}_{superclass_path.stem}"] = (class_loader, idx_to_class, class_to_dset)
        return loaders

    def split_loaders(self):
        assert len(self.which) == 1, f"To use split loaders, 'which' needs to contain only one element"

        # Save original method, will be override
        orig_method = copy.deepcopy(torchvision.datasets.folder.find_classes)

        # Place loaders from [0, n_classes)
        loaders = []
        root_path = Path(self.dset_path) / self.which[0]
        for i in range(len(list(root_path.iterdir()))):
            # Overwrite this method when constructing a DatasetFolder
            def find_classes(directory):
                # Need label 1.0 (float for BCELoss) as all samples are from the given class
                return [str(i)], {str(i): 1.0}

            torchvision.datasets.folder.find_classes = find_classes

            class_dataset = torchvision.datasets.folder.DatasetFolder(
                root=f"{root_path}",
                extensions="npy",
                loader=image_loader()
            )

            # Use copy/deepcopy to have a new object with new reference
            class_loader = copy.copy(self)
            # So that we have always a batch of the right size
            class_loader.drop_last = True
            class_loader.modify_dataset(class_dataset)
            loaders.append(class_loader)

        # Replace original find_classes method to avoid problems
        torchvision.datasets.folder.find_classes = orig_method

        return loaders

    def reload_iter(self):
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.ds_folder,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True
        )
        self.loader_iter = iter(self.train_loader)

    def get_batch(self):
        try:
            return next(self.loader_iter)
        except StopIteration:
            self.reload_iter()
            return next(self.loader_iter)


def image_loader():
    # Load an image, convert it to a tensor
    def the_loader(path):
        # Load the data
        ary = np.load(path)
        assert len(ary.shape) == 2 or len(ary.shape) == 3, f'Expect inputs to have 2 or 3 channels, got {len(ary.shape)}'
        # We need a 3rd dimension for "channels", so add one if contains only 2 channels
        if len(ary.shape) == 2:
            ary.shape = (1, *ary.shape) # same as `reshape()` but "inplace"
        # Need to convert to float for matrix multiplications
        # float() is equivalent to float 32-bits since model weights are 32-bits
        tensor = torch.from_numpy(ary).float()
        return tensor
    return the_loader


# ------------------------------------------------------------------------------------------------------------------
# New version of random_split of the new Pytorch version (>2.0)
# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#random_split
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset

def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
# ------------------------------------------------------------------------------------------------------------------


# Main method to call
# type is either 'train', 'spec' or 'bootstrap'
# ultra_typhon is if we need to split 1 dataset into subdataset according the label
# dl for data_loader and ll for loop_loader
def load_data(type, ultra_typhon, twolevels, dsets_names, dset_splits, paths, batch_size, training_task, cuda_device):
    if ultra_typhon:
        bootstrap_dl, train_ll, train_dl, val_dl, test_dl = \
            load_data_ultra_typhon(type, dsets_names, dset_splits, paths, batch_size, training_task, cuda_device)
    elif twolevels:
        bootstrap_dl, train_ll, train_dl, val_dl, test_dl, idx_to_class, full_train_idx_to_class, class_to_dset = \
            load_data_twolevels(type, dsets_names, dset_splits, paths, batch_size, training_task, cuda_device)
        return bootstrap_dl, train_ll, train_dl, val_dl, test_dl, idx_to_class, full_train_idx_to_class, class_to_dset
    else:
        bootstrap_dl, train_ll, train_dl, val_dl, test_dl = \
            load_data_normal(type, dsets_names, dset_splits, paths, batch_size, training_task, cuda_device)

    print(f"> All data loaded")
    return bootstrap_dl, train_ll, train_dl, val_dl, test_dl


def load_data_twolevels(type, dsets_names, dset_splits, paths, batch_size, training_task, cuda_device):
    print(f"> Loading data for twolevels ultra typhon")
    bootstrap_dl = {}
    train_ll = {}
    train_dl = {}
    val_dl = {}
    test_dl = {}
    idx_to_class = {}

    # Only one split: dset_name and _n is the number of super-classes
    dset_name, _n = list(dset_splits.items())[0]

    if type == 'bootstrap':
        # So we can load next loaders properly
        # Use batch size of evaluation
        type = 'evaluation'

        # for dset_name, n in dset_splits.items():
        bootstrap_loader = LoopLoader(
            dset_path=paths['dsets'][dset_name],
            # Use both train and val sets, more data for bootstrap!
            # which=['train', 'val'],
            which=['val'],
            batch_size=batch_size[type],
            training_task=training_task
        )

        bootstrap_dl[dset_name] = bootstrap_loader.test_loader
        print(f">> Bootstrap data loaded for dataset {dset_name} from {paths['dsets'][dset_name]} ({len(bootstrap_loader.ds_folder)} samples)")

    train_loaders = LoopLoader(
        dset_path=paths['dsets'][dset_name],
        which=['train'],
        batch_size=batch_size[type],
        training_task=training_task
    ).split_loaders_twolevels()

    # This will only be used to compute performance on the training set
    full_train_loader = LoopLoader(
        dset_path=paths['dsets'][dset_name],
        which=['full_train'],
        batch_size=batch_size['evaluation'],
        training_task=training_task
    )

    val_loader = LoopLoader(
        dset_path=paths['dsets'][dset_name],
        which=['val'],
        batch_size=batch_size['evaluation'],
        training_task=training_task
    )

    test_loader = LoopLoader(
        dset_path=paths['dsets'][dset_name],
        which=['test'],
        batch_size=batch_size['evaluation'],
        training_task=training_task
    )

    class_to_dset_list = []
    for dset in dsets_names:
        # Put loaders for each class-subsets
        train_loop_loader, this_idx_to_class, class_to_dset = train_loaders[dset]
        train_ll[dset] = train_loop_loader
        idx_to_class[dset] = this_idx_to_class
        class_to_dset_list.append(class_to_dset)
        print(f""">> Data loaded for dataset {dset} from {paths['dsets'][dset_name]}
            train: {len(train_loop_loader.ds_folder)} samples
        """)

    train_dl[dset_name] = full_train_loader.test_loader
    val_dl[dset_name] = val_loader.test_loader
    test_dl[dset_name] = test_loader.test_loader

    print(f""">> Data loaded for dataset {dset_name} from {paths['dsets'][dset_name]}
        train: {len(full_train_loader.ds_folder)} samples
        validation: {len(val_loader.ds_folder)} samples
        test: {len(test_loader.ds_folder)} samples
    """)

    from itertools import chain
    # This is to map the class to the containing superclass (used only to cheat)
    class_to_dset = dict(chain.from_iterable(d.items() for d in class_to_dset_list))
    # This is to map the index of the full train set to the real label
    full_train_class_to_idx = full_train_loader.ds_folder.datasets[0].class_to_idx
    full_train_idx_to_class = {}
    for class_name, idx in full_train_class_to_idx.items():
        full_train_idx_to_class[int(idx)] = int(class_name)

    return bootstrap_dl, train_ll, train_dl, val_dl, test_dl, idx_to_class, full_train_idx_to_class, class_to_dset


def load_data_ultra_typhon(type, dsets_names, dset_splits, paths, batch_size, training_task, cuda_device):
    print(f"> Loading data for ultra typhon")
    bootstrap_dl = {}
    train_ll = {}
    train_dl = {}
    val_dl = {}
    test_dl = {}

    # Only one split: dset_name and n is the number of classes
    dset_name, n = list(dset_splits.items())[0]

    if type == 'bootstrap':
        # So we can load next loaders properly
        # Use batch size of evaluation, since we do not train in bootstrap
        type = 'evaluation'

        bootstrap_loader = LoopLoader(
            dset_path=paths['dsets'][dset_name],
            # Use both train and val sets, more data for bootstrap!
            # which=['train', 'val'],
            which=['val'],
            batch_size=batch_size[type],
            training_task=training_task
        )

        bootstrap_dl[dset_name] = bootstrap_loader.test_loader
        print(f">> Bootstrap data loaded for dataset {dset_name} from {paths['dsets'][dset_name]} ({len(bootstrap_loader.ds_folder)} samples)")

    # Data loader from the train set split by class (only used in training)
    train_loaders = LoopLoader(
        dset_path=paths['dsets'][dset_name],
        which=['train'],
        batch_size=batch_size[type],
        training_task=training_task
    # Here all labels will be 1.0
    ).split_loaders()

    # Full training set, used in the evaluation
    full_train_loader = LoopLoader(
        dset_path=paths['dsets'][dset_name],
        which=['train'],
        batch_size=batch_size['evaluation'],
        training_task=training_task
    )

    val_loader = LoopLoader(
        dset_path=paths['dsets'][dset_name],
        which=['val'],
        batch_size=batch_size['evaluation'],
        training_task=training_task
    )

    test_loader = LoopLoader(
        dset_path=paths['dsets'][dset_name],
        which=['test'],
        batch_size=batch_size['evaluation'],
        training_task=training_task
    )

    for i in range(n):
        class_name = f"{dset_name}_{i}"
        # Put loaders for each class-subsets
        train_ll[class_name] = train_loaders[i]
        print(f""">> Data loaded for dataset {class_name} from {paths['dsets'][dset_name]}
        train: {len(train_loaders[i].ds_folder)} samples
        """)

    train_dl[dset_name] = full_train_loader.test_loader
    val_dl[dset_name] = val_loader.test_loader
    test_dl[dset_name] = test_loader.test_loader
    print(f""">> Data loaded for dataset {dset_name} from {paths['dsets'][dset_name]}
    train: {len(full_train_loader.ds_folder)} samples
    validation: {len(val_loader.ds_folder)} samples
    test: {len(test_loader.ds_folder)} samples
    """)

    return bootstrap_dl, train_ll, train_dl, val_dl, test_dl


def load_data_normal(type, dsets_names, dset_splits, paths, batch_size, training_task, cuda_device):
    print(f"> Loading normal data")
    train_ll = {}
    train_dl = {}
    val_dl = {}
    test_dl = {}
    bootstrap_dl = {}

    if type == 'bootstrap':
        # So we can load next loaders properly
        # Use batch size of evaluation
        type = 'evaluation'

        for dset_name, n_split in dset_splits.items():
            bootstrap_loop_loader = LoopLoader(
                dset_path=paths['dsets'][dset_name],
                # Use both train and val sets, more data for bootstrap!
                # which=['train', 'val'],
                which=['val'],
                batch_size=batch_size[type],
                training_task=training_task
            )

            # Data for the head of the entire dataset
            bootstrap_dl[dset_name] = bootstrap_loop_loader.test_loader
            print(f">> Bootstrap data loaded for dataset {dset_name} from {paths['dsets'][dset_name]} ({len(bootstrap_loop_loader.ds_folder)} samples)")

            # Data for the heads of the subsets (if there is a split)
            for idx in range(n_split):
                bootstrap_dl[f"{dset_name}_{idx}"] = bootstrap_loop_loader.test_loader
                print(f">> Bootstrap data loaded for dataset {dset_name}_{idx} from {paths['dsets'][dset_name]} ({len(bootstrap_loop_loader.ds_folder)} samples)")

    for dset_name, n_split in dset_splits.items():
        # The 3 full datasets (no split)
        train_loop_loader = LoopLoader(
            dset_path=paths['dsets'][dset_name],
            which=['train'],
            batch_size=batch_size[type],
            training_task=training_task
        )

        validation_loop_loader = LoopLoader(
            dset_path=paths['dsets'][dset_name],
            which=['val'],
            batch_size=batch_size['evaluation'],
            training_task=training_task
        )

        test_loop_loader = LoopLoader(
            dset_path=paths['dsets'][dset_name],
            which=['test'],
            batch_size=batch_size['evaluation'],
            training_task=training_task
        )

        # Loaders for the full datasets heads
        train_ll[dset_name] = train_loop_loader
        train_dl[dset_name] = train_loop_loader.test_loader
        val_dl[dset_name] = validation_loop_loader.test_loader
        test_dl[dset_name] = test_loop_loader.test_loader

        print(f""">> Data loaded for dataset {dset_name} from {paths['dsets'][dset_name]}
            train: {len(train_loop_loader.ds_folder)} samples
            validation: {len(validation_loop_loader.ds_folder)} samples
            test: {len(test_loop_loader.ds_folder)} samples
        """)

        # Split the train dataset into n_split subsets if split is specified
        if n_split != 0:
            splits = [1 / n_split]*n_split
            # For reproducibility
            generator = torch.Generator().manual_seed(42)
            # Actual random split
            dataset_splits = utils.random_split(train_loop_loader.ds_folder, splits, generator)
            for idx, dataset in enumerate(dataset_splits):
                # Loaders for the sub-datasets heads
                sub_train_loop_loader = LoopLoader(
                    dset_path=paths['dsets'][dset_name],
                    which=['train'],
                    batch_size=batch_size[type],
                    training_task=training_task
                )
                # Change the actual dataset here
                sub_train_loop_loader.modify_dataset(dataset)

                train_ll[f"{dset_name}_{idx}"] = sub_train_loop_loader
                train_dl[f"{dset_name}_{idx}"] = sub_train_loop_loader.test_loader
                val_dl[f"{dset_name}_{idx}"] = validation_loop_loader.test_loader
                test_dl[f"{dset_name}_{idx}"] = test_loop_loader.test_loader

                print(f""">> Data loaded for dataset {dset_name}_{idx} from {paths['dsets'][dset_name]}
                    train: {len(sub_train_loop_loader.ds_folder)} samples
                    validation: {len(validation_loop_loader.ds_folder)} samples
                    test: {len(test_loop_loader.ds_folder)} samples
                """)

    return bootstrap_dl, train_ll, train_dl, val_dl, test_dl
