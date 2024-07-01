import torch
import time
import copy
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from typhon_model import TyphonModel
from loop_loader import load_data, LoopLoader
import utils
import metrics


class Typhon(object):
    def __init__(self,
            paths,
            dsets_names,
            n_classes,
            ultra_typhon,
            twolevels,
            dset_splits,
            k_fold,
            architecture,
            initialization,
            bootstrap_size,
            nb_batches_per_epoch,
            n_negative_heads,
            nb_epochs,
            lrates,
            dropouts,
            loss_functions,
            optim_class,
            opt_metric,
            epochs_to_evaluate,
            eval_within_epoch,
            training_task,
            mu_var_loss,
            batch_size,
            epochs_to_multiply_bs,
            cuda_device,
            resume,
            time_threshold
        ):

        self.paths = paths
        self.dsets_names = dsets_names
        self.n_classes = n_classes
        self.ultra_typhon = ultra_typhon
        self.twolevels = twolevels
        self.dset_splits = dset_splits
        self.k_fold = k_fold
        self.architecture = architecture
        self.initialization = initialization
        self.bootstrap_size = bootstrap_size
        self.nb_batches_per_epoch = nb_batches_per_epoch
        self.n_negative_heads = n_negative_heads
        self.nb_epochs = nb_epochs
        self.lrates = lrates
        self.dropouts = dropouts
        self.loss_functions = loss_functions
        self.optim_class = optim_class
        self.opt_metric = opt_metric
        self.epochs_to_evaluate = epochs_to_evaluate
        self.eval_within_epoch = eval_within_epoch
        self.training_task = training_task
        self.mu_var_loss = mu_var_loss
        self.batch_size = batch_size
        self.epochs_to_multiply_bs = epochs_to_multiply_bs
        self.cuda_device = cuda_device
        self.resume = resume
        self.metrics_plot = pd.DataFrame(columns=['type', 'feature_extractor', 'epoch', 'n_samples', 'dataset', 'trained_on', 'split', 'metric', 'value'])
        self.best_models = {}
        self.best_metrics_dict = {}
        self.nb_dataset = len(self.paths['dsets'])
        # Dict ('train' and 'spec') to have the total time needed to compute the metrics
        self.total_metrics_time = {'train': 0, 'spec': 0}
        # Only used in cross validation
        self.current_test_fold = 0
        self.time_threshold = time_threshold
        self.head_trained_on = None

        if self.ultra_typhon or self.twolevels:
            self.n_more_batches = 0
            self.pointers = {}
            for idx, dset_name in enumerate(self.dsets_names):
                # At the beginning points to next head
                value = (idx + 1) % len(self.dsets_names)
                self.pointers[dset_name] = value

    def increase_pointer(self, dset_name):
        self.pointers[dset_name] = (self.pointers[dset_name] + 1) % len(self.dsets_names)
        if self.dsets_names[self.pointers[dset_name]] == dset_name:
            self.increase_pointer(dset_name)


    # Test the model on a given dataset, according the training task
    @torch.no_grad()
    def test_model(self, model, dset_name, test_data_loader, verbose=False):
        if self.twolevels:
            return self.test_two_levels(model, test_data_loader, verbose)

        # Only for ultra_typhon
        if self.ultra_typhon:
            return self.test_ultra_typhon(model, test_data_loader, verbose)

        # Only for single-class classification
        if (self.training_task == 'classification') and (self.n_classes[dset_name] == 1):
            return self.test_single_classification_model(model, dset_name, test_data_loader, verbose)

        # Only for binary classification
        if (self.training_task == 'classification') and (self.n_classes[dset_name] == 2):
            return self.test_binary_classification_model(model, dset_name, test_data_loader, verbose)

        # Only for multi-class classification
        if (self.training_task == 'classification') and (self.n_classes[dset_name] > 2):
            return self.test_multi_classification_model(model, dset_name, test_data_loader, verbose)

        if self.training_task == 'segmentation':
            return self.test_segmentation_model(model, dset_name, test_data_loader, verbose)

        if self.training_task == 'autoencoding':
            return self.test_autoencoding_model(model, dset_name, test_data_loader, verbose)


    # Only for two levels Typhon, validation and test sets (train is redirected to single-class)
    @torch.no_grad()
    def test_two_levels(self, model, test_data_loader, verbose):
        # This only sets the model to "eval mode" (and disables specific
        # layers such as dropout and batchnorm). Opposite: `model.train()`
        model.eval()
        assert model.training == False, "Model not in eval mode"

        results = {
            # Real label [0, n_classes)
            'labels': torch.tensor([]).to(self.cuda_device),
            # One-hot encoding label with 1 to the correct class and 0 to other
            'one_hot_labels': torch.tensor([]).to(self.cuda_device),
            # Raw unnormalized predictions per batch
            'raw_per_batch': torch.tensor([]).to(self.cuda_device),
            # Sorted version of raw_per_batch, w.r.t. the real classes
            'sorted_raw_per_batch': torch.tensor([]).to(self.cuda_device),
            # All predictions (real classes [0, n_classes))
            'predictions': torch.tensor([]).to(self.cuda_device),
            # All raw output from the network (unnormalized)
            'raw_predictions': torch.tensor([]).to(self.cuda_device),
            # Probabilities for each class
            'probabilities': torch.tensor([]).to(self.cuda_device),
        }

        sigmoid = torch.nn.Sigmoid()

        # Construct mapping from order of activation of the network to the real classes
        sorted_index = np.array([], dtype='uint8')
        for temp_dset_name in self.dsets_names:
            # To correct the order later
            for i in range(len(self.idx_to_class[temp_dset_name])):
                sorted_index = np.append(sorted_index, np.array(self.idx_to_class[temp_dset_name][i], dtype='uint8'))
        # Get the real order of the activations w.r.t. the real classes
        index_in_order = [idx1 for idx1, _idx2 in sorted(list(enumerate(sorted_index)), key=lambda item: item[1])]

        # For each batch
        # "fake_labels" because of course they correspond to an internal mapping in Pytorch
        for inputs, fake_labels in test_data_loader:
            # Send data to GPU if available
            inputs, fake_labels = inputs.to(self.cuda_device), fake_labels.to(self.cuda_device)
            df_scores = pd.DataFrame()
            # Activate FE only once to speed up
            fe_activations = model.forward_fe(inputs)
            for head_idx, temp_dset_name in enumerate(self.dsets_names):
                outputs = model.forward_dm(fe_activations, temp_dset_name)
                # Save activations
                results['raw_per_batch'] = torch.cat((results['raw_per_batch'], outputs), 1)
                # Now get the scores
                probabilities = sigmoid(outputs)
                # Score: max of the sigmoid
                scores, _ = torch.max(probabilities, dim=1)
                df_scores[head_idx] = scores.cpu().numpy()

            # Sort the activations w.r.t. the true real labels
            for i in index_in_order:
                # Unsqueeze adds one dimension in position 1
                prediction_i = results['raw_per_batch'][:,i].unsqueeze(1)
                results['sorted_raw_per_batch'] = torch.cat((results['sorted_raw_per_batch'], prediction_i), 1)
            results['raw_predictions'] = torch.cat((results['raw_predictions'], results['sorted_raw_per_batch']), 0)
            results['probabilities'] = torch.cat((results['probabilities'], sigmoid(results['sorted_raw_per_batch'])), 0)

            # Heads with highest score for the whole batch
            heads_to_activate = df_scores.idxmax(axis=1)
            # For each sample with index sample_idx, we have to activate the head_idx head
            for sample_idx, head_idx in heads_to_activate.items():
                # Because the labels from the batch are not the real labels!
                real_label = self.full_train_idx_to_class[fake_labels[sample_idx].item()]
                label_tensor = torch.tensor([real_label]).to(self.cuda_device)
                results['labels'] = torch.cat((results['labels'].long(), label_tensor), 0)

                # ----------------------------------------------------
                # CHEATING: HEAD_IDX IS SELECTED W.R.T. THE LABEL
                # SHOULD ONLY BE USED TO HAVE A LOOK AT THE PERFORMANCE
                # dset_name_to_activate = self.class_to_dset[real_label]
                # head_idx = self.dsets_names.index(dset_name_to_activate)
                # -----------------------------------------------------

                # Take the best head's activations w.r.t. the score and compute the pred
                # As above the score is the highest activation, this will also return the same prediction
                # But it will change if we decide to change the score
                n_classes = self.n_classes[self.dsets_names[head_idx]]
                head_activations = results['raw_per_batch'][sample_idx, head_idx*n_classes:(head_idx+1)*n_classes]
                _, predicted_idx = torch.max(head_activations, 0)
                # Transform the label of the specific head to the real class
                converted_pred = self.idx_to_class[self.dsets_names[head_idx]][predicted_idx.item()]

                # Append prediction
                pred_tensor = torch.tensor([converted_pred]).to(self.cuda_device)
                results['predictions'] = torch.cat((results['predictions'].long(), pred_tensor), 0)

            # Reset for next batch
            results['raw_per_batch'] = torch.tensor([]).to(self.cuda_device)
            results['sorted_raw_per_batch'] = torch.tensor([]).to(self.cuda_device)

        # Construct one-hot labels: (batch_size, n_classes) -> treated as multi-label classification (ONLY FOR THE LOSS!)
        results['one_hot_labels'] = torch.nn.functional.one_hot(results['labels'], num_classes=sum(self.n_classes.values())).float()

        metrics_test = {}
        for dset_name in self.dsets_names:
            # Get all real classes from this specific dataset (in order 0 to n_classes w.r.t. the internal Pytorch order)
            classes = list(self.idx_to_class[dset_name].values())
            metrics_test[dset_name] = metrics.get_single_head_two_levels(self.loss_functions[dset_name], results, classes=classes)
            metrics_test[dset_name]['overfitting'] = self.compute_overfitting_score(dset_name)

        # For the full model
        dset_name = dset_name.split('_')[0]
        # Take the loss function of any heads
        metrics_test[dset_name] = metrics.get_multiclass_metrics(torch.nn.BCEWithLogitsLoss(), results, is_two_levels=True)
        metrics_test[dset_name]['overfitting'] = self.compute_overfitting_score(dset_name)
        if verbose: utils.print_results(metrics_test[dset_name], 'LEVEL 2 ULTRA TYPHON')

        return metrics_test


    # Only for Ultra Typhon
    @torch.no_grad()
    def test_ultra_typhon(self, model, test_data_loader, verbose=False):
        # This only sets the model to "eval mode" (and disables specific
        # layers such as dropout and batchnorm). Opposite: `model.train()`
        model.eval()
        assert model.training == False, "Model not in eval mode"

        results = {
            # Real label [0, n_classes)
            'labels': torch.tensor([]).to(self.cuda_device),
            # One-hot encoding label with 1 to the correct class and 0 to other
            'one_hot_labels': torch.tensor([]).to(self.cuda_device),
            # Raw unnormalized predictions per batch
            'raw_per_batch': torch.tensor([]).to(self.cuda_device),
            # All raw output from the network (unnormalized)
            'raw_predictions': torch.tensor([]).to(self.cuda_device),
            # All predictions
            'predictions': torch.tensor([]).to(self.cuda_device),
            # Probabilities per batch
            'probabilities_per_batch': torch.tensor([]).to(self.cuda_device),
            # Probabilities for each class
            'probabilities': torch.tensor([]).to(self.cuda_device),
            # Probabilities for each class (after softmax, so they sum up to 1)
            'probabilities_sum_to_1': torch.tensor([]).to(self.cuda_device),
        }

        softmax = torch.nn.Softmax(dim=1)
        sigmoid = torch.nn.Sigmoid()

        # For each batch
        for inputs, labels in test_data_loader:
            # Send data to GPU if available
            inputs, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)
            # Activate FE only once to speed up
            fe_activations = model.forward_fe(inputs)
            # Activate each head
            for temp_dset_name in self.dsets_names:
                # Feed the model and get outputs
                # Raw, unnormalized output
                outputs = model.forward_dm(fe_activations, temp_dset_name)
                results['raw_per_batch'] = torch.cat((results['raw_per_batch'], outputs), 1)
                # Sigmoid to get confidence [0,1] per each class
                probabilities = sigmoid(outputs)
                results['probabilities_per_batch'] = torch.cat((results['probabilities_per_batch'], probabilities), 1)

            # Compute class label that has the highest probability
            _, predictions = torch.max(results['probabilities_per_batch'], 1)
            # Softmax for probabilities to sum to 1
            prob_sum_to_1 = softmax(results['raw_per_batch'])
            results['probabilities_sum_to_1'] = torch.cat((results['probabilities_sum_to_1'], prob_sum_to_1), 0)

            # Aggregate everything
            results['labels'] = torch.cat((results['labels'].long(), labels), 0)
            results['raw_predictions'] = torch.cat((results['raw_predictions'], results['raw_per_batch']), 0)
            results['probabilities'] = torch.cat((results['probabilities'], results['probabilities_per_batch']), 0)
            results['predictions'] = torch.cat((results['predictions'], predictions), 0)
            # Reset for next batch
            results['probabilities_per_batch'] = torch.tensor([]).to(self.cuda_device)
            results['raw_per_batch'] = torch.tensor([]).to(self.cuda_device)

        # Construct one-hot labels: (batch_size, n_classes) -> treated as multi-label classification (ONLY FOR THE LOSS!)
        results['one_hot_labels'] = torch.nn.functional.one_hot(results['labels'], num_classes=len(self.dsets_names)).float()

        metrics_test = {}
        for dset_idx, dset_name in enumerate(self.dsets_names):
            metrics_test[dset_name] = metrics.get_single_head_ultra_typhon_metrics(self.loss_functions[dset_name], results, dset_idx)
            metrics_test[dset_name]['overfitting'] = self.compute_overfitting_score(dset_name)

        # For the full model
        dset_name = dset_name.split('_')[0]
        # Take the loss function of any heads
        metrics_test[dset_name] = metrics.get_multiclass_metrics(self.loss_functions[self.dsets_names[0]], results, is_ultra_typhon=True)
        metrics_test[dset_name]['overfitting'] = self.compute_overfitting_score(dset_name)
        if verbose: utils.print_results(metrics_test[dset_name], 'ULTRA TYPHON')

        return metrics_test


    # Only for single-class classification
    @torch.no_grad()
    def test_single_classification_model(self, model, dset_name, test_data_loader, split, verbose=False):
        # This only sets the model to "eval mode" (and disables specific
        # layers such as dropout and batchnorm). Opposite: `model.train()`
        model.eval()
        assert model.training == False, "Model not in eval mode"

        results = {
            'labels': torch.tensor([]).to(self.cuda_device),
            # Raw output from the network (unnormalized)
            'raw_predictions': torch.tensor([]).to(self.cuda_device),
            # Discrete class predicted (integer)
            'predictions': torch.tensor([]).to(self.cuda_device),
        }

        sigmoid = torch.nn.Sigmoid()

        # For each batch
        for inputs, labels in test_data_loader:
            # Send data to GPU if available
            inputs, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)
            # When computing metrics for the validation set but only on samples of that class
            if split == 'val_eval':
                true_label = int(dset_name.split('_')[1])
                mask = labels == true_label
                indices = np.argwhere(np.asarray(mask.cpu())).flatten()
                inputs = inputs[indices]
                labels = torch.tensor([1.0]*len(inputs)).to(self.cuda_device)
            # Feed the model and get outputs
            # Raw, unnormalized output required to compute the loss (with BCE)
            outputs = model(inputs, dset_name)
            # squeeze() reshape the tensor from shape (batch_size, 1) to (batch_size,)
            outputs = outputs.squeeze(dim=1)
            # Sigmoid to get probabilities
            probabilities = sigmoid(outputs)
            # Turn into label (0 not in the class, 1 in the class) with threshold at 0.5
            predicted = (probabilities > 0.5).int()

            results['predictions'] = torch.cat((results['predictions'], predicted), 0)
            results['raw_predictions'] = torch.cat((results['raw_predictions'], outputs), 0)
            results['labels'] = torch.cat((results['labels'], labels), 0)

        metrics_test = metrics.get_singleclass_metrics(self.loss_functions[dset_name], results)
        metrics_test['overfitting'] = self.compute_overfitting_score(dset_name)

        if verbose: utils.print_results(metrics_test, 'SINGLE-CLASS CLASSIFICATION')

        return metrics_test


    # Only for binary classification
    @torch.no_grad()
    def test_binary_classification_model(self, model, dset_name, test_data_loader, verbose=False):
        # This only sets the model to "eval mode" (and disables specific
        # layers such as dropout and batchnorm). Opposite: `model.train()`
        model.eval()
        assert model.training == False, "Model not in eval mode"

        results = {
            # Labels as a long PyTorch tensor
            'labels': torch.tensor([]).to(self.cuda_device),
            # Raw output from the network (unnormalized)
            'raw_predictions': torch.tensor([]).to(self.cuda_device),
            # Discrete class predicted (integer)
            'predictions': torch.tensor([]).to(self.cuda_device),
            # Probability of being the positive class
            'predictions_positive_class': torch.tensor([]).to(self.cuda_device),
        }

        softmax = torch.nn.Softmax(dim=1)

        # For each batch
        for inputs, labels in test_data_loader:
            # Send data to GPU if available
            inputs, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)
            # Feed the model and get outputs
            # Raw, unnormalized output required to compute the loss (with CrossEntropyLoss)
            outputs = model(inputs, dset_name)
            _, predicted = torch.max(outputs, 1)

            # Probabilities required to compute roc_auc_score, so use a softmax
            proba_classes = softmax(outputs)
            all_positives = torch.index_select(outputs, 1, torch.tensor([1]).to(self.cuda_device))

            results['predictions_positive_class'] = torch.cat((results['predictions_positive_class'], all_positives), 0)
            results['raw_predictions'] = torch.cat((results['raw_predictions'], outputs), 0)
            results['labels'] = torch.cat((results['labels'].long(), labels), 0)
            results['predictions'] = torch.cat((results['predictions'].long(), predicted), 0)

        metrics_test =  metrics.get_binaryclass_metrics(self.loss_functions[dset_name], results)
        metrics_test['overfitting'] = self.compute_overfitting_score(dset_name)
        if verbose: utils.print_results(metrics_test, 'BINARY CLASSIFICATION')

        return metrics_test


    # Only for multi-class classification
    @torch.no_grad()
    def test_multi_classification_model(self, model, dset_name, test_data_loader, verbose=False):
        # This only sets the model to "eval mode" (and disables specific
        # layers such as dropout and batchnorm). Opposite: `model.train()`
        model.eval()
        assert model.training == False, "Model not in eval mode"

        results = {
            # Labels as a long PyTorch tensor
            'labels': torch.tensor([]).to(self.cuda_device),
            # Raw output from the network (unnormalized)
            'raw_predictions': torch.tensor([]).to(self.cuda_device),
            # Discrete class predicted (integer)
            'predictions': torch.tensor([]).to(self.cuda_device),
            # Probabilities for each class
            'probabilities': torch.tensor([]).to(self.cuda_device),
            # Probabilities for each class (after softmax, so they sum up to 1)
            'probabilities_sum_to_1': torch.tensor([]).to(self.cuda_device),
        }

        softmax = torch.nn.Softmax(dim=1)

        # For each batch
        for inputs, labels in test_data_loader:
            # Send data to GPU if available
            inputs, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)
            # Feed the model and get outputs
            # Raw, unnormalized output required to compute the loss (with CrossEntropyLoss)
            outputs = model(inputs, dset_name)
            # Softmax to get probabilities
            proba_classes = softmax(outputs)
            _, predicted = torch.max(proba_classes, 1)

            results['predictions'] = torch.cat((results['predictions'].long(), predicted), 0)
            results['raw_predictions'] = torch.cat((results['raw_predictions'], outputs), 0)
            results['labels'] = torch.cat((results['labels'].long(), labels), 0)
            results['probabilities'] = torch.cat((results['probabilities'], proba_classes), 0)
            results['probabilities_sum_to_1'] = torch.cat((results['probabilities_sum_to_1'], proba_classes), 0)

        metrics_test = metrics.get_multiclass_metrics(self.loss_functions[dset_name], results)
        metrics_test['overfitting'] = self.compute_overfitting_score(dset_name)
        if verbose: utils.print_results(metrics_test, 'MULTI-CLASS CLASSIFICATION')

        return metrics_test


    @torch.no_grad()
    def test_segmentation_model(self, model, dset_name, test_data_loader, verbose=False):
        # This only sets the model to "eval mode" (and disables specific
        # layers such as dropout and batchnorm). Opposite: `model.train()`
        model.eval()
        assert model.training == False, "Model not in eval mode"

        # List of losses
        losses = []
        # List of Hausdorff distances
        hausdorff_distances = []
        confusion_matrix_dict = {}

        # For each batch
        for inputs, labels in test_data_loader:
            # Send data to GPU if available
            inputs, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)
            # Feed the model and get outputs
            # Sigmoid at the end of the model, so output already as probabilities
            outputs = model(inputs, dset_name)
            # Set the values of the output to 0 or 1 (tumor at pixel xy or not) and cast to int
            predicted = (outputs > 0.5).int()
            # Compute loss and Hausdorff distance per batch, to limit memory consumption
            ls = self.loss_functions[dset_name](outputs, labels).item()
            losses.append(ls)
            hd = metrics.hausdorff_dist(predicted, labels)
            hausdorff_distances.append(hd)

            tp = torch.sum((predicted==labels) * (predicted==1)).item()
            tn = torch.sum((predicted==labels) * (predicted==0)).item()
            fp = torch.sum((predicted!=labels) * (predicted==1)).item()
            fn = torch.sum((predicted!=labels) * (predicted==0)).item()
            conf_matrix_per_batch = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}

            for key, value in conf_matrix_per_batch.items():
                confusion_matrix_dict.setdefault(key, []).append(value)

        metrics_test = metrics.get_segmentation_metrics(losses, hausdorff_distances, confusion_matrix_dict)

        if verbose: utils.print_results(metrics_test, 'SEGMENTATION')

        return metrics_test


    @torch.no_grad()
    def test_autoencoding_model(self, model, dset_name, test_data_loader, verbose=False):
        # This only sets the model to "eval mode" (and disables specific
        # layers such as dropout and batchnorm). Opposite: `model.train()`
        model.eval()
        assert model.training == False, "Model not in eval mode"

        # List of losses
        losses = []

        results = {
            # Labels as a long PyTorch tensor
            'labels': torch.tensor([]),
            # Output/predictions of the model
            'predictions': torch.tensor([]),
            # Contain loss per batch
            'losses': torch.tensor([]),
        }

        # For each batch
        for inputs, labels in test_data_loader:
            # Send data to GPU if available
            inputs, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)
            # Feed the model and get outputs
            # Sigmoid at the end of the model, so output already as probabilities
            outputs = model(inputs, dset_name)
            # Compute loss per batch, to limit memory consumption
            # unsqueeze transform the 0-dim tensor to 1-dim tensor, so we can concatenate
            ls = self.loss_functions[dset_name](outputs, labels).unsqueeze(0).cpu()
            results['losses'] = torch.cat((results['losses'], ls), 0)
            # losses.append(ls)
            results['labels'] = torch.cat((results['labels'], labels.cpu().flatten()), 0)
            results['predictions'] = torch.cat((results['predictions'], outputs.cpu().flatten()), 0)

        metrics_test = metrics.get_autoencoding_metrics(results)
        if verbose: utils.print_results(metrics_test, 'AUTOENCODING')

        return metrics_test


    # Load the model from the given model, and set the optimizers
    # type is either 'train' or 'spec'
    def load_model_and_optims(self, model_path, type, frozen=False):
        print(f"> Loading models from {model_path} and optimizers")
        loaded_state_dicts = torch.load(model_path, map_location=self.cuda_device)
        self.dsets_names = loaded_state_dicts['variables']['dsets_names']
        self.model = TyphonModel.from_state_dict(loaded_state_dicts)
        self.model.set_dropout(*self.dropouts[type])
        self.optimizers = {}
        # Send model to GPU if available
        self.model.to(self.cuda_device)
        # Split the model, to be used in specialization
        self.spec_models = self.model.split_typhon()
        for dset_name in self.dsets_names:
            # Send models to GPU if available
            self.spec_models[dset_name].to(self.cuda_device)

        if type == 'train':
            # Additional option for hydra
            if frozen: type = 'frozen'
            for dset_name in self.dsets_names:
                # Here we keep only the parameters of the FE and a specific DM, for each dataset
                params = torch.nn.ParameterList([param for name, param in self.model.named_parameters() if ('fe' in name) or (dset_name in name)])
                optim = self.optim_class[dset_name](params, lr=self.lrates[type][dset_name])
                self.optimizers[dset_name] = optim

        if type == 'spec':
            # Additional option for hydra
            if frozen: type = 'frozen'
            for dset_name in self.dsets_names:
                # Here we keep only the parameters of the FE and a specific DM, for each dataset
                params = torch.nn.ParameterList([param for name, param in self.spec_models[dset_name].named_parameters() if ('fe' in name) or (dset_name in name)])
                optim = self.optim_class[dset_name](params, lr=self.lrates[type][dset_name])
                self.optimizers[dset_name] = optim

        print(f"> Models and optimizers loaded")


    # Train one model on one batch from one dataset
    def train_on_batch(self, model, dset_name, batch):
        assert model.training == True, "Model not in training mode"
        inputs, labels = batch
        # Send data to GPU if available
        inputs, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)
        # Run the model on the batch and get predictions
        predictions = model(inputs, dset_name)
        # Some models (in particular VAEs) returns mu and var together with the output, to compute the loss
        if self.mu_var_loss:
            # Extract mu and var
            predictions, mu, logvar = predictions
            # Compute loss between prediction and labels
            loss = self.loss_functions[dset_name](predictions, labels, mu, logvar)
        else:
            # Handle data for single class classification
            if (self.training_task == 'classification') and (self.n_classes[dset_name] == 1):
                # squeeze() reshape the tensor from shape (batch_size, 1) to (batch_size,)
                predictions = predictions.squeeze(dim=1)
            # Compute loss between prediction and labels
            loss = self.loss_functions[dset_name](predictions, labels)
        # Backpropagation computes dloss/dx for each x param
        loss.backward()
        # Optimizer.step performs a parameter update based on gradients
        self.optimizers[dset_name].step()
        # Clear old gradients (default is to accumulate)
        self.optimizers[dset_name].zero_grad(set_to_none=True)


    def train_step_ultratyphon_twolevelstyphon(self, model, dset_name):
        model.train()
        assert model.training == True, "Model not in training mode"
        for nbatch in range(self.nb_batches_per_epoch):
            # First get the true head with label 1 (positive training)
            batch = self.train_loop_loaders[dset_name].get_batch()
            self.train_on_batch(model, dset_name, batch)
            if self.epoch in self.epochs_to_evaluate['train'] and self.eval_within_epoch:
                self.head_trained_on = dset_name
                self.n_more_batches += 1
                self.p_train_compute_metrics_end_epoch()
                model.train()

            # Then train the pointed head with label 0 (negative training), on n_negative_heads
            inputs, old_labels = batch
            if self.ultra_typhon:
                # Negative label for one sample is 0
                new_labels = torch.zeros(len(old_labels)).to(self.cuda_device)
            if self.twolevels:
                # Negative label for one sample is [0,0,...,0]
                new_labels = torch.zeros(len(old_labels), self.n_classes[dset_name]).to(self.cuda_device)
            new_batch = (inputs, new_labels)
            for _i in range(self.n_negative_heads):
                head_to_train = self.dsets_names[self.pointers[dset_name]]
                self.train_on_batch(model, head_to_train, new_batch)
                self.increase_pointer(dset_name)
                if self.epoch in self.epochs_to_evaluate['train'] and self.eval_within_epoch:
                    self.head_trained_on = head_to_train
                    self.n_more_batches += 1
                    self.p_train_compute_metrics_end_epoch()
                    model.train()


    # train_on is either 'all' or 'some' batch(es)
    def train_step(self, model, dset_name, train_on):
        if self.ultra_typhon or self.twolevels:
            self.train_step_ultratyphon_twolevelstyphon(model, dset_name)
            return

        model.train()
        assert train_on in ['all', 'some'], "train_on must be either 'all' or 'some'"
        if train_on == 'some':
            for nbatch in range(self.nb_batches_per_epoch):
                batch = self.train_loop_loaders[dset_name].get_batch()
                self.train_on_batch(model, dset_name, batch)

        elif train_on == 'all':
            print(f">>> Training on all batches")
            for batch in self.train_data_loaders[dset_name]:
                self.train_on_batch(model, dset_name, batch)


    # Compute metrics for train, val and test sets
    def compute_metrics(self, model, dset_name):
        print(f">>>> Collecting performance on training, validation and test set")
        metrics_training = self.test_model(
            model=model,
            dset_name=dset_name,
            test_data_loader=self.train_data_loaders[dset_name]
        )

        metrics_validation = self.test_model(
            model=model,
            dset_name=dset_name,
            test_data_loader=self.validation_data_loaders[dset_name]
        )

        metrics_test = self.test_model(
            model=model,
            dset_name=dset_name,
            test_data_loader=self.test_data_loaders[dset_name]
        )

        return metrics_training, metrics_validation, metrics_test


    # type is either 'train' or 'spec'
    def compare_models(self, model, dset_name, type, save_path, metrics_validation):
        # At first epoch save the model and the score to have a baseline
        if self.epoch == 0 or dset_name not in self.best_metrics_dict.keys():
            self.best_metrics_dict[dset_name] = copy.deepcopy(metrics_validation)
            self.best_metrics_dict[dset_name]['epoch'] = self.epoch
            self.best_models[dset_name] = copy.deepcopy(model)
            torch.save(self.best_models[dset_name].to_state_dict(), save_path)
            print(f">>>> First model saved: {self.opt_metric}: {self.best_metrics_dict[dset_name][self.opt_metric]}")
            return

        # Compare scores and save model if better
        new_opt = metrics_validation[self.opt_metric]
        best_opt = self.best_metrics_dict[dset_name][self.opt_metric]

        # Metric to minimize to be better
        if self.opt_metric in ['loss', 'hd']:
            found_new_best = new_opt < best_opt
        # Metric to maximize to be better
        else:
            found_new_best = new_opt > best_opt

        if found_new_best:
            print(f">>>> New best: {self.opt_metric}: {best_opt} -> {new_opt}")
            # Setting new best data
            self.best_metrics_dict[dset_name] = copy.deepcopy(metrics_validation)
            self.best_metrics_dict[dset_name]['epoch'] = self.epoch
            # Save model
            self.best_models[dset_name] = copy.deepcopy(model)
            torch.save(self.best_models[dset_name].to_state_dict(), save_path)
            print(f">>>> New best model saved at epoch {self.best_metrics_dict[dset_name]['epoch']},", end=' ')
            print(f"{self.opt_metric}: {self.best_metrics_dict[dset_name][self.opt_metric]}")


    # Save a sample of the current model
    def save_sample(self, path, model, dset_name, epoch):
        # No sample to save in classification
        if self.training_task == 'classification': return
        data_loader = self.test_data_loaders[dset_name]
        # Load 1 batch
        inputs, labels = next(iter(data_loader)) # Access only 1 batch
        inputs, labels = inputs.to(self.cuda_device), labels.to(self.cuda_device)
        # Pass to model
        outputs = model(inputs, dset_name)
        # Some models (in particular VAEs) returns mu and var together with the output, to compute the loss
        if self.mu_var_loss:
            outputs, mu, var = outputs
        # Convert to numpy
        inp, out, lab = inputs.cpu().detach().numpy(), outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
        # Select first image of each batch and move color channel at the end
        inp, out, lab = inp[0].transpose(1, 2, 0), out[0].transpose(1, 2, 0), lab[0].transpose(1, 2, 0)
        inp, out, lab = (inp*255).astype(np.uint8), (out*255).astype(np.uint8), (lab*255).astype(np.uint8)
        # Save input for all tasks
        cv2.imwrite(str(path) + f'/ep{epoch}_{dset_name}_input.png', inp)
        # For segmentation, transform the output into binary and save label as well
        if self.training_task == 'segmentation':
            cv2.imwrite(str(path) + f'/ep{epoch}_{dset_name}_label.png', lab.astype(np.float32) * 255)
            out = (out > 0.5).astype(np.uint8)
        # Save output for all tasks
        cv2.imwrite(str(path) + f'/ep{epoch}_{dset_name}_output.png', out)


###############################################################################################################################
############################ CROSS VALIDATION #################################################################################
###############################################################################################################################
    # transfer is 'sequential' or 'parallel'
    # type is 'train' or 'spec'
    def cross_validation(self, model_path, transfer, type):
        # List of test metrics for each dataset
        self.complete_metrics_crossval = {}

        # Dataset splits for each dataset
        dataset_splits = {}
        for dset_name in self.dsets_names:
            full_dataset = LoopLoader(
                dset_path=self.paths['dsets'][dset_name],
                # Aggregate all data
                which=['train', 'val', 'test'],
                batch_size=self.batch_size[type],
                cuda_device=self.cuda_device,
                training_task=self.training_task
            # Only take the actual data
            ).ds_folder
            splits = [1 / self.k_fold]*self.k_fold
            # For reproducibility
            generator = torch.Generator().manual_seed(42)
            # Actual random split
            splits = utils.random_split(full_dataset, splits, generator)
            dataset_splits[dset_name] = splits
            # Initialize empty list for metrics
            self.complete_metrics_crossval[dset_name] = []

        # Main loop of the cross validation
        for idx in range(self.k_fold):
            start = time.perf_counter()
            self.current_test_fold = idx
            utils.print_time(f"CROSS VALIDATION RUN {self.current_test_fold}")

            self.train_loop_loaders = {}
            self.train_data_loaders = {}
            self.validation_data_loaders = {}
            self.test_data_loaders = {}

            for dset_name in self.dsets_names:
                # The 3 full datasets (no split)
                train_loop_loader = LoopLoader(
                    dset_path=self.paths['dsets'][dset_name],
                    which=['train'],
                    batch_size=self.batch_size[type],
                    cuda_device=self.cuda_device,
                    training_task=self.training_task
                )

                validation_loop_loader = LoopLoader(
                    dset_path=self.paths['dsets'][dset_name],
                    which=['val'],
                    batch_size=self.batch_size['evaluation'],
                    cuda_device=self.cuda_device,
                    training_task=self.training_task
                )

                test_loop_loader = LoopLoader(
                    dset_path=self.paths['dsets'][dset_name],
                    which=['test'],
                    batch_size=self.batch_size['evaluation'],
                    cuda_device=self.cuda_device,
                    training_task=self.training_task
                )

                # Test set is idx
                test_loop_loader.modify_dataset(dataset_splits[dset_name][idx])
                # Validation set is same as test (no need, since not simulating early stopping)
                validation_loop_loader.modify_dataset(dataset_splits[dset_name][idx])
                # Train set is the concatenation of the other ones
                train_loop_loader.modify_dataset(torch.utils.data.ConcatDataset([dataset_splits[dset_name][i] for i in range(self.k_fold) if i != idx]))

                # Loaders for the datasets
                self.train_loop_loaders[dset_name] = train_loop_loader
                self.train_data_loaders[dset_name] = train_loop_loader.test_loader
                self.validation_data_loaders[dset_name] = validation_loop_loader.test_loader
                self.test_data_loaders[dset_name] = test_loop_loader.test_loader

                print(f""">> Data loaded for dataset {dset_name} from {self.paths['dsets'][dset_name]}
                    train: {len(train_loop_loader.ds_folder)} samples
                    validation: {len(validation_loop_loader.ds_folder)} samples
                    test: {len(test_loop_loader.ds_folder)} samples
                """)

            # Launch the run
            if transfer == 'sequential':
                if type == 'train':
                    self.s_train(model_path)
                if type == 'spec':
                    self.s_specialization(model_path)
            if transfer == 'parallel':
                if type == 'train':
                    self.p_train(model_path)
                if type == 'spec':
                    self.p_specialization(model_path)

            stop = time.perf_counter()
            run_time = stop - start
            print(f"> Run {self.current_test_fold} took {int(run_time / 3600)} hours {int((run_time % 3600) / 60)} minutes {run_time % 60:.1f} seconds")

            # Reset metrics for the next run
            self.metrics_plot = pd.DataFrame(columns=['type', 'feature_extractor', 'epoch', 'n_samples', 'dataset', 'split', 'metric', 'value'])

        # Compute metrics (average and standard deviation) over all run over all metrics
        print(f"> Compute metrics over cross evaluation")
        # Key is dataset name and value is dict containing list of all metrics over all runs
        full_metrics = {}
        for dset_name, list_metrics in self.complete_metrics_crossval.items():
            # Initialize complete dict
            temp_dict = copy.deepcopy(list_metrics[0])
            full_metrics[dset_name] = {}
            for metric_name, value in temp_dict.items():
                # Each entry is an array
                full_metrics[dset_name][metric_name] = np.array([value])
            # Add values from other runs
            for metrics_dict in list_metrics[1:]:
                for metric_name, value in metrics_dict.items():
                    full_metrics[dset_name][metric_name] = np.append(full_metrics[dset_name][metric_name], np.array([value]))

        for dset_name, metrics_dict in full_metrics.items():
            print(f"""
          RESULTS OF CROSS VALIDATION ON {dset_name}
          -----------------------------------------------------------------""")
            for metric_name, values in metrics_dict.items():
                print(f"""          {metric_name}: {np.mean(values)} +/- {np.std(values)}""")
            print(f"""          -----------------------------------------------------------------""")


###############################################################################################################################
############################ PARALLEL TRANSFER LEARNING #######################################################################
###############################################################################################################################
    def p_train(self, model_path):
        # Typhon has external loop for epochs, then loops on the
        # datasets in turn, and for each it trains on a single batch for each epoch.
        utils.print_time("PARALLEL TRAINING")
        start_run_time = time.perf_counter()
        # Load data only if no cross validation, otherwise data is already loaded
        if self.k_fold == 0:
            if self.twolevels:
                _, self.train_loop_loaders, self.train_data_loaders, self.validation_data_loaders, self.test_data_loaders, self.idx_to_class, self.full_train_idx_to_class, self.class_to_dset = load_data(
                    type='train',
                    ultra_typhon=self.ultra_typhon,
                    twolevels=self.twolevels,
                    dsets_names=self.dsets_names,
                    dset_splits=self.dset_splits,
                    paths=self.paths,
                    batch_size=self.batch_size,
                    training_task=self.training_task,
                    cuda_device=self.cuda_device
                )
            else:
                _, self.train_loop_loaders, self.train_data_loaders, self.validation_data_loaders, self.test_data_loaders = load_data(
                    type='train',
                    ultra_typhon=self.ultra_typhon,
                    twolevels=self.twolevels,
                    dsets_names=self.dsets_names,
                    dset_splits=self.dset_splits,
                    paths=self.paths,
                    batch_size=self.batch_size,
                    training_task=self.training_task,
                    cuda_device=self.cuda_device
                )

        self.load_model_and_optims(model_path, 'train')
        self.model.print_stats()
        range_epochs = range(1, self.nb_epochs['train'] + 1)

        if self.resume:
            assert (self.paths['metrics'] / 'metrics.csv').is_file(), "Cannot resume empty experiment"
            # index_col avoids adding new column and take first column as index
            self.metrics_plot = pd.read_csv(self.paths['metrics'] / 'metrics.csv', index_col=0)
            # Delete 'test' results and 'specialized' metrics if any
            self.metrics_plot.drop(self.metrics_plot[self.metrics_plot['epoch'] == -1].index, inplace=True)
            self.metrics_plot.drop(self.metrics_plot[self.metrics_plot['type'] == 'specialized'].index, inplace=True)
            start_epoch = self.metrics_plot['epoch'].max() + 1
            range_epochs = range(start_epoch, start_epoch + self.nb_epochs['train'])
            print(f"> Resuming training from epoch {start_epoch}")

        # Compute the epochs at which the batch size will be doubled
        self.epochs_to_multiply_bs = {int(np.floor(len(range_epochs) * frac) + min(range_epochs)): multiplicator for frac, multiplicator in self.epochs_to_multiply_bs.items()}

        # Outer loop: epochs
        for epoch in tqdm(range_epochs):
            self.epoch = epoch
            # Only used in ultra typhon to compute the real samples seen within the epoch
            if self.ultra_typhon or self.twolevels: self.n_more_batches = 0

            # Check if timer threshold has exceeded
            if (time.perf_counter() - start_run_time) > self.time_threshold:
                print(f"> Time threshold exceeded, stopping experiment")
                self.p_train_compute_metrics_end_epoch()
                break

            # Check if the batch size needs to be changed
            if epoch in self.epochs_to_multiply_bs.keys():
                old_bs = self.batch_size['train']
                self.batch_size['train'] = int(self.batch_size['train'] * self.epochs_to_multiply_bs[epoch])
                for loader in self.train_loop_loaders.values():
                    loader.batch_size = self.batch_size['train']
                    # To actually modify the batch size need to reload
                    loader.reload_iter()
                print(f">> Multiplying batch size at epoch {epoch} by {self.epochs_to_multiply_bs[epoch]}: {old_bs} -> {self.batch_size['train']}")

            # Inner loop: datasets/heads
            # Train on each dataset
            for dset_name in self.dsets_names:
                self.train_step(self.model, dset_name, 'some')

            # Compute metrics, compare models and save models every x epochs, including last epoch
            if epoch in self.epochs_to_evaluate['train'] and not self.eval_within_epoch:
                # Internally it loops on all heads
                self.p_train_compute_metrics_end_epoch()

        # Test and save trained models at the end of the training
        print(f"> Models training completed, testing now")
        if self.ultra_typhon or self.twolevels:
            dset_name = self.dsets_names[0].split('_')[0]
            print(f">> Results for {dset_name}, WITHOUT specialization")
            metrics_test = self.test_model(
                # Here best models are the best_model_DATASET_p.pth
                model=self.best_models[dset_name],
                dset_name=dset_name,
                test_data_loader=self.test_data_loaders[dset_name],
                verbose=True
            )

            self.aggregate_metrics(metrics_test[dset_name], 'test', dset_name, -1, 'trained', 'unfrozen')

        else:
            for dset_name in self.dsets_names:
                print(f">> Results for {dset_name}, WITHOUT specialization")
                metrics_test = self.test_model(
                    # Here best models are the best_model_DATASET_p.pth
                    model=self.best_models[dset_name],
                    dset_name=dset_name,
                    test_data_loader=self.test_data_loaders[dset_name],
                    verbose=True
                )

                # Add to list of metrics for cross validation
                if self.k_fold > 0:
                    self.complete_metrics_crossval[dset_name].append(metrics_test)

                self.aggregate_metrics(metrics_test, 'test', dset_name, -1, 'trained', 'unfrozen')

        # Save the last full model
        model_state = copy.deepcopy(self.model.to_state_dict())
        torch.save(model_state, self.paths['train_model_p'])
        print(f"> Training complete")
        print(f"> Metrics computation time in training: {int(self.total_metrics_time['train'] / 3600)} hours {int((self.total_metrics_time['train'] % 3600) / 60)} minutes {self.total_metrics_time['train'] % 60:.1f} seconds")


    def p_train_compute_metrics_end_epoch(self):
        start_metrics_time = time.time()
        print(f">> Evaluating at epoch {self.epoch}")
        if self.ultra_typhon or self.twolevels:
            # Dict metrics for each head + full model
            # Here dset_name is only used to pick the data_loader
            metrics_training, metrics_validation, metrics_test = self.compute_metrics(self.model, dset_name=self.dsets_names[0].split('_')[0])
            # Cannot iterate on self.dsets_names because there is not the full dataset so use the keys of metrics_training
            for dset_name in metrics_training.keys():
                self.aggregate_metrics(metrics_training[dset_name], 'train', dset_name, self.epoch, 'trained', 'unfrozen')
                self.aggregate_metrics(metrics_validation[dset_name], 'validation', dset_name, self.epoch, 'trained', 'unfrozen')
                self.aggregate_metrics(metrics_test[dset_name], 'test', dset_name, self.epoch, 'trained', 'unfrozen')
                # Compare models only on the full model
                if '_' not in dset_name:
                    self.compare_models(
                        model=self.model,
                        dset_name=dset_name,
                        type='train',
                        # Save to a path that exist
                        save_path=self.paths['best_models_p'][dset_name],
                        metrics_validation=metrics_validation[dset_name]
                    )

        else:
            for dset_name in self.dsets_names:
                print(f"\n>>> Dset {dset_name}")
                metrics_training, metrics_validation, metrics_test = self.compute_metrics(self.model, dset_name)
                print(f">>>> Aggregating metrics for dataset {dset_name}")
                self.aggregate_metrics(metrics_training, 'train', dset_name, self.epoch, 'trained', 'unfrozen')
                self.aggregate_metrics(metrics_validation, 'validation', dset_name, self.epoch, 'trained', 'unfrozen')
                self.aggregate_metrics(metrics_test, 'test', dset_name, self.epoch, 'trained', 'unfrozen')
                print(f">>>> {self.opt_metric} train: {metrics_training[self.opt_metric]}")
                print(f">>>> {self.opt_metric} val: {metrics_validation[self.opt_metric]}")
                print(f">>>> {self.opt_metric} test: {metrics_test[self.opt_metric]}")
                # Save a sample
                self.save_sample(path=self.paths['samples_training'], model=self.model, dset_name=dset_name, epoch=self.epoch)
                # In cross validation, just save the current model (no comparison)
                if self.k_fold > 0:
                    self.best_metrics_dict[dset_name] = copy.deepcopy(metrics_validation)
                    self.best_metrics_dict[dset_name]['epoch'] = self.epoch
                    self.best_models[dset_name] = copy.deepcopy(self.model)
                    torch.save(self.best_models[dset_name].to_state_dict(), self.paths['best_models_p'][dset_name])
                # Otherwise compare models if there is an improvement (i.e. simulate early stopping)
                else:
                    # Save model if it is better than the previous best one
                    self.compare_models(
                        model=self.model,
                        dset_name=dset_name,
                        type='train',
                        save_path=self.paths['best_models_p'][dset_name],
                        metrics_validation=metrics_validation
                    )

        # Save after each interval, so we can quit and resume at any time
        # to_state_dict returns a reference to the state and not the copy, thus it will be modified
        # So we need a deepcopy
        model_state = copy.deepcopy(self.model.to_state_dict())
        torch.save(model_state, self.paths['train_model_p'])
        stop_metrics_time = time.time()
        self.total_metrics_time['train'] += stop_metrics_time - start_metrics_time


    # Specialization after the parallel training
    def p_specialization(self, model_path):
        utils.print_time("SPECIALIZATION")
        # Load only if no cross validation, otherwise data is already loaded
        if self.k_fold == 0:
            _, self.train_loop_loaders, self.train_data_loaders, self.validation_data_loaders, self.test_data_loaders = load_data(
                type='spec',
                ultra_typhon=self.ultra_typhon,
                twolevels=self.twolevels,
                dsets_names=self.dsets_names,
                dset_splits=self.dset_splits,
                paths=self.paths,
                batch_size=self.batch_size,
                training_task=self.training_task,
                cuda_device=self.cuda_device
            )
        self.load_model_and_optims(model_path, 'spec')
        # Best model per each epoch to simulate early stopping on max validation
        best_spec_dict = {}
        best_spec_models = {}

        # Save a first sample, to visualize bootstrap output
        for dset_name in self.dsets_names:
            self.save_sample(path=self.paths['samples_training'], model=self.spec_models[dset_name], dset_name=dset_name, epoch=0)

        for dset_name in self.dsets_names:
            utils.print_time(f">> Dataset {dset_name}")

            # Loop for the specialization epochs
            for epoch in tqdm(range(1, self.nb_epochs['spec'] + 1)):
                self.epoch = epoch
                print(f">>> Epoch {self.epoch}")
                self.spec_models[dset_name].train()
                self.train_step(self.spec_models[dset_name], dset_name, 'all')

                # Compute metrics, compare models and save models every x epochs, including last epoch
                if epoch in self.epochs_to_evaluate['spec']:
                    start_metrics_time = time.time()

                    metrics_training, metrics_validation, metrics_test = self.compute_metrics(self.spec_models[dset_name], dset_name)
                    print(f">>> Aggregating metrics")
                    self.aggregate_metrics(metrics_training, 'train', dset_name, epoch, 'specialized', 'unfrozen')
                    self.aggregate_metrics(metrics_validation, 'validation', dset_name, epoch, 'specialized', 'unfrozen')
                    self.aggregate_metrics(metrics_test, 'test', dset_name, epoch, 'specialized', 'unfrozen')

                    # Save a sample
                    self.save_sample(path=self.paths['samples_spec'], model=self.spec_models[dset_name], dset_name=dset_name, epoch=epoch)

                    # In cross validation, just save the current model (no comparison, no early stopping)
                    if self.k_fold > 0:
                        self.best_metrics_dict[dset_name] = copy.deepcopy(metrics_validation)
                        self.best_metrics_dict[dset_name]['epoch'] = epoch
                        self.best_models[dset_name] = copy.deepcopy(self.spec_models[dset_name])
                        torch.save(self.best_models[dset_name].to_state_dict(), self.paths['spec_models_p'][dset_name])
                    # Otherwise compare models if there is an improvement (i.e. simulate early stopping)
                    else:
                        self.compare_models(
                            model=self.spec_models[dset_name],
                            dset_name=dset_name,
                            type='spec',
                            save_path=self.paths['spec_models_p'][dset_name],
                            metrics_validation=metrics_validation
                        )

                    stop_metrics_time = time.time()
                    self.total_metrics_time['spec'] += stop_metrics_time - start_metrics_time

            # Test the best model (the one that has been saved)
            print(f">> Results for {dset_name}, WITH specialization")
            metrics_test = self.test_model(
                model=self.best_models[dset_name],
                dset_name=dset_name,
                test_data_loader=self.test_data_loaders[dset_name],
                verbose=True
            )

            # Add to list of metrics for cross validation
            if self.k_fold > 0:
                self.complete_metrics_crossval[dset_name].append(metrics_test)

            self.aggregate_metrics(metrics_test, 'test', dset_name, -1, 'specialized', 'unfrozen')

        print(f"> Specialization completed")
        print(f"> Metrics computation time in specialization: {int(self.total_metrics_time['spec'] / 3600)} hours {int((self.total_metrics_time['spec'] % 3600) / 60)} minutes {self.total_metrics_time['spec'] % 60:.1f} seconds")


###############################################################################################################################
############################ SEQUENTIAL TRANSFER LEARNING #####################################################################
###############################################################################################################################
    def s_train(self, model_path):
        # Hydra loops on the datasets in turn
        # and has internal loop for epochs
        utils.print_time("SEQUENTIAL TRAINING")
        # Load data only if no cross validation, otherwise data is already loaded
        if self.k_fold == 0:
            _, self.train_loop_loaders, self.train_data_loaders, self.validation_data_loaders, self.test_data_loaders = load_data(
                type='train',
                ultra_typhon=self.ultra_typhon,
                twolevels=self.twolevels,
                dsets_names=self.dsets_names,
                dset_splits=self.dset_splits,
                paths=self.paths,
                batch_size=self.batch_size,
                training_task=self.training_task,
                cuda_device=self.cuda_device
            )
        self.load_model_and_optims(model_path, 'train')

        # For each dataset, Hydra trains on some epochs for all batches
        for idx, dset_name in enumerate(self.dsets_names):
            print(f">> Dset {dset_name}")

            for feature_extractor in ['frozen', 'unfrozen']:
                # Only train with unfrozen feature extractor for the first dataset
                if idx == 0 and feature_extractor == 'frozen':
                    continue

                # Initialization for further use
                best_train_dict = {}
                best_train_model = {}

                # First passage with frozen FE
                if feature_extractor == 'frozen':
                    self.load_model_and_optims(self.paths['train_model_s'], 'train', frozen=True)
                    print(f">>> Train {dset_name} with frozen feature extractor")

                # Second passage with unfrozen FE
                if feature_extractor == 'unfrozen':
                    if idx != 0:
                        self.load_model_and_optims(self.paths['train_model_s'], 'train', frozen=False)
                    print(f">>> Train {dset_name} with unfrozen feature extractor")

                # Save a first sample, to visualize bootstrap output
                self.save_sample(path=self.paths['samples_training'], model=self.model, dset_name=dset_name, epoch=0)
                for epoch in tqdm(range(1, self.nb_epochs['train'] + 1)):
                    self.epoch = epoch
                    print(f">>>> Epoch {self.epoch}")
                    self.model.train()
                    if feature_extractor == 'frozen': self.model.freeze_fe()
                    if feature_extractor == 'unfrozen': self.model.unfreeze_fe()
                    self.train_step(self.model, dset_name, 'all')

                    # Compute metrics, compare models and save models every x epochs, including last epoch
                    if epoch in self.epochs_to_evaluate['train']:
                        start_metrics_time = time.time()
                        metrics_training, metrics_validation, metrics_test = self.compute_metrics(self.model, dset_name)
                        # Add training and validation metrics for this epoch
                        print(f">>>> Aggregating metrics")
                        self.aggregate_metrics(metrics_training, 'train', dset_name, epoch, 'trained', feature_extractor)
                        self.aggregate_metrics(metrics_validation, 'validation', dset_name, epoch, 'trained', feature_extractor)
                        self.aggregate_metrics(metrics_test, 'test', dset_name, epoch, 'trained', feature_extractor)
                        print(f">>> {self.opt_metric} train: {metrics_training[self.opt_metric]} ")
                        print(f">>> {self.opt_metric} val: {metrics_validation[self.opt_metric]} ")
                        print(f">>> {self.opt_metric} test: {metrics_test[self.opt_metric]} ")

                        # Save a sample
                        self.save_sample(path=self.paths['samples_training'], model=self.model, dset_name=dset_name, epoch=epoch)

                        # In cross validation, just save the current model (no comparison)
                        if self.k_fold > 0:
                            self.best_metrics_dict[dset_name] = copy.deepcopy(metrics_validation)
                            self.best_metrics_dict[dset_name]['epoch'] = epoch
                            self.best_models[dset_name] = copy.deepcopy(self.model)
                            torch.save(self.best_models[dset_name].to_state_dict(), self.paths['gen_model_s'])
                        # Otherwise compare models if there is an improvement (i.e. simulate early stopping)
                        else:
                            if (feature_extractor == 'unfrozen') and (idx == 0):
                                # Save also the very first base model, after the "normal training"
                                self.compare_models(
                                    model=self.model,
                                    dset_name=dset_name,
                                    type='train',
                                    save_path=self.paths['gen_model_s'],
                                    metrics_validation=metrics_validation
                                )

                            self.compare_models(
                                model=self.model,
                                dset_name=dset_name,
                                type='train',
                                save_path=self.paths['train_model_s'],
                                metrics_validation=metrics_validation
                            )

                        stop_metrics_time = time.time()
                        self.total_metrics_time['train'] += stop_metrics_time - start_metrics_time

            # Test first (target) dataset
            if idx == 0:
                print(f">> Results for {dset_name}, WITHOUT specialization")
                metrics_test = self.test_model(
                    model=self.best_models[dset_name],
                    dset_name=dset_name,
                    test_data_loader=self.test_data_loaders[dset_name],
                    verbose=True)

                # Add to list of metrics for cross validation
                if self.k_fold > 0:
                    self.complete_metrics_crossval[dset_name].append(metrics_test)

                self.aggregate_metrics(metrics_test, 'test', dset_name, -1, 'trained', 'unfrozen')

        print(f"> Training complete")
        print(f"> Metrics computation time in training: {int(self.total_metrics_time['train'] / 3600)} hours {int((self.total_metrics_time['train'] % 3600) / 60)} minutes {self.total_metrics_time['train'] % 60:.1f} seconds")


    # Specialization after the sequential training
    def s_specialization(self, model_path):
        utils.print_time("SPECIALIZATION")
        # Load data only if no cross validation, otherwise data is already loaded
        if self.k_fold == 0:
            _, self.train_loop_loaders, self.train_data_loaders, self.validation_data_loaders, self.test_data_loaders = load_data(
                type='spec',
                ultra_typhon=self.ultra_typhon,
                twolevels=self.twolevels,
                dsets_names=self.dsets_names,
                dset_splits=self.dset_splits,
                paths=self.paths,
                batch_size=self.batch_size,
                training_task=self.training_task,
                cuda_device=self.cuda_device
            )
        self.load_model_and_optims(model_path, 'spec', frozen=True)

        # Specialization only on the first/target dataset
        dset_name = self.dsets_names[0]

        utils.print_time(f">> Dataset {dset_name}")
        # Best model per each epoch to simulate early stopping on max validation
        best_spec_dict = {}
        best_spec_models = {}

        for feature_extractor in ['frozen', 'unfrozen']:
            # First passage with frozen FE
            if feature_extractor == 'frozen':
                print(f">> Train {dset_name} with frozen feature extractor")

            # Second passage with unfrozen FE
            if feature_extractor == 'unfrozen':
                self.load_model_and_optims(self.paths['spec_models_s'][dset_name], 'spec', frozen=False)
                print(f">> Train {dset_name} with unfrozen feature extractor")

            # Save a first sample, to visualize bootstrap output
            self.save_sample(path=self.paths['samples_training'], model=self.spec_models[dset_name], dset_name=dset_name, epoch=0)
            # Loop for the specialization epochs
            for epoch in range(1, self.nb_epochs['spec'] + 1):
                self.epoch = epoch
                print(f">>> Epoch {self.epoch}")
                self.spec_models[dset_name].train()
                if feature_extractor == 'frozen': self.spec_models[dset_name].freeze_fe()
                if feature_extractor == 'unfrozen': self.spec_models[dset_name].unfreeze_fe()
                self.train_step(self.spec_models[dset_name], dset_name, 'all')

                # Compute metrics, compare models and save models every x epochs, including last epoch
                if epoch in self.epochs_to_evaluate['spec']:
                    start_metrics_time = time.time()
                    metrics_training, metrics_validation, metrics_test = self.compute_metrics(self.spec_models[dset_name], dset_name)
                    print(f">>> Aggregating metrics")
                    self.aggregate_metrics(metrics_training, 'train', dset_name, epoch, 'specialized', feature_extractor)
                    self.aggregate_metrics(metrics_validation, 'validation', dset_name, epoch, 'specialized', feature_extractor)
                    self.aggregate_metrics(metrics_test, 'test', dset_name, epoch, 'specialized', feature_extractor)

                    # Save a sample
                    self.save_sample(path=self.paths['samples_spec'], model=self.spec_models[dset_name], dset_name=dset_name, epoch=epoch)

                    # In cross validation, just save the current model (no comparison)
                    if self.k_fold > 0:
                        self.best_metrics_dict[dset_name] = copy.deepcopy(metrics_validation)
                        self.best_metrics_dict[dset_name]['epoch'] = epoch
                        self.best_models[dset_name] = copy.deepcopy(self.spec_models[dset_name])
                        torch.save(self.best_models[dset_name].to_state_dict(), self.paths['spec_models_s'][dset_name])
                    # Otherwise compare models if there is an improvement (i.e. simulate early stopping)
                    else:
                        self.compare_models(
                            model=self.spec_models[dset_name],
                            dset_name=dset_name,
                            type='spec',
                            save_path=self.paths['spec_models_s'][dset_name],
                            metrics_validation=metrics_validation
                        )

                    stop_metrics_time = time.time()
                    self.total_metrics_time['spec'] += stop_metrics_time - start_metrics_time

        # Test the best model (the one that has been saved)
        print(f"> Results for {dset_name}, WITH specialization")
        metrics_test = self.test_model(
            model=self.best_models[dset_name],
            dset_name=dset_name,
            test_data_loader=self.test_data_loaders[dset_name],
            verbose=True
        )

        # Add to list of metrics for cross validation
        if self.k_fold > 0:
            self.complete_metrics_crossval[dset_name].append(metrics_test)

        self.aggregate_metrics(metrics_test, 'test', dset_name, -1, 'specialized', 'unfrozen')

        print(f"> Specialization completed")
        print(f"> Metrics computation time in specialization: {int(self.total_metrics_time['spec'] / 3600)} hours {int((self.total_metrics_time['spec'] % 3600) / 60)} minutes {self.total_metrics_time['spec'] % 60:.1f} seconds")


###############################################################################################################################
############################ BOOTSTRAP ########################################################################################
###############################################################################################################################
    def smaller_data_loader(self, limit, loader):
        i = 0
        for i, el in enumerate(loader):
            i += 1
            if i >= limit:
                yield el
                return
            else:
                yield el


    @torch.no_grad()
    def bootstrap(self):
        self.epoch = 0
        if self.training_task == 'autoencoding':
            return self.bootstrap_autoencoding()
        if self.ultra_typhon or self.twolevels:
            return self.bootstrap_ultratyphon_twolevelstyphon()
        utils.print_time("BOOTSTRAP")
        self.bootstrap_data_loaders, self.train_loop_loaders, self.train_data_loaders, self.validation_data_loaders, self.test_data_loaders = load_data(
            type='bootstrap',
            ultra_typhon=self.ultra_typhon,
            twolevels=self.twolevels,
            dsets_names=self.dsets_names,
            dset_splits=self.dset_splits,
            paths=self.paths,
            batch_size=self.batch_size,
            training_task=self.training_task,
            cuda_device=self.cuda_device
        )

        # Best metrics of each head
        # Key 'model' contains the model
        best = {dset:{} for dset in self.dsets_names}

        for nmodel in tqdm(range(self.bootstrap_size)):
            # Take the dropouts of the training (no impact since we only test)
            dropout_fe, dropouts_dm = self.dropouts['train']

            model = TyphonModel(
                dropout_fe=dropout_fe,
                dropouts_dm=dropouts_dm,
                architecture=self.architecture,
                dsets_names=self.dsets_names,
                n_classes=self.n_classes,
                training_task=self.training_task
            ).to(self.cuda_device)

            nbetterheads = 0

            # Save max diff of best and worse head
            max_diff = float("inf")

            # Need to reset the dict at each new model
            current = {dset:{} for dset in self.dsets_names}
            current['model'] = model
            # To speed up bootstrap, go to next iteration when the model is bad
            bad_model = False

            for dset_name in self.dsets_names:
                print(f">>> {dset_name}")
                metrics_test = self.test_model(
                    model=model,
                    dset_name=dset_name,
                    test_data_loader=self.bootstrap_data_loaders[dset_name],
                    # Use the smaller data loader to speed up the bootstrap
                    # test_data_loader=self.smaller_data_loader(1000, self.bootstrap_data_loaders[dset_name]),
                )

                current[dset_name] = metrics_test

                # We need a basis model at the first iteration
                if nmodel == 0:
                    best['model'] = model
                    best[dset_name] = metrics_test
                    print(f">>> First iteration for {dset_name}, {self.opt_metric}: {best[dset_name][self.opt_metric]}")
                    continue

                new_score = current[dset_name][self.opt_metric]
                best_score = best[dset_name][self.opt_metric]

                # Metrics to minimize (list to complete)
                if self.opt_metric in ['hd']:
                    if new_score < best_score:
                        nbetterheads += 1
                        print(f">>> Current better `{self.opt_metric}` for {dset_name}: {new_score}")
                # Otherwise metrics to maximize
                else:
                    if new_score > best_score:
                        nbetterheads += 1
                        print(f">>> Current better `{self.opt_metric}` for {dset_name}: {new_score}")

                # Make sure this is only when using AUC
                if new_score < 0.5 and (self.opt_metric == 'auc'):
                    bad_model = True
                    # Directly go to the next model
                    break

            # Make sure this is only when using AUC
            # Throw the model to speed up bootstrap and avoid computations
            if bad_model and (self.opt_metric == 'auc'):
                print(f">> One head is <0.5 AUC, throw the model")
                continue

            opt_metric = [current[dset_name][self.opt_metric] for dset_name in self.dsets_names]

            if self.opt_metric == 'iou':
                # At least two better heads and a better total
                # iou mostly around 0 but sometime up to 7/8
                if (nbetterheads > 1) and sum(opt_metric) > sum([best[dset_name][self.opt_metric] for dset_name in self.dsets_names]):
                    print(f">> New best model")
                    best = current
                    for dset_name in self.dsets_names:
                        print(f">>> New {self.opt_metric} score for {dset_name}: {best[dset_name][self.opt_metric]}")
            elif self.opt_metric == 'hd':
                # At least two better heads and max difference of 150 -> better model
                if (nbetterheads > 1) and ((max(opt_metric) - min(opt_metric)) < 150):
                    print(f">> New best model")
                    best = current
                    for dset_name in self.dsets_names:
                        print(f">>> New {self.opt_metric} score for {dset_name}: {best[dset_name][self.opt_metric]}")
            elif self.opt_metric == 'dice':
                # At least two better heads and small difference between best and worse head
                new_diff = (max(opt_metric) - min(opt_metric))
                if (nbetterheads > 1) and (new_diff < max_diff):
                    print(f">> New best model")
                    best = current
                    max_diff = new_diff
                    for dset_name in self.dsets_names:
                        print(f">>> New {self.opt_metric} score for {dset_name}: {best[dset_name][self.opt_metric]}")
            else:
                # At least two better heads and max difference of 0.2 -> better model
                if (nbetterheads > 0) and ((max(opt_metric) - min(opt_metric)) < 0.2):
                    print(f">> New best model")
                    best = current
                    for dset_name in self.dsets_names:
                        print(f">>> New {self.opt_metric} score for {dset_name}: {best[dset_name][self.opt_metric]}")

        # Compute metrics for all datasets on the bootstrap model (epoch 0 considered as bootstrap)
        for dset_name in self.dsets_names:
            print(f">> Computing metrics for {dset_name}")
            metrics_training, metrics_validation, metrics_test = self.compute_metrics(best['model'], dset_name)
            # To avoid problems when plotting, put epoch 0 either in train or spec
            if self.nb_epochs['train'] == 0:
                type = 'specialized'
            else:
                type = 'trained'
            self.aggregate_metrics(metrics_training, 'train', dset_name, 0, type, 'unfrozen')
            self.aggregate_metrics(metrics_validation, 'validation', dset_name, 0, type, 'unfrozen')
            self.aggregate_metrics(metrics_test, 'test', dset_name, 0, type, 'unfrozen')

        torch.save(best['model'].to_state_dict(), self.paths['bootstrap_model'])
        print("> Bootstrap done, best model is saved:")
        for dset_name in self.dsets_names:
            print(f"> {self.opt_metric} score for {dset_name}: {best[dset_name][self.opt_metric]}")


    @torch.no_grad()
    def bootstrap_ultratyphon_twolevelstyphon(self):
        utils.print_time("BOOTSTRAP ULTRA TYPHON")
        if self.twolevels:
            self.bootstrap_data_loaders, self.train_loop_loaders, self.train_data_loaders, self.validation_data_loaders, self.test_data_loaders, self.idx_to_class, self.full_train_idx_to_class, self.class_to_dset = load_data(
                type='bootstrap',
                ultra_typhon=self.ultra_typhon,
                twolevels=self.twolevels,
                dsets_names=self.dsets_names,
                dset_splits=self.dset_splits,
                paths=self.paths,
                batch_size=self.batch_size,
                training_task=self.training_task,
                cuda_device=self.cuda_device
            )
        else:
            self.bootstrap_data_loaders, self.train_loop_loaders, self.train_data_loaders, self.validation_data_loaders, self.test_data_loaders = load_data(
                type='bootstrap',
                ultra_typhon=self.ultra_typhon,
                twolevels=self.twolevels,
                dsets_names=self.dsets_names,
                dset_splits=self.dset_splits,
                paths=self.paths,
                batch_size=self.batch_size,
                training_task=self.training_task,
                cuda_device=self.cuda_device
            )
        assert self.opt_metric in ['accuracy', 'auc'], f"Bootstrap not implemented for this metric ({self.opt_metric})"
        good_fe = False
        total_fe_iterations = 0
        full_set = self.dsets_names[0].split('_')[0]

        # PHASE I: get a good FE, at least half of the heads "good"
        print(f"> Phase I: get a good FE")
        while not good_fe:
            n_bad_heads = 0
            good_heads = []
            # Take the dropouts of the training (no impact since we only test)
            dropout_fe, dropouts_dm = self.dropouts['train']
            self.model = TyphonModel(
                dropout_fe=dropout_fe,
                dropouts_dm=dropouts_dm,
                architecture=self.architecture,
                dsets_names=self.dsets_names,
                n_classes=self.n_classes,
                training_task=self.training_task
            ).to(self.cuda_device)

            metrics_test = self.test_model(
                model=self.model,
                dset_name=full_set,
                test_data_loader=self.bootstrap_data_loaders[full_set],
                # Use the smaller data loader to speed up the bootstrap
                # test_data_loader=self.smaller_data_loader(1000, self.bootstrap_data_loaders[full_set]),
            )

            for dset_name in self.dsets_names:
                if (self.opt_metric in ['accuracy', 'auc']) and (metrics_test[dset_name][self.opt_metric] > 0.002) and (metrics_test[dset_name][self.opt_metric] < 0.99):
                    # Keep already good heads
                    good_heads.append(dset_name)
                else: n_bad_heads += 1

                if n_bad_heads > len(self.dsets_names) // 2:
                    print('.', flush=True, end='')
                    break

            total_fe_iterations += 1
            if len(good_heads) >= len(self.dsets_names) // 2:
                print()
                print(f">> Good FE found in {total_fe_iterations} iterations")
                good_fe = True

        # PHASE II: with a good FE reload heads so that they are decent
        print(f"> Phase II: get good heads")
        total_dm_iterations = 0
        good_heads.sort()
        while good_heads != sorted(self.dsets_names):
            metrics_test = self.test_model(
                model=self.model,
                dset_name=full_set,
                # test_data_loader=self.bootstrap_data_loaders[full_set],
                test_data_loader=self.smaller_data_loader(1000, self.bootstrap_data_loaders[full_set]),
            )
            for dset_name in self.dsets_names:
                if dset_name in good_heads: continue
                if (self.opt_metric in ['accuracy', 'auc']) and (metrics_test[dset_name][self.opt_metric] > 0.01) and (metrics_test[dset_name][self.opt_metric] < 0.99):
                    print()
                    print(f">>> Good {dset_name} head found with {self.opt_metric}: {metrics_test[dset_name][self.opt_metric]}")
                    good_heads.append(dset_name)
                    good_heads.sort()
                else:
                    self.model.reload_dm(dset_name)
            print('.', flush=True, end='')
            total_dm_iterations += 1
        print()
        print(f">> Good heads found in {total_dm_iterations} iterations")

        self.p_train_compute_metrics_end_epoch()
        torch.save(self.model.to_state_dict(), self.paths['bootstrap_model'])
        print("> Bootstrap done, best model is saved")


    @torch.no_grad()
    def bootstrap_autoencoding(self):
        utils.print_time("BOOTSTRAP")
        self.bootstrap_data_loaders, self.train_loop_loaders, self.train_data_loaders, self.validation_data_loaders, self.test_data_loaders = load_data(
            type='bootstrap',
            ultra_typhon=self.ultra_typhon,
            twolevels=self.twolevels,
            dsets_names=self.dsets_names,
            dset_splits=self.dset_splits,
            paths=self.paths,
            batch_size=self.batch_size,
            training_task=self.training_task,
            cuda_device=self.cuda_device
        )

        best = {dset:{} for dset in self.dsets_names}

        for nmodel in tqdm(range(self.bootstrap_size)):
            # Take the dropouts of the training (no impact since we only test)
            dropout_fe, dropouts_dm = self.dropouts['train']

            model = TyphonModel(
                dropout_fe=dropout_fe,
                dropouts_dm=dropouts_dm,
                architecture=self.architecture,
                dsets_names=self.dsets_names,
                n_classes=self.n_classes,
                training_task=self.training_task
            ).to(self.cuda_device)

            # Need to reset the dict at each new model
            current = {dset:{} for dset in self.dsets_names}
            current['model'] = model

            # TODO: allow to use only some dsets in bootstrap, possibly with weights
            for dset_name in self.dsets_names:
                # At least for the moment, use loss
                assert self.opt_metric == 'loss'

                # Test model
                print(f">>> {dset_name}")
                metrics_test = self.test_model(
                    model=model,
                    dset_name=dset_name,
                    test_data_loader=self.bootstrap_data_loaders[dset_name],
                    # Use the smaller data loader to speed up the bootstrap
                    # test_data_loader=self.smaller_data_loader(1000, self.bootstrap_data_loaders[dset_name]),
                )

                current[dset_name] = metrics_test

                # We need a basis model at the first iteration
                if nmodel == 0:
                    best['model'] = model
                    best[dset_name] = metrics_test
                    print(f">>> First iteration for {dset_name}, {self.opt_metric}: {best[dset_name][self.opt_metric]}")
                    continue

                new_score = current[dset_name][self.opt_metric]
                best_score = best[dset_name][self.opt_metric]

            opt_metric = [current[dset_name][self.opt_metric] for dset_name in self.dsets_names]
            best_metrics = [best[dset_name][self.opt_metric] for dset_name in self.dsets_names]

            if sum(opt_metric) < sum(best_metrics):
                best = current
                for dset_name in self.dsets_names:
                    print(f">>> ep {nmodel}: New {self.opt_metric} score for {dset_name}: {best[dset_name][self.opt_metric]}")
                print(f'>>> Sum: {sum(opt_metric)}')

        torch.save(best['model'].to_state_dict(), self.paths['bootstrap_model'])

        print("> Bootstrap done, best model is saved:")
        for dset_name in self.dsets_names:
            print(f"> {self.opt_metric} score for {dset_name}: {best[dset_name][self.opt_metric]}")


    def random_initialization(self):
        utils.print_time("RANDOM INITIALIZATION")
        if self.twolevels:
            self.bootstrap_data_loaders, self.train_loop_loaders, self.train_data_loaders, self.validation_data_loaders, self.test_data_loaders, self.idx_to_class, self.full_train_idx_to_class, self.class_to_dset = load_data(
                type='bootstrap',
                ultra_typhon=self.ultra_typhon,
                twolevels=self.twolevels,
                dsets_names=self.dsets_names,
                dset_splits=self.dset_splits,
                paths=self.paths,
                batch_size=self.batch_size,
                training_task=self.training_task,
                cuda_device=self.cuda_device
            )
        else:
            self.bootstrap_data_loaders, self.train_loop_loaders, self.train_data_loaders, self.validation_data_loaders, self.test_data_loaders = load_data(
                type='bootstrap',
                ultra_typhon=self.ultra_typhon,
                twolevels=self.twolevels,
                dsets_names=self.dsets_names,
                dset_splits=self.dset_splits,
                paths=self.paths,
                batch_size=self.batch_size,
                training_task=self.training_task,
                cuda_device=self.cuda_device
            )
        self.epoch = 0

        # Take the dropouts of the training (no impact since we only test)
        dropout_fe, dropouts_dm = self.dropouts['train']

        self.model = TyphonModel(
            dropout_fe=dropout_fe,
            dropouts_dm=dropouts_dm,
            architecture=self.architecture,
            dsets_names=self.dsets_names,
            n_classes=self.n_classes,
            training_task=self.training_task
        ).to(self.cuda_device)

        self.p_train_compute_metrics_end_epoch()
        torch.save(self.model.to_state_dict(), self.paths['bootstrap_model'])
        print("> Random initialization done, model is saved")


    def aggregate_metrics(self, metrics, split, dset_name, epoch, type, feature_extractor):
        # Compute the number of samples trained on so far
        if type == 'trained':
            # n batches per epoch with batch size, on n datasets
            n_samples = epoch * self.nb_batches_per_epoch * self.batch_size['train'] * len(self.dsets_names)
            if self.twolevels:
                n_samples = epoch * self.nb_batches_per_epoch * self.batch_size['train'] * (1 + self.n_negative_heads)
            # Also add the additional batches when computing within the epoch
            if self.ultra_typhon:
                # Evaluation within the epoch, so epoch is currently not finished -> epoch-1 (except epoch 0)
                n_samples = max(epoch-1, 0) * self.nb_batches_per_epoch * self.batch_size['train'] * (1 + self.n_negative_heads) + \
                    self.n_more_batches * self.batch_size['train']
        if type == 'specialized':
            # In specialization, one epoch is over the full training set
            n_samples = epoch * len(self.train_loop_loaders[dset_name].ds_folder)

        # Add all training metrics
        for metric, value in metrics.items():
            # Need to be a dataframe to concatenate
            new_row = pd.DataFrame({
                # Type is either trained or specialized
                'type': type,
                # Feature_extractor is either frozen or unfrozen
                'feature_extractor': feature_extractor,
                'epoch': epoch,
                'n_samples': n_samples,
                'dataset': dset_name,
                # Only used in ultra typhon, to see where we are within the epoch
                # Default is set to None
                'trained_on': self.head_trained_on,
                # Split is either train, validation or test
                'split': split,
                'metric': metric,
                'value': value,
            # Need to pass an index to concatenate
            }, index=[0])
            self.metrics_plot = pd.concat([self.metrics_plot, new_row], ignore_index=True)

        # For cross validation save metrics to different file for each step
        if self.k_fold > 0:
            self.metrics_plot.to_csv(self.paths['metrics'] / f"metrics_{self.current_test_fold}.csv")
        else:
            self.metrics_plot.to_csv(self.paths['metrics'] / 'metrics.csv')


    def compute_overfitting_score(self, dset_name):
        metrics_df = self.metrics_plot[self.metrics_plot['dataset'] == dset_name]
        # Remove best value on the test set
        metrics_df = metrics_df[metrics_df['epoch'] != -1]
        # Actual values that will be plotted
        end_epoch_metrics = pd.DataFrame(columns=metrics_df.columns)
        # Take only the highest n_samples (to remove within epoch values due to ultra typhon)
        for temp_epoch in pd.unique(metrics_df['epoch']):
            # Take only the epoch
            metrics_epoch = metrics_df[metrics_df['epoch'] == temp_epoch]
            metrics_epoch = metrics_epoch[metrics_epoch['n_samples'] == metrics_epoch['n_samples'].max()]
            end_epoch_metrics = pd.concat([end_epoch_metrics, metrics_epoch])

        metrics_loss = end_epoch_metrics[end_epoch_metrics['metric'] == 'loss']
        # Remove first point (bootstrap)
        losses_train = metrics_loss[metrics_loss['split'] == 'train']['value'].to_numpy()[1:]
        losses_val = metrics_loss[metrics_loss['split'] == 'validation']['value'].to_numpy()[1:]
        # Epochs are same for both curves so can take either validation or train
        epochs = metrics_loss[metrics_loss['split'] == 'validation']['n_samples'].to_numpy()[1:]

        return metrics.overfitting_score(losses_train, losses_val, epochs)
