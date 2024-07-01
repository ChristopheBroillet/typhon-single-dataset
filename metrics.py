import numpy as np
import sklearn.metrics
import torch
from scipy import ndimage


###################################################################################
############### CLASSIFICATION ####################################################
###################################################################################
def get_singleclass_metrics(loss_function, results):
    input = results['raw_predictions']
    target = results['labels']
    loss = loss_function(input, target).item()

    # Put everything on CPU as numpy array
    results['labels'] = results['labels'].cpu().numpy()
    results['predictions'] = results['predictions'].cpu().numpy()

    accuracy = sklearn.metrics.accuracy_score(results['labels'], results['predictions'])

    return {
        'loss': loss, 'accuracy': accuracy
    }


def get_binaryclass_metrics(loss_function, results):
    input = results['raw_predictions']
    target = results['labels']
    loss = loss_function(input, target).item()

    # Put everything on CPU as numpy array
    results['labels'] = results['labels'].cpu().numpy()
    results['predictions'] = results['predictions'].cpu().numpy()
    results['predictions_positive_class'] = results['predictions_positive_class'].cpu().numpy()

    (tn, fp), (fn, tp) = sklearn.metrics.confusion_matrix(
        results['labels'], results['predictions'], labels=[0,1])

    accuracy = sklearn.metrics.accuracy_score(results['labels'], results['predictions'])
    # If TP+FP=0 then by default set precision to 0 (can happen if model outputs only negative predictions)
    precision = sklearn.metrics.precision_score(results['labels'], results['predictions'], zero_division=0)
    recall = sklearn.metrics.recall_score(results['labels'], results['predictions'])
    f1score = sklearn.metrics.f1_score(results['labels'], results['predictions'])
    auc = sklearn.metrics.roc_auc_score(results['labels'], results['predictions_positive_class'])

    if tn + fp:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0

    return {
        'loss': loss, 'accuracy': accuracy,
        'precision': precision, 'recall': recall,
        'f1score': f1score, 'specificity': specificity, 'auc': auc
    }


def get_multiclass_metrics(loss_function, results, is_ultra_typhon=False, is_two_levels=False):
    input = results['raw_predictions']
    # Full model in ultra_typhon or two_levels: target are one-hot labels
    if is_ultra_typhon or is_two_levels:
        target = results['one_hot_labels']
    else:
        target = results['labels']
    if loss_function is not None:
        loss = loss_function(input, target).item()
    else: loss = None

    # Put everything on CPU as numpy array
    results['labels'] = results['labels'].cpu().numpy()
    results['predictions'] = results['predictions'].cpu().numpy()

    acc = sklearn.metrics.accuracy_score(results['labels'], results['predictions'])
    # If TP+FP=0 then by default set precision to 0
    precision = sklearn.metrics.precision_score(results['labels'], results['predictions'], average='macro', zero_division=0)
    recall = sklearn.metrics.recall_score(results['labels'], results['predictions'], average='macro')
    # F1-score with micro-average is equivalent to accuracy, so use macro average instead
    f1score = sklearn.metrics.f1_score(results['labels'], results['predictions'], average='macro')

    # No sense to compute top-2 accuracy for Ultra Typhon -> will always be 1 since we have two classes
    if not is_ultra_typhon:
        results['probabilities'] = results['probabilities'].cpu().numpy()
        acc2 = sklearn.metrics.top_k_accuracy_score(results['labels'], results['probabilities'], k=2)
    else: acc2 = None

    if is_two_levels:
        auc = sklearn.metrics.roc_auc_score(results['one_hot_labels'].cpu().numpy(), results['probabilities'], multi_class='ovo', average='macro')
    else:
        results['probabilities_sum_to_1'] = results['probabilities_sum_to_1'].cpu().numpy()
        auc = sklearn.metrics.roc_auc_score(results['labels'], results['probabilities_sum_to_1'], multi_class='ovo', average='macro')

    return {
        'loss': loss, 'accuracy': acc,
        'accuracy-top-2': acc2,
        'precision': precision, 'recall': recall, 'f1-score': f1score,
        'auc': auc
    }


def get_single_head_two_levels(loss_function, results, classes):
    input = results['raw_predictions'][:,classes]
    labels = results['one_hot_labels'][:,classes]
    loss = loss_function(input, labels).item()

    # Put everything on CPU as numpy array
    labels = labels.cpu().numpy()
    sigmoid = torch.nn.Sigmoid()
    # Prediction will also be either 0 or 1 as the one-hot labels
    predictions = (sigmoid(input) > 0.5).int().cpu().numpy()

    acc = sklearn.metrics.accuracy_score(labels, predictions)
    # If TP+FP=0 then by default set precision to 0
    precision = sklearn.metrics.precision_score(labels, predictions, average='macro', zero_division=0)
    recall = sklearn.metrics.recall_score(labels, predictions, average='macro')
    # F1-score with micro-average is equivalent to accuracy, so use macro average instead
    f1score = sklearn.metrics.f1_score(labels, predictions, average='macro')
    auc = sklearn.metrics.roc_auc_score(labels, sigmoid(input).cpu().numpy(), multi_class='ovo', average='macro')

    return {
        'loss': loss, 'accuracy': acc, 'precision': precision, 'recall': recall, 'f1-score': f1score, 'auc': auc,
    }


def get_single_head_ultra_typhon_metrics(loss_function, results, dset_idx):
    input = results['raw_predictions'][:, dset_idx]
    labels = results['one_hot_labels'][:, dset_idx]
    loss = loss_function(input, labels).item()

    # Put everything on CPU as numpy array
    labels = labels.cpu().numpy()
    probabilities = results['probabilities'][:, dset_idx].cpu().numpy()
    # Prediction: 1 if it is equal to the dset_idx 0 otherwise
    predictions = (results['predictions'].cpu().numpy() == dset_idx).astype(int)

    acc = sklearn.metrics.accuracy_score(labels, predictions)
    # If all predictions are 0, then TP+FP = 0, so by default put precision to 0 in that case
    precision = sklearn.metrics.precision_score(labels, predictions, zero_division=0)
    recall = sklearn.metrics.recall_score(labels, predictions)
    f1score = sklearn.metrics.f1_score(labels, predictions)
    auc = sklearn.metrics.roc_auc_score(labels, probabilities)

    return {
        'loss': loss, 'accuracy': acc, 'precision': precision, 'recall': recall, 'f1-score': f1score, 'auc': auc
    }


###################################################################################
############### SEGMENTATION ######################################################
###################################################################################
def get_segmentation_metrics(losses, hausdorff_distances, confusion_matrix_dict):
    # Get totals
    tp = sum(confusion_matrix_dict['TP'])
    fp = sum(confusion_matrix_dict['FP'])
    tn = sum(confusion_matrix_dict['TN'])
    fn = sum(confusion_matrix_dict['FN'])

    if tp + fp + tn + fn:
        accuracy = (tp + tn) / (tp + fp + tn + fn)
    else:
        accuracy = 0.0

    if tp + fp:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall:
        f1score = 2 * ((precision * recall) / (precision + recall))
    else:
        f1score = 0.0

    if tn + fp:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0

    if tp + fp + fn:
        iou = tp / (tp + fp + fn)
    else:
        iou = 1.0

    # Compute dice same as paper
    smooth = 0.0001
    i = tp + fn # label = 1
    j = tp + fp # prediction = 1
    intersection = tp
    dice = (2. * intersection + smooth) / (i + j + smooth)

    # Receive directly the per-batch losses and average them
    loss = np.mean(losses)

    # Receive directly the per-batch Hausdorff distances and average them
    hd = np.mean(hausdorff_distances)

    return {
        'loss': loss, 'accuracy': accuracy,
        'precision': precision, 'recall': recall,
        'f1score': f1score, 'specificity': specificity, 'iou': iou, 'dice': dice, 'hd': hd
    }


def hausdorff_dist(input, target):
    """
    Computes the Hausdorff distance between target and input
    Args:
        input: Input tensor (predictions of the model), binary image
        target: Groundtruth segmentation, binary image
    """

    assert input.size() == target.size(), f"'input' and 'target' must have the same shape, got {input.size()} and {target.size()}"

    distances = []

    # Inspired from https://cs.stackexchange.com/questions/117989/hausdorff-distance-between-two-binary-images-according-to-distance-maps
    # input and target come from a batch, so iterate on each single image-label pair
    for prediction, label in zip(input, target):
        # Transform to the required type for cv2 (i.e. Numpy binary array of 8-bits per pixel)
        # Take the first (only) channel, otherwise cv2 crashes
        prediction = prediction.cpu().numpy().astype(np.uint8)[0]
        label = label.cpu().numpy().astype(np.uint8)[0]

        dist_map_prediction = ndimage.distance_transform_edt(1 - prediction)
        try:
            distance_1 = np.max(dist_map_prediction[label.astype(bool)])
        except ValueError:
            distance_1 = 0

        inverted_label = 1 - label
        dist_map_label = ndimage.distance_transform_edt(inverted_label)

        # Prediction could be fully 0, thus no values will be retained when masking
        try:
            distance_2 = np.max(dist_map_label[prediction.astype(bool)])
        except ValueError:
            distance_2 = 0

        distances.append(max(distance_1, distance_2))

    return np.mean(distances)


# Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797
# Usually used as loss function in segmentation task
class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, target):
        assert input.size() == target.size(), f"'input' and 'target' must have the same shape, got {input.size()} and {target.size()}"
        input = input.flatten()
        target = target.flatten()
        intersect = (input * target).sum(-1)
        # Here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
        try:
            return 1 - (2 * (intersect / denominator))
        except ZeroDivisionError:
            return 1 - (2 * (intersect / denominator.clamp(min=1e-6)))


###################################################################################
############### AUTOENCODING ######################################################
###################################################################################
def get_autoencoding_metrics(results):
    # Receive directly the per-batch losses and average them
    loss = torch.mean(results['losses']).item()
    r2score = sklearn.metrics.r2_score(results['labels'], results['predictions'])

    return {
        'loss': loss,
        'r2score': r2score,
    }



###################################################################################
############### OTHERS ############################################################
###################################################################################

def overfitting_score(losses_train, losses_val, epochs):
    assert len(losses_train) == len(losses_val), f"losses_train (len: {len(losses_train)}) not the same length as losses_val (len: {len(losses_val)})"
    assert len(losses_train) == len(epochs), f"losses_train (len: {len(losses_train)}) not the same length as epochs (len: {len(epochs)})"
    if len(losses_train) < 3: return 0
    # Start by computing the moving average, to smooth the polyline
    moving_avg_window_size = 4
    losses_train = np.convolve(losses_train, np.ones(moving_avg_window_size), mode='same') / moving_avg_window_size
    losses_val = np.convolve(losses_val, np.ones(moving_avg_window_size), mode='same') / moving_avg_window_size
    # We can compute discrete derivatives with (i) forward or backward finite differences (error is order 1 in delta_x)
    # (ii) centered finite differences (error is order 2 in delta_x)
    # https://math.stackexchange.com/questions/302160/correct-way-to-calculate-numeric-derivative-in-discrete-time
    # We choose to use (ii): discrete derivative in point x = (f(x+1)-f(x-1)) / 2*delta_x
    # With (ii) we cannot compute derivative for first and last points
    epochs = epochs / np.max(epochs)
    losses_train = losses_train / np.max(losses_train)
    losses_val = losses_val / np.max(losses_val)
    # Compute f(x+1)-f(x-1) using convolution (careful that convolve flips the second array)
    # mode='valid' to compute values where both arrays completely overlap
    delta_f_x_train = np.convolve(losses_train, np.array([1, 0, -1]), mode='vaild')
    delta_f_x_val = np.convolve(losses_val, np.array([1, 0, -1]), mode='vaild')
    # Compute delta_x using convolution
    delta_x = np.convolve(epochs, np.array([1, 0, -1]), mode='valid')
    # Compute derivatives and overfitting score
    discrete_derivatives_train = delta_f_x_train / delta_x
    discrete_derivatives_val = delta_f_x_val / delta_x
    overfitting_scores = []
    for train_derivative, val_derivative in zip(discrete_derivatives_train, discrete_derivatives_val):
        # No overfitting in those cases: if val_derivative negative, or if both derivatives are positives (can happen due to moving target)
        if (val_derivative <= 0) or ((val_derivative > 0) and (train_derivative > 0)):
            overfitting_scores.append(0)
            continue
        # Else, remaining case is when train_derivative is negative while val_derivative is positive
        overfitting_scores.append(val_derivative - train_derivative)

    overfitting_score = np.mean(overfitting_scores)
    return overfitting_score
