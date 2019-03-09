import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    length = prediction.shape[0]
    prediction = prediction.astype(int) # 0 and 1
    ground_truth = ground_truth.astype(int) * 2 # 0 and 2

    #FN: 0 - 2 = -2 (0)
    #TP: 1 - 2 = -1 (1)
    #TN: 0 - 0 = 0 (2)
    #FP: 1 - 0 = 1 (3)
    result = prediction - ground_truth

    # add 2 to support np.bincount (positive numbers)
    result = result + 2
    result = np.bincount(result)
    # void exception
    result = np.append(result, [0,0,0,0])

    FN = result[0]
    TP = result[1]
    TN = result[2]
    FP = result[3]

    precision = np.divide(TP, (TP + FP))
    recall = np.divide(TP, (TP + FN))
    accuracy = np.divide((TP + TN), length)

    f1 = np.divide((2 * precision * recall), (precision + recall))
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    
    length = prediction.shape[0]
    
    result = np.abs(prediction - ground_truth)
    result = np.bincount(result)

    # void exception
    result = np.append(result, 0)

    # zero is TP
    TP = result[0]

    return np.divide(TP, length)
