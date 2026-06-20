import numpy as np

def create_confusion_matrix(labels, logits):
    # convert one-hot to class indices
    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes), dtype=int)

    # fill matrix
    for t, p in zip(true_labels, pred_labels):
        confusion[t, p] += 1

    return confusion
