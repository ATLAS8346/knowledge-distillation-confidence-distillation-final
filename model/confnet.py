import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CONFNET(nn.Module):
    def __init__(self, params):
        super(CONFNET, self).__init__()
        self.fc1 = nn.Linear(params.input_size, params.hidden1_size)
        self.drop1 = nn.Dropout(params.dropout_rate)
        self.fc2 = nn.Linear(params.hidden1_size, params.hidden2_size)
        self.drop2 = nn.Dropout(params.dropout_rate)
        self.fc3 = nn.Linear(params.hidden2_size, params.num_classes)

    def forward(self, din):
        dout = nn.functional.relu(self.drop1(self.fc1(din)))
        dout = nn.functional.relu(self.drop2(self.fc2(dout)))
        return self.fc3(dout)


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def loss_fn_conf(outputs, labels, teacher_confs, params):
    # print(F.softmax(outputs))
    outputs = F.log_softmax(outputs)
    weighted_outputs = teacher_confs.reshape((teacher_confs.shape[0], 1)) * outputs
    conf_loss = F.nll_loss(weighted_outputs, labels)
    # print(outputs, labels, teacher_confs, weighted_outputs, conf_loss)
    return conf_loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) [0, 1, ..., num_classes-1]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}