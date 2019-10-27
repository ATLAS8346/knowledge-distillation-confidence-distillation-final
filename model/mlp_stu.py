import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MLP_STU(nn.Module):
    def __init__(self, params):
        super(MLP_STU, self).__init__()
        self.fc1 = nn.Linear(params.input_size, params.hidden1_size)
        self.drop1 = nn.Dropout(params.dropout_rate)
        self.fc2 = nn.Linear(params.hidden1_size, params.hidden2_size)
        self.drop2 = nn.Dropout(params.dropout_rate)
        self.fc3 = nn.Linear(params.hidden2_size, params.hidden3_size)
        self.drop3 = nn.Dropout(params.dropout_rate)
        self.fc4 = nn.Linear(params.hidden3_size, params.num_classes)
        self.reg = nn.Linear(params.hidden3_size, params.teacher_size)

    def forward(self, din):
        # din = din.view(-1, 28 * 28)
        dout = nn.functional.relu(self.drop1(self.fc1(din)))
        dout = nn.functional.relu(self.drop2(self.fc2(dout)))
        dout = nn.functional.relu(self.drop3(self.fc3(dout)))
        return self.fc4(dout)

    def repre(self, din):
        dout = nn.functional.relu(self.drop1(self.fc1(din)))
        dout = nn.functional.relu(self.drop2(self.fc2(dout)))
        dout = nn.functional.relu(self.drop3(self.fc3(dout)))
        return self.reg(dout)
        # return self.forward(din)

    def prediction(self, din):
        return torch.argmax(self.forward(din), dim=1)
        # return torch.max(F.softmax(self.forward(din), dim=1), dim=1)[0]


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


def loss_fn_kd_non_conf(outputs, labels, repres, teacher_repres, params):
    alpha = params.alpha
    T = params.temperature

    # Trade-off term lambda: sigmoid()

    KD_loss = nn.MSELoss()(F.log_softmax(repres / T, dim=1),
                             F.softmax(teacher_repres / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

    # return nn.CrossEntropyLoss()(outputs, labels)

    # return nn.MSELoss()(repres, teacher_repres)


def loss_fn_kd_conf(outputs, labels, repres, teacher_repres, confs, prior, params):
    importance = (confs/prior).reshape((confs.shape[0], 1))
    alpha = params.alpha
    T = params.temperature
    # importance = torch.sigmoid((importance - 0.75) * 35) #0.6 30 0.75 35
    # Trade-off term lambda: sigmoid()
    KD_loss = nn.MSELoss(reduce=False)(F.log_softmax(repres / T, dim=1),
                                         F.softmax(teacher_repres / T, dim=1)) * (alpha * T * T)

    CE_loss = nn.CrossEntropyLoss(reduce=False)(outputs, labels) * (1. - alpha)
    CE_loss = CE_loss.reshape((labels.shape[0], 1))

    loss = importance * (KD_loss+CE_loss)
    return torch.mean(loss)
    # return nn.CrossEntropyLoss()(outputs, labels)

    # loss = nn.MSELoss()(torch.sqrt(importance) * repres, torch.sqrt(importance) * teacher_repres)
    # return loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}