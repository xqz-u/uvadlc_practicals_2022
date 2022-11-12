################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""


"""
Computes the confusion matrix, i.e. the number of true positives, false
positives, true negatives and false negatives.

Args:
  predictions: 2D float array of size [batch_size, n_classes], predictions
               of the model (logits)
  labels: 1D int array of size [batch_size]. Ground truth labels for
          each sample in the batch
Returns:
  confusion_matrix: confusion matrix per class, 2D float array of size
                    [n_classes, n_classes]
"""


"""
Converts a confusion matrix to accuracy, precision, recall and f1 scores.
Args:
    confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
Returns: a dictionary with the following keys:
    accuracy: scalar float, the accuracy of the confusion matrix
    precision: 1D float array of size [n_classes], the precision for each class
    recall: 1D float array of size [n_classes], the recall for each clas
    f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
"""


"""
Performs the evaluation of the MLP model on a given dataset.

Args:
  model: An instance of 'MLP', the model to evaluate.
  data_loader: The data loader of the dataset to evaluate.
Returns:
    metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

Hint: make sure to return the average accuracy of the whole dataset,
      independent of batch sizes (not all batches might be the same size).
"""


"""
Performs a full training cycle of MLP model.

Args:
  hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
  lr: Learning rate of the SGD to apply.
  use_batch_norm: If True, adds batch normalization layer into the network.
  batch_size: Minibatch size for the data loaders.
  epochs: Number of training epochs to perform.
  seed: Seed to use for reproducible results.
  data_dir: Directory where to store/find the CIFAR10 dataset.
Returns:
  model: An instance of 'MLP', the trained model that performed best on the validation set.
  val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                  validation set per epoch (element 0 - performance after epoch 1)
  test_accuracy: scalar float, average accuracy on the test dataset of the model that
                 performed best on the validation.
  logging_info: An arbitrary object containing logging information. This is for you to
                decide what to put in here.

Hint: you can save your best model by deepcopy-ing it.
"""
