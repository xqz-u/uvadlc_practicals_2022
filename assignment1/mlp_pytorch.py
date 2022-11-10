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
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import, division, print_function

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(
        self, n_inputs: int, n_hidden: List[int], n_classes: int, use_batch_norm=False
    ):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.

        TODO:
        Implement module setup of the network. DONE
        The linear layer have to initialized according to the Kaiming
        initialization. Add the Batch-Normalization _only_ is use_batch_norm is
        True. DONE

        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss
        module for loss calculation.
        """
        super().__init__()
        layers = []
        in_dim = n_inputs
        for hidden_dim in n_hidden:
            layer = nn.Linear(in_dim, hidden_dim, bias=False)
            # look at source of `nn.init.calculate_gain` to check that
            # nonlinearity='relu' indeed initializes weights to
            # N(0, np.sqrt(2/hidden_dim))
            # NOTE
            # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html#How-to-find-appropriate-initialization-values they they use a different gain for the input layer
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            layers.append(layer)
            layers.append(nn.ELU())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        out_layer = nn.Linear(in_dim, n_classes, bias=False)
        nn.init.kaiming_normal_(out_layer.weight, nonlinearity="relu")
        self.layers = nn.Sequential(*layers, out_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass of the input. Here an input tensor x is
        transformed through several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        # flatten the image and its channels but maintain the batch dimension
        return self.layers(x.reshape(x.size(0), -1))

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some
        situations.
        """
        return next(self.parameters()).device
