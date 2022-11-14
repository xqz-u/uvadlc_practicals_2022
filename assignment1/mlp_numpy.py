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
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import, division, print_function

from copy import deepcopy
from typing import List, Tuple

import numpy as np

import modules as m


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    layers: List[Tuple[m.ELUModule, m.LinearModule]]
    softmax_module: m.SoftMaxModule

    def __init__(self, n_inputs: int, n_hidden: List[int], n_classes: int):
        n_hidden.append(n_classes)
        self.layers = [(None, m.LinearModule(n_inputs, n_hidden[0], input_layer=True))]
        self.layers += [
            (m.ELUModule(), m.LinearModule(*dims))
            for dims in zip(n_hidden, n_hidden[1:])
        ]
        self.softmax_module = m.SoftMaxModule()

    @property
    def input_layer(self) -> m.LinearModule:
        return self.layers[0][1]

    @property
    def hidden_layers(self) -> List[Tuple[m.CrossEntropyModule, m.LinearModule]]:
        return self.layers[1:]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs forward pass of the input. Here an input tensor x is
        transformed through several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        out = self.input_layer.forward(x.reshape(x.shape[0], -1))
        for activation, linear in self.hidden_layers:
            out = linear.forward(activation.forward(out))
        return self.softmax_module.forward(out)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss
        """
        dout = self.softmax_module.backward(dout)
        for activation, linear in self.hidden_layers:
            dout = activation.backward(linear.backward(dout))
        return self.input_layer.backward(dout)

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to
        save it.
        """
        self.input_layer.clear_cache()
        for activation, linear in self.hidden_layers:
            activation.clear_cache()
            linear.clear_cache()

    def state_dict(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return deepcopy(
            [
                (linear.params["weight"], linear.params["bias"])
                for _, linear in self.layers
            ]
        )

    def load_state_dict(self, state_dict: List[Tuple[np.ndarray, np.ndarray]]):
        for (_, linear), (weight, bias) in zip(self.layers, state_dict):
            linear.params["weight"] = weight
            linear.params["bias"] = bias
