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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
from typing import Dict

import numpy as np

import utils as u


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    params: Dict[str, np.ndarray]
    grads: Dict[str, np.ndarray]
    last_input: np.ndarray = None

    def __init__(self, in_features: int, out_features: int, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        Hint: the input_layer argument might be needed for the initialization
        Also, initialize gradients with zeros.
        """
        self.params = {
            "weight": np.random.normal(
                0.0,
                np.sqrt((1 if input_layer else 2) / in_features),
                size=(out_features, in_features),
            ),
            "bias": np.zeros(out_features),
        }
        self.grads = {
            "weight": np.zeros_like(self.params["weight"]),
            "bias": np.zeros_like(self.params["bias"]),
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        Hint: You can store intermediate variables inside the object. They can
        be used in backward pass computation.
        """
        self.last_input = x
        return (x @ self.params["weight"].T) + self.params["bias"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """
        self.grads["weight"] = dout.T @ self.last_input
        self.grads["bias"] = np.ones(dout.shape[0]) @ dout
        return dout @ self.params["weight"]

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to
        save it.
        """
        self.last_input = None


class ELUModule(object):
    """
    ELU activation module.
    """

    last_input: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        Hint: You can store intermediate variables inside the object. They can
        be used in backward pass computation.
        """
        self.last_input = x
        return np.where(x > 0, x, np.exp(x) - 1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """
        return dout * np.where(self.last_input > 0, 1, np.exp(self.last_input))

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        self.last_input = None


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    last_input: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        To stabilize computation you should use the so-called Max Trick -
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can
        be used in backward pass computation.
        """
        y = np.exp(x - x.max(1)[:, None])
        probs = y / y.sum(1)[:, None]
        self.last_input = probs
        return probs

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
        """
        dx = dout * self.last_input
        dx = dx @ np.ones((self.last_input.shape[1],) * 2)
        dx = dout - dx
        dx = self.last_input * dx
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        self.last_input = None


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
        """
        return -np.sum(u.one_hot(x.shape[1], y) * np.log(x)) / x.shape[0]

    def backward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
        """
        return -u.one_hot(x.shape[1], y) / (x * x.shape[0])
