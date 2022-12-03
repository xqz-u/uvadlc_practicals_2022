################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################
"""Defines various kinds of visual-prompting modules for images."""

import numpy as np
import torch
import torch.nn as nn


class PadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """

    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        self.pad_size = pad_size
        self.imsize = image_size

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Define the padding as variables self.pad_left, self.pad_right,
        # self.pad_up, self.pad_down

        # Hints:
        # - Each of these are parameters that we need to learn. So how would
        #   you define them in torch?
        # - See Fig 2(c) in the assignment to get a sense of how each of these
        #   should look like.
        # - Shape of self.pad_up and self.pad_down should be (1, 3, pad_size,
        #   image_size)
        # - See Fig 2.(g)/(h) and think about the shape of self.pad_left and
        #   self.pad_right

        self.pad_left = nn.Parameter(
            torch.randn(1, 3, pad_size, image_size - 2 * pad_size)
        )
        self.pad_right = nn.Parameter(
            torch.randn(1, 3, pad_size, image_size - 2 * pad_size)
        )
        self.pad_up = nn.Parameter(torch.randn(1, 3, image_size, pad_size))
        self.pad_down = nn.Parameter(torch.randn(1, 3, image_size, pad_size))

        # self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - 2 * pad_size, pad_size]))
        # self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - 2 * pad_size, pad_size]))
        # self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        # self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, add the prompt as a padding to the image.

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        x[
            :, :, : self.pad_size, self.pad_size : self.imsize - self.pad_size
        ] += self.pad_left
        x[
            :,
            :,
            self.imsize - self.pad_size :,
            self.pad_size : self.imsize - self.pad_size,
        ] += self.pad_right
        x[:, :, :, : self.pad_size] += self.pad_up
        x[:, :, :, self.imsize - self.pad_size :] += self.pad_down
        return x

        #######################
        # END OF YOUR CODE    #
        #######################


class FixedPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a fixed patch over an image.
    For reference, this prompt should look like Fig 2(a) in the PDF.
    """

    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        self.patch = nn.Parameter(torch.randn(1, 3, args.prompt_size, args.prompt_size))

    def forward(self, x):
        # x = x.clone()
        x[:, :, : self.patch.size(2), : self.patch.size(3)] += self.patch
        return x


class RandomPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a random patch in the image.
    For reference, this prompt should look like Fig 2(b) in the PDF.
    """

    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can be defined as self.patch) of size [prompt_size, prompt_size]
        # that is located at the top-left corner of the image.

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn

        self.patch = nn.Parameter(torch.randn(1, 3, args.prompt_size, args.prompt_size))
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - Note that, here, you need to place the patch at a random location
        #   and not at the top-left corner.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        xcoord, ycoord = torch.randint(0, x.size(2) - self.patch.size(2), (2,))
        x[
            :,
            :,
            xcoord : xcoord + self.patch.size(2),
            ycoord : ycoord + self.patch.size(3),
        ] += self.patch
        return x

        #######################
        # END OF YOUR CODE    #
        #######################


class CheckersPrompt(nn.Module):
    def __init__(self, image_size: int, square_size: int):
        super().__init__()

        assert isinstance(image_size, int), "image_size must be an integer"
        assert isinstance(square_size, int), "square_size must be an integer"
        assert (
            not image_size % square_size
        ), "image_size must be divisible by square_size"

        self.tile = nn.Parameter(torch.randn(1, 3, square_size, square_size))
        self.tile_size = square_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, row_id in enumerate(range(0, x.size(2), self.tile_size), 1):
            for j, col_id in enumerate(range(0, x.size(3), self.tile_size), 1):
                if (i % 2 and j % 2) or not (i % 2 or j % 2):
                    x[
                        :,
                        :,
                        row_id : row_id + self.tile_size,
                        col_id : col_id + self.tile_size,
                    ] += self.tile
        return x


checkers_prompt = lambda args: CheckersPrompt(args.image_size, args.prompt_size)
