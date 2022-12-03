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
"""Defines the VisualPrompting model (based on CLIP)"""

import os
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from clip import clip

import clipzs
from vp import FixedPatchPrompter, PadPrompter, RandomPatchPrompter, checkers_prompt

PROMPT_TYPES = {
    "padding": PadPrompter,
    "random_patch": RandomPatchPrompter,
    "fixed_patch": FixedPatchPrompter,
    "checkers": checkers_prompt,
}


class CustomCLIP(nn.Module):
    """Modified CLIP module to support prompting."""

    def __init__(self, args, dataset, template="This is a photo of {}"):
        super(CustomCLIP, self).__init__()
        classnames = dataset.classes

        print(f"Loading CLIP (backbone: {args.arch})")
        clip_model = self.load_clip_to_cpu(args)
        clip_model.to(args.device)

        # Hack to make model as float() (This is a CLIP hack)
        if args.device == "cpu":
            clip_model = clip_model.float()

        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        if args.verbose:
            print("List of prompts:")
            pprint(prompts)

        self.clip_model = clip_model
        self.text_features = self.precompute_text_features(prompts, args.device)
        self.logit_scale = self.clip_model.logit_scale.exp().detach()

        assert args.method in PROMPT_TYPES, f"{args.method} is not supported :)!"
        self.prompt_learner = PROMPT_TYPES[args.method](args)

        if args.visualize_prompt:
            self.visualize_prompt(args.method)

    def precompute_text_features(self, prompts: List[str], device: str) -> torch.Tensor:
        return clipzs.ZeroshotCLIP.precompute_text_features(
            self.clip_model, prompts, device
        )
        # necessary since we only call this once on initialization, and learning the
        # prompts does not rely on these gradients
        # with torch.no_grad():
        #     text_tokens = clip.tokenize(prompts).to(device)
        #     text_embeddings = self.clip_model.encode_text(text_tokens)
        #     text_embeddings = text_embeddings / text_embeddings.norm(dim=0)
        # return text_embeddings

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        images = self.prompt_learner(images)
        image_embeddings = self.clip_model.encode_image(images)
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        similarity = self.logit_scale * (image_embeddings @ self.text_features.T)
        return similarity

    def load_clip_to_cpu(self, args):
        """Loads CLIP model to CPU."""
        backbone_name = args.arch
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url, args.root)
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = clip.build_model(state_dict or model.state_dict())
        return model

    @torch.no_grad()
    def visualize_prompt(self, method):
        """Visualizes the prompt."""
        fake_img = torch.ones(1, 3, 224, 224)
        prompted_img = self.prompt_learner(fake_img)[0].cpu()
        prompted_img = torch.clamp(prompted_img, 0, 1)

        print("Visualizing prompt...")
        plt.imsave(f"prompt_{method}.png", prompted_img.permute(1, 2, 0).numpy())
