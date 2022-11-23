#!/bin/bash

micromamba activate dl2021

# install additional packages
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# verify installation
python -c "import clip; print(f'CLIP available models: {clip.available_models()}')"

micromamba deactivate
