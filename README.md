# GeM-NR: Geometry-Aware Multi-View Editing for Nonrigid Scene Changes

<p align="center">
    <a href=https://gem-nr.github.io>project page</a> · 
    <a href=https://arxiv.org/abs/2606.05142>paper</a><br><br>
    <img src="https://ylochman.github.io/papers/gemnr/teaser.png">
    <br>
    <em>GeM-NR is a fast and flexible training-free pipeline for general multi-view consistent image editing, including edits that drastically change the geometry and appearance of the scene.</em><br>
</p>

## Installation
The code development was done with python 3.13.5. Install the following packages to re-create our environment:
```bash
pip install -r requirements.txt

# Torch
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# Depth Anything 3
pip install --no-deps -e third_parties/Depth-Anything-3-dev

# RoMa
pip install --no-deps -e third_parties/RoMa-0.0.1

# GeM-NR
pip install -e .
```

## Usage
```python
from gemnr import GemNR

gem_nr = GemNR(resolution=512)
    
# im_pil_list = ...
# edit_text_prompt = ...
# anchor_idx = ...

out_im_pil_list = gem_nr.edit(im_pil_list, edit_text_prompt, anchor_idx)
```