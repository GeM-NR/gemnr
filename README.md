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

im_pil_list = [gem_nr.crop_resize(im_pil) for im_pil in im_pil_list]
out_im_pil_list = gem_nr.edit(im_pil_list, edit_text_prompt, anchor_idx)
```

NOTE: you need to provide your Hugging Face token as either environment variable `export HF_TOKEN=...` or extra argument to the class `GemNR(resolution=512, token=...)`.

### Demo
```bash
python run.py
```
You can also use the same script to run the method on your inputs:
```bash
python run.py --interactive
```
or
```bash
python run.py -i "path_to_img1 ... path_to_imgN" -e "Your edit" -a 0
```
or
```bash
python run.py -i path_to_img_folder -e "Your edit" -a 0
```
See details in [`run.py`](run.py).

## BibTeX
If you use this work or find it helpful, please consider citing:

```bibtex
@misc{bengtson2026gemnrgeometryawaremultiviewediting,
      title={GeM-NR: Geometry-Aware Multi-View Editing for Nonrigid Scene Changes}, 
      author={Josef Bengtson and Yaroslava Lochman and Fredrik Kahl},
      year={2026},
      eprint={2606.05142},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.05142}, 
}
```

