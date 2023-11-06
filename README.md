# GUI Interface for [MedSAM](https://github.com/bowang-lab/MedSAM)

An easy to use GUI for bounding box prompted and automatic segmentation of medical images.

## Requirements
- python > 3.10.*
- pytorch
- torchvision
- numpy
- opencv-python

## Usage
- create a directory inside `MedSAM_GUI` and rename it to `checkpoints`.
- Download the finetuned MedSAM [checkpoint](https://drive.google.com/file/d/1bxsrFWT5NXH-ZhWht-KU9vDk-nGSFNa5/view?usp=drive_link)
- Optionally you can download the SAM [checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) which is not finetuned for medical images.
- move the files to `checkpoints` directory.

``` sh
# On linux or mac
conda activate <your-environment>
cd MedSAM_GUI 
pip install -r requirements.txt

python app.py
```
## Features
- [ ] Automatic segmentation of breast ultrasound images using trained Unet model.
- [ ] Semi-automatic segmentation of breast ultrasound or any medical images using box promted SAM mode.
- [ ] Risk of Malignancy prediction for breast ultrasound images.

