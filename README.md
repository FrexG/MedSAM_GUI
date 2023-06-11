# GUI Interface for [MedSAM](https://github.com/bowang-lab/MedSAM)

A very basic graphical interface for bounding box prompted segmentation of medical images.

## Requirements
- python > 3.10.*
- pytorch
- torchvision
- numpy
- opencv-python
- pillow

``` sh
# On linux or mac
conda activate <your-environment>
cd MedSAM_GUI 
pip install -r requirements.txt
```
## Usage
- create a directory inside `MedSAM_GUI` and rename it to `finetune_weights`.
- Download the finetuned MedSAM [checkpoint](https://drive.google.com/file/d/1bxsrFWT5NXH-ZhWht-KU9vDk-nGSFNa5/view?usp=drive_link)
- Optionally you can download the SAM [checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) which is not finetuned for medical images.
- move the files to `finetune_weights` directory.

``` sh
python app.py
```
---
# On progress ... 
Feel free to contribute
## Todo List
- [ ] Support dynamic image size
- [ ] Support text prompting.
- [X] Modern GUI :)
