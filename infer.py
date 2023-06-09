import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import torchvision.transforms.functional as TF
from segment_anything.utils.transforms import ResizeLongestSide
from dataclasses import dataclass


@dataclass()
class Config:
    sam_checkpoint_dir: str = "finetune_weights/sam_vit_b_01ec64.pth"
    med_sam_checkpoint_dir: str = "finetune_weights/medsam_20230423_vit_b_0.0.1.pth"

    model_type: str = "vit_b"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def infer(image: torch.Tensor, bbox: tuple):
    config = Config()
    # initialize sam model

    image = (image - image.min()) / (image.max() - image.min()) * 255.0

    H, W = image.shape[2], image.shape[3]

    image = image.to(config.device)

    sam_model = sam_model_registry[config.model_type](
        checkpoint=config.med_sam_checkpoint_dir
    ).to(config.device)

    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
    # trans_image = image.unsqueeze(0).to(device)
    trans_image = TF.resize(image, (1024, 1024))

    box = sam_trans.apply_boxes(bbox, (H, W))

    box_tensor = torch.as_tensor(box, dtype=torch.float, device=config.device)
    # Get predictioin mask

    image_embeddings = sam_model.image_encoder(trans_image)  # (B,256,64,64)

    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=None,
        boxes=box_tensor,
        masks=None,
    )
    mask_predictions, _ = sam_model.mask_decoder(
        image_embeddings=image_embeddings.to(config.device),  # (B, 256, 64, 64)
        image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )
    mask_predictions = torch.sigmoid(mask_predictions)

    mask_predictions = (mask_predictions > 0.5).int() * 255
    torch.cuda.empty_cache()

    del sam_model

    return mask_predictions
