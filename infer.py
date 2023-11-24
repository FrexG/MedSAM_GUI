import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models.resnet import resnet18

from elunet.elunet import ELUnet
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry

from dataclasses import dataclass


@dataclass()
class Config:
    sam_checkpoint_dir: str = "checkpoints/medsam_vit_b.pth"
    model_type: str = "vit_b"
    device: str = "cpu"
    unet_size = (256, 256)
    resnet_size = (224, 224)
    prob_cancer = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
    num_classes = 3


config = Config()


def load_sam():
    # initialize sam model
    sam_model = sam_model_registry[config.model_type](
        checkpoint=config.sam_checkpoint_dir
    ).to(config.device)

    return sam_model


def load_unet():
    elunet = ELUnet(1, 1, 16)
    elunet.to("cpu")
    elunet.load_state_dict(
        torch.load("checkpoints/e_fold_1_eps_0.pth", map_location="cpu")
    )
    return elunet


def load_resnet():
    resnet_model = resnet18(weights=None)
    # change the last fc layer
    # resnet_model.conv1 = nn.Conv2d(
    #    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    # )

    resnet_model.fc = nn.Linear(
        resnet_model.fc.in_features, config.num_classes, bias=True
    )
    resnet_model.load_state_dict(
        torch.load("checkpoints/resnet18_birads_2023-11-22.pth", map_location="cpu")
    )

    return resnet_model


@torch.inference_mode()
def infer(image: torch.Tensor, bbox, sam_model):
    H, W = image.shape[1], image.shape[2]
    image = image.to(config.device)
    image = image.unsqueeze(0)

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

    return mask_predictions


@torch.inference_mode()
def infer_auto(image: torch.Tensor, elunet):
    # resize to (256,256)
    input_image = TF.resize(image, config.unet_size).unsqueeze(0)

    print(f"{input_image.shape=}")

    mask_predictions = elunet(input_image)

    return mask_predictions


@torch.inference_mode()
def birads_infer(image: torch.Tensor, resnet):
    # input image to resnet is (224,224)
    # resize
    _image = TF.resize(image, config.resnet_size)
    # add batch
    _image = _image.unsqueeze(0)
    # get prediction
    logits = resnet(_image)
    # get softmax probablity
    print(f"{logits.shape=}")
    print(f"{logits=}")

    probs = torch.softmax(logits, dim=1)
    # get the class label
    pred_class = torch.argmax(probs, dim=1)
    # clear
    del _image
    del logits
    del probs
    return config.prob_cancer[pred_class.item()]
