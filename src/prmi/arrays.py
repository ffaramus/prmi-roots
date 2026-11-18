"""Petites opérations sur les images, communes à l'entraînement (`dataset`) et à l'inférence (`inference`) :
réduction, standardisation ImageNet, complétion à un multiple de 16 (taille exigée par le U-Net)."""
import numpy as np
import torch
from PIL import Image

MULTIPLE = 16
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def shrink(array, scale):
    """Réduit un tableau flottant (H, W) par moyenne de surface."""
    h, w = array.shape
    return np.asarray(Image.fromarray(array).resize((round(w / scale), round(h / scale)), Image.BOX))


def shrink_image(image, scale):
    """Réduit une image (H, W, 3) flottante, canal par canal."""
    return np.stack([shrink(image[..., c], scale) for c in range(image.shape[2])], axis=2)


def to_tensors(image, mask):
    """Standardisation ImageNet, puis tenseurs (3, H, W) et (1, H, W)."""
    x = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(mask[None].astype(np.float32))


def to_display(x):
    """Inverse de la standardisation : tenseur (3, H, W) vers image RGB uint8, pour l'affichage."""
    img = x.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def pad_image(image, mask):
    """Complète l'image en bas et à droite (miroir) pour que ses côtés soient multiples de 16. Le masque reste entier."""
    h, w = image.shape[:2]
    pad = ((0, -h % MULTIPLE), (0, -w % MULTIPLE), (0, 0))
    return to_tensors(np.pad(image, pad, mode="reflect"), mask)
