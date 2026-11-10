"""U-Net à encodeur EfficientNet-B0 pré-entraîné sur ImageNet. Une seule sortie : la probabilité de racine par pixel."""
import segmentation_models_pytorch as smp
import torch


def build_model(weights="imagenet"):
    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=weights,
        encoder_depth=4,                     # carré de 256 -> goulot de 16x16
        decoder_channels=(128, 64, 32, 16),
        in_channels=3,
        classes=1,
    )


def count_used_parameters(model):
    """Paramètres réellement utilisés : avec encoder_depth=4, smp garde les derniers blocs de l'encodeur sans les appeler."""
    device = next(model.parameters()).device
    model.eval()
    model(torch.zeros(1, 3, 256, 256, device=device)).sum().backward()
    n = sum(p.numel() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return n
