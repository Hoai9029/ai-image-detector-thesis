import timm


def build_model(name: str = "efficientnet_b0", pretrained: bool = True):
    return timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=1
    )