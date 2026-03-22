import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


class DoubleConv(nn.Module):
    """[Conv -> BN -> ReLU] -> [Conv -> BN -> ReLU]"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upsample + concat with skip + DoubleConv"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Выравнивание размеров на случай небольших расхождений
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet_ResNet(nn.Module):
    """
    U-Net с ResNet18 в качестве энкодера
    Вход: 256x256 → Выход: 256x256
    """

    def __init__(self, n_classes=1, in_channels=3, weights=ResNet18_Weights.IMAGENET1K_V1):
        super().__init__()

        resnet = models.resnet18(weights=weights)

        # Адаптация первого слоя под нужное количество каналов
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # === ENCODER (ResNet18) ===
        self.initial = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )  # 256x256 → 64x64, 64 канала
        self.layer1 = resnet.layer1  # 64x64 → 64x64, 64 канала (skip для up3)
        self.layer2 = resnet.layer2  # 64x64 → 32x32, 128 каналов (skip для up2)
        self.layer3 = resnet.layer3  # 32x32 → 16x16, 256 каналов (skip для up1)
        self.layer4 = resnet.layer4  # 16x16 → 8x8, 512 каналов (bottleneck)

        # === DECODER ===
        # Три шага с skip-connections
        self.up1 = Up(512, 256, 256)  # 8x8 → 16x16
        self.up2 = Up(256, 128, 128)  # 16x16 → 32x32
        self.up3 = Up(128, 64, 64)  # 32x32 → 64x64

        # Два шага без skip-connections (нет фичей энкодера на 128×128 и 256×256)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 64x64 → 128x128
            DoubleConv(32, 64)
        )
        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 128x128 → 256x256

        # Выходной слой
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # === Encoder ===
        x0 = self.initial(x)  # 64x64
        x1 = self.layer1(x0)  # 64x64 — skip для up3
        x2 = self.layer2(x1)  # 32x32 — skip для up2
        x3 = self.layer3(x2)  # 16x16 — skip для up1
        x4 = self.layer4(x3)  # 8x8 — bottleneck

        # === Decoder ===
        d1 = self.up1(x4, x3)  # 16x16
        d2 = self.up2(d1, x2)  # 32x32
        d3 = self.up3(d2, x1)  # 64x64
        d4 = self.up4(d3)  # 128x128
        d5 = self.up5(d4)  # 256x256

        return self.final(d5)


# Проверка модели
if __name__ == "__main__":
    model = UNet_ResNet(n_classes=1, in_channels=3)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")