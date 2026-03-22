# src/model.py
import torch
import torch.nn as nn

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


class Down(nn.Module):
    """Уменьшение размера в 2 раза + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Увеличение размера в 2 раза + конкатенация + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    Классический U-Net для изображений 256x256
    """

    def __init__(self, n_classes=1, in_channels=3):
        super().__init__()

        # ENCODER
        self.inc = DoubleConv(in_channels, 64)  # 256 -> 256
        self.down1 = Down(64, 128)  # 256 -> 128
        self.down2 = Down(128, 256)  # 128 -> 64
        self.down3 = Down(256, 512)  # 64 -> 32
        self.down4 = Down(512, 1024)  # 32 -> 16

        # BOTTLENECK
        self.bottleneck = DoubleConv(1024, 1024)  # 16 -> 16

        # DECODER
        self.up1 = Up(1024, 512)  # 16 -> 32
        self.up2 = Up(512, 256)  # 32 -> 64
        self.up3 = Up(256, 128)  # 64 -> 128
        self.up4 = Up(128, 64)  # 128 -> 256

        # OUTPUT
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.inc(x)  # 256x256
        e2 = self.down1(e1)  # 128x128
        e3 = self.down2(e2)  # 64x64
        e4 = self.down3(e3)  # 32x32
        e5 = self.down4(e4)  # 16x16

        # Bottleneck
        b = self.bottleneck(e5)  # 16x16

        # Decoder
        d1 = self.up1(b, e4)  # 16->32, concat с e4 (32x32)
        d2 = self.up2(d1, e3)  # 32->64, concat с e3 (64x64)
        d3 = self.up3(d2, e2)  # 64->128, concat с e2 (128x128)
        d4 = self.up4(d3, e1)  # 128->256, concat с e1 (256x256)

        return self.final(d4)


# Проверка модели
if __name__ == "__main__":
    model = UNet(n_classes=1, in_channels=3)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")