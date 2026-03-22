import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model import UNet
from utils import load_config


def debug_model():
    config = load_config("D:/portfolio/project1_Sigmentation/configs/config.yaml")
    device = torch.device('cpu')

    # Загрузка модели
    model = UNet(n_classes=1)
    checkpoint = torch.load("/src/weights/final_model_old.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Загрузка одного примера
    from dataset import SegmentationData
    data_root = Path(config['data']['root_dir'])
    x = [x for x in (data_root / config['data']['img_dir']).iterdir()]
    dataset = SegmentationData(
        x[:471],
        [x for x in (data_root / config['data']['mask_dir']).iterdir()],
        config,
        augment=False
    )

    img, mask = dataset[0]

    # Предсказание
    with torch.no_grad():
        img_batch = img.unsqueeze(0)
        pred = model(img_batch)
        pred_prob = torch.sigmoid(pred).squeeze().numpy()
        pred_binary = (pred_prob > 0.5).astype(np.float32)

    print(f" Статистика предсказания:")
    print(f"  Mean probability: {pred_prob.mean():.4f}")
    print(f"  Max probability: {pred_prob.max():.4f}")
    print(f"  Pixels > 0.5: {(pred_prob > 0.5).sum()}")
    print(f"  Pixels > 0.3: {(pred_prob > 0.3).sum()}")
    print(f"  Pixels > 0.1: {(pred_prob > 0.1).sum()}")

    # Визуализация
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img.permute(1, 2, 0))
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    axes[2].imshow(pred_prob, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (prob)\nmean={pred_prob.mean():.3f}")
    axes[2].axis('off')

    axes[3].imshow(pred_binary, cmap='gray')
    axes[3].set_title(f"Prediction (binary >0.5)\nsum={pred_binary.sum():.0f}")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig("debug_prediction.png", dpi=150)
    plt.show()

    # Проверка масок
    print(f"\n Проверка масок:")
    mask_sample = cv2.imread(str(list(data_root / config['data']['mask_dir']).glob("*.jpg")[0]), cv2.IMREAD_GRAYSCALE)
    print(f"  Уникальные значения в масках: {np.unique(mask_sample)}")
    print(f"  Min: {mask_sample.min()}, Max: {mask_sample.max()}")
    print(f"  % белых пикселей: {(mask_sample > 127).sum() / mask_sample.size * 100:.2f}%")


if __name__ == "__main__":
    debug_model()