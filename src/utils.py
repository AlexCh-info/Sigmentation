import random
import cv2
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

def load_config(config_path="D:/portfolio/project1_Sigmentation/configs/config.yaml"):
    """Загружаем конфигурацию"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def explore_dataset(config):
    """Считываем файлы, проверяем размеры, показываем примеры"""
    data_root = Path(config['data']['root_dir'])
    img_dir = data_root / config['data']['img_dir']
    mask_dir = data_root / config['data']['mask_dir']

    #Сортируем, чтобы картинки и маски совпадали по индексам
    img_paths = list(img_dir.iterdir())
    mask_paths = list(mask_dir.iterdir())

    print(f"Всего изображений {len(img_paths)}")
    print(f"Всего масок {len(mask_paths)}")

    if len(img_paths) == 0:
        raise ValueError("Нет релевантных фотографий")

    fig, axes = plt.subplots(5, 2, figsize=(10, 20))

    indices = random.sample(range(len(img_paths)), min(5, len(img_paths)))

    for i, idx in enumerate(indices):
        img = cv2.imread(str(img_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"image {idx}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f"Mask {idx}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    config = load_config()
    explore_dataset(config)