from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


class SegmentationData(Dataset):
    """
    Датасет для сегментации.
    Принимает либо пути к папкам, либо готовые списки путей к файлам.
    """

    def __init__(self, img_source: list, mask_source: list, config, augment=False):
        self.config = config
        self.augment = augment
        self.img_paths = img_source
        self.mask_paths = mask_source


        # Проверка на пустоту
        if len(self.img_paths) == 0:
            raise ValueError("Список изображений пуст! Проверьте пути и расширения файлов.")
        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError(f"Несоответствие пар! Изображений: {len(self.img_paths)}, Масок: {len(self.mask_paths)}")

        print(f"Инициализировано датасет: {len(self.img_paths)} пар файлов.")

        #Аугментации
        self.transform = A.Compose([
            A.Resize(config['image']['height'], config['image']['width']),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            ], p=0.3),
        ], is_check_shapes=True) if augment else A.Compose([
            A.Resize(config['image']['height'], config['image']['width']),
        ], is_check_shapes=True)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, item) -> torch.tensor:
        # Чтение изображения
        img_path = str(self.img_paths[item])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Чтение маски
        mask_path = str(self.mask_paths[item])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Не удалось прочитать маску: {mask_path}")

        if img.shape[:2] != mask.shape:
            print('Изменяем размер маски')
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Применяем аугментацию
        try:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        except Exception as e:
            print(f"Ошибка при аугментации файла {img_path}: {e}")
            raise e

        img = img.astype(np.float32) / 255.0

        # Бинаризация маски (0 и 1)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # Переводим в тензоры
        # Image: (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        # Mask: (H, W) -> (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask


def get_dataloaders(config):
    """
    Создаем dataloader с правильным разделением на train/val
    """
    data_root = Path(config['data']['root_dir'])
    img_dir = data_root / config['data']['img_dir']
    mask_dir = data_root / config['data']['mask_dir']

    #Получаем ВСЕ файлы и синхронизируем их сразу
    all_raw_imgs = list(img_dir.iterdir())
    print(len(all_raw_imgs))
    all_raw_masks = list(mask_dir.iterdir())
    print(len(all_raw_masks))

    # Здесь просто продублируем логику sync_file_lists
    mask_dict = {i.stem: m for m, i in zip(all_raw_masks, all_raw_imgs)}
    synced_imgs = [p for p in all_raw_imgs if p.stem in mask_dict.keys()]
    synced_masks = [mask_dict[p.stem] for p in synced_imgs]

    # Разбиваем на train/val
    split_idx = int(len(synced_imgs) * config['data']['train_split'])

    # Перемешиваем перед разделением
    import random
    random.seed(42)  # Для воспроизводимости
    combined = list(zip(synced_imgs, synced_masks))
    random.shuffle(combined)
    synced_imgs, synced_masks = zip(*combined)
    synced_imgs = list(synced_imgs)
    synced_masks = list(synced_masks)

    train_imgs = synced_imgs[:split_idx]
    train_masks = synced_masks[:split_idx]

    val_imgs = synced_imgs[split_idx:]
    val_masks = synced_masks[split_idx:]

    # Создаем датасеты
    # Передаем сразу списки путей
    train_dataset = SegmentationData(train_imgs, train_masks, config, augment=True)
    val_dataset = SegmentationData(val_imgs, val_masks, config, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )

    return train_loader, val_loader


# Проверка
if __name__ == '__main__':
    from utils import load_config

    config = load_config()
    try:
        train_loader, val_loader = get_dataloaders(config)

        print(f'Train batches: {len(train_loader)}')
        print(f'Val batches: {len(val_loader)}')

        imgs, masks = next(iter(train_loader))
        print(f'Batch shape - Images: {imgs.shape}, Masks: {masks.shape}')
    except Exception as e:
        print(f"Ошибка: {e}")