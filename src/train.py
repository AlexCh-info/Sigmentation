import os
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Импорт модулей
from dataset import get_dataloaders
from utils import load_config
from model import UNet
from losses import BCEDiceLoss
from metrics import MetricTracker


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """ Одна эпоха обучения"""
    model.train()
    metrics = MetricTracker()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(imgs)

        # Вычисляем лосс
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Обновляем метрики
        metrics.update(outputs, masks, loss)

        avg_metrics = metrics.get_average()
        pbar.set_postfix({
            'loss': f"{avg_metrics['loss']:.4f}",
            'iou': f"{avg_metrics['iou']:.4f}",
            'dice': f"{avg_metrics['dice']:.4f}"
        })
    return metrics.get_average()


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    metrics = MetricTracker()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        metrics.update(outputs, masks, loss)

    return metrics.get_average()


def save_checkpoint(model, optimizer, epoch, metrics, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    print(' Model was saved')


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    Загружает веса модели, оптимизатора и номер эпохи
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Продолжаем со следующей эпохи
        print(f" Веса загружены из {checkpoint_path}")
        print(f" Продолжаем с эпохи {start_epoch}")
        return start_epoch, checkpoint.get('metrics', {})
    else:
        print(" Чекпоинт не найден. Начинаем обучение с нуля.")
        return 1, {}


def plot_training_history(history, save_path='D:/portfolio/project1_Sigmentation/training_history_progress.png'):
    if len(history['train_loss']) == 0:
        print("️ История пуста, не строим график")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train loss')
    axes[0].plot(epochs, history['val_loss'], label='Validation loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # IoU
    axes[1].plot(epochs, history['train_iou'], label='Train IoU')
    axes[1].plot(epochs, history['val_iou'], label='Validation IoU')
    axes[1].set_title('IoU Score')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].legend()
    axes[1].grid(True)

    # Dice
    axes[2].plot(epochs, history['train_dice'], label='Train Dice')
    axes[2].plot(epochs, history['val_dice'], label='Validation Dice')
    axes[2].set_title('Dice Coefficient')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Dice')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f'График сохранён: {save_path}')
    except Exception as e:
        print(f"Cant save picture with error {str(e)}")
    plt.show()


def train(config_path="D:/portfolio/project1_Sigmentation/configs/config.yaml"):
    try:
        config = load_config(config_path)
        print(' Config loaded')
    except Exception as e:
        print(f" Path not found with error {str(e)}")
        return

    try:
        os.makedirs(config['paths']['weights_dir'], exist_ok=True)
        os.makedirs(config['paths']['logs_dir'], exist_ok=True)
        print(' Directories created')
    except Exception as e:
        print(f" Error with making directories: {str(e)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Device: {device}")

    try:
        train_loader, val_loader = get_dataloaders(config)
        print(f" Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except ValueError as e:
        print(f" Невозможно загрузить dataloaders: {str(e)}")
        return

    try:
        model = UNet(n_classes=1)
        model = model.to(device)
        print('Model initialized')
    except Exception as e:
        print(f"Cant initialize model with error: {str(e)}")
        return

    # Loss & Optimizer
    criterion = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # --- ЗАГРУЗКА ЧЕКПОИНТА (НОВЫЙ КОД) ---
    # Приоритет: final_model_old.pth (последнее состояние)
    checkpoint_path = f"{config['paths']['weights_dir']}/final_model_old.pth"
    start_epoch, last_metrics = load_checkpoint(model, optimizer, scheduler, checkpoint_path, device)

    # Загрузка истории
    history_path = f"{config['paths']['logs_dir']}/training_history_old.pth"
    if os.path.exists(history_path) and start_epoch > 1:
        try:
            history = torch.load(history_path)
            print(f"История обучения загружена ({len(history['train_loss'])} эпох)")
        except:
            history = {
                'train_loss': [], 'val_loss': [],
                'train_iou': [], 'val_iou': [],
                'train_dice': [], 'val_dice': []
            }
    else:
        history = {
            'train_loss': [], 'val_loss': [],
            'train_iou': [], 'val_iou': [],
            'train_dice': [], 'val_dice': []
        }

    # Параметры для обучения
    best_val_loss = last_metrics.get('loss', float('inf'))

    # Цикл обучения
    num_epochs = config['training']['num_epochs']

    # Если продолжаем обучение, но epochs в конфиге меньше чем уже прошло — увеличиваем
    if start_epoch > num_epochs:
        print(f"Увеличиваем num_epochs с {num_epochs} до {start_epoch + 40}")
        num_epochs = start_epoch + 20

    print(f'Training from epoch {start_epoch} to {num_epochs}....\n')

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Эпоха {epoch}/{num_epochs}")

        # Обучение
        try:
            # Тренировочные показатели
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            print(
                f"Train loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}")

            # Валидационные показатели
            val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
            print(
                f"Val loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")
        except Exception as e:
            print(f'Метрики не были загружены с ошибкой: {str(e)}')
            return

        try:
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_iou'].append(train_metrics['iou'])
            history['val_iou'].append(val_metrics['iou'])
            history['train_dice'].append(train_metrics['dice'])
            history['val_dice'].append(val_metrics['dice'])
        except Exception as e:
            print(f'Невозможно добавить метрику в историю с ошибкой: {str(e)}')

        scheduler.step(val_metrics['loss'])

        # Сохранение лучшей модели
        try:
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(
                    model, optimizer, epoch, val_metrics,
                    f"{config['paths']['weights_dir']}/best_model.pth"
                )
                print('Новая лучшая модель сохранена!')
        except Exception as e:
            print(f"Невозможно сохранить лучшую модель, ошибка {str(e)}")

        # Сохранение чекпоинта каждые 5 эпох
        try:
            if epoch % 5 == 0:
                save_checkpoint(
                    model, optimizer, epoch, val_metrics,
                    f"{config['paths']['weights_dir']}/checkpoint_epoch_{epoch}.pth"
                )
                print(f"Чекпоинт сохранён на эпохе {epoch}")
        except Exception as e:
            print(f"Чекпоинт на эпохе {epoch} невозможно сохранить, ошибка {str(e)}")

        # Сохранение истории после каждой эпохи
        try:
            torch.save(history, history_path)
        except:
            pass

    try:
        # Сохраняем финальную модель
        save_checkpoint(
            model, optimizer, num_epochs, val_metrics,
            f"{config['paths']['weights_dir']}/final_model.pth"
        )
        print('Обучение завершено')
        print(f'Лучший Val loss: {best_val_loss:.4f}')
    except Exception as e:
        print(f'Невозможно сохранить финальную модель из-за ошибки: {str(e)}')

    # Построение графика
    plot_training_history(history)

    return history


if __name__ == '__main__':
    history = train()





























