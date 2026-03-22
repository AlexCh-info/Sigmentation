import torch


def calculate_iou(pred, target, threshold=0.3, smooth=0.5):
    # Порог к предсказанию
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    # Приводим target к тому же типу, что и pred_binary
    if target.dtype != pred_binary.dtype:
        target = target.to(pred_binary.dtype)

    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def calculate_dice(pred, target, threshold=0.3, smooth=1e-6):
    """
    Dice Coefficient
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    # Приводим target к тому же типу, что и pred_binary
    if target.dtype != pred_binary.dtype:
        target = target.to(pred_binary.dtype)

    intersection = (pred_binary * target).sum()
    dice = (2. * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)
    return dice.item()

class MetricTracker:
    """
    Класс для отслеживания метрик
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.iou_sum = 0
        self.dice_sum = 0
        self.count = 0
        self.loss_sum = 0

    def update(self, pred, target, loss):
        """ Обновляет метрики после одного батча"""
        batch_size = pred.size(0)
        self.iou_sum += calculate_iou(pred, target) * batch_size
        self.dice_sum += calculate_dice(pred, target) * batch_size
        self.loss_sum += loss.item() * batch_size
        self.count += batch_size

    def get_average(self):
        """ Возвращает среднее значение за эпоху"""
        if self.count == 0:
            return {'loss': 0, "iou": 0, "dice": 0}
        return {
            "loss": self.loss_sum / self.count,
            "iou": self.iou_sum / self.count,
            "dice": self.dice_sum / self.count
        }
