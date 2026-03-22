# Сигментация дорожного покрытия с помощью модели UNet 

## Стурктура проекта
```
-src
--\dataset.py
--\debug.py
--\losses.py
--\metrics.py
--\train.py
--\utils.py
```

## Немного о проекте 

### Архитектура 
В репозитории представлен проект полной реализации модели для сегментирования объектов на основе Unet архитектуры.
Модель обучалась на датасете CRACK500 ('https://www.kaggle.com/datasets/aryansingh2809/crack500').
В начале работы предполагалось использовать предобученную модель ResNet18 для улучшения модели Unet т.к. ResNet уже обучен на ImageNet, однако
из-за проблем возникших в ходе работы было принято решения вручную реализовать архитектуру, частично похожую на ту, что была реализована в репозитории 
resUnet- ('https://github.com/AlexCh-info/resUnet-'). Из-за этого модель не может показать хорошие показатели (архитектура не позволяет).

### Loss (файл losses.py)
В качестве loss функции были выбраны две функции: простой DiceLoss и смежный BCE и DiceLoss функция, другие loss функции плохо подходят т.к. штрафуют модель 
не за то. В ходе обучения DiceLoss не использовался, он был реализован, как класс, который был использован при создании loss функции BCEDiceLoss, которая и была использована 
для штрафов модели.

### Метрики (файл metrics.py)
В качестве метрик были выбраны:
  - Intersection over Union (IoU), метрика реализована в функции calculate_iou
  - Dice Metric (Коэффициент Сёренсена-Дайса, DSC), метрика была реализована в функции calculate_dice
Для мониторинга метрик был создан класс MetricTracker, в который записывают метркии по ходу обучения, после завершения эпохи метрики обнуляются

### Датасет (файл dataset.py)
Для загрузки и обработки датасета был реализован класс SegmentationData в нем:
  - Принимаются списки путей на изображения и маски
  - Изменяется размер масок и изображений так, чтобы они были одинаковыми
  - Создаются torch.tensor для дальнейшего обучения модели
  - Проводится аугментация с использованием библиотеки albumentations
Так же реализована функция get_dataloaders, которая создает даталоудеры на основе torch.data.Dataset

### Проверка создания и загрузки датасета (файл utils.py)
В файле реализованы функции:
  - load_config, проверяет загрузился ли конфиг или нет
  - explore_dataset выводит информацию о данных и датасете
Пример работы программы:
```
Всего изображений 472
Всего масок 471
```
<img width="1470" height="345" alt="image" src="https://github.com/user-attachments/assets/700b47d9-3b2f-4f4a-ae15-35da9c13a85d" />

### Обучение (файл train.py)
В файле раеализованы пять функций:

  - train_epoch, эпоха train. Основные фун-ии: загрузка даталоудера, подключение режима (cpu - gpu), обновление весов (forward, backward) для тренировочной выборки.
  - val_epoch, все тоже самое, но только для тестовой выборки + мы просто проверяем модель так что она стоит в режиме eval().
  - save_checkpoint, сохраняет показатели модели (веса, эпоху, метрики, лосс).
  - load_checkpoint, фун-ия предназначена для fine-tuning (дообучения модели), что я делал неоднократно, фун-ия загружает модель, веса, оптимайзер и т.д., по эпохе скрипт определеят с какой эпохи надо начать дообучение.
  - plot_training_history, визуализация метрик, лоссов во время обучения и сохранения их графиков в локальный репозиторий.
<img width="1500" height="400" alt="training_history_progress" src="https://github.com/user-attachments/assets/35579edf-d91c-4b30-a53c-c61ed96f5f55" />

  - train, основная функция, тут происходит обучение или дообучение модели, так же реализованы доп. проверки.
Этапы обучения:

  - Первый этап 20 эпох:
    
      - После пятой эпохи результаты не особо утешительные (хотя если приглядеться, то можно найти на изображении границы трещины) (основное обучение):
        <img width="2400" height="600" alt="debug_prediction_check_5" src="https://github.com/user-attachments/assets/ce1b365f-5317-49eb-a66d-119be2ce75d8" />
      - Десятая эпоха (основное обучение):
        <img width="2400" height="600" alt="debug_prediction_check_10" src="https://github.com/user-attachments/assets/fead85f2-22e5-443d-ae16-292226467b1e" />
      - Пятнадцатая эпоха (основное обучение):
        <img width="2400" height="600" alt="debug_prediction_check_15" src="https://github.com/user-attachments/assets/e81bdbe9-1fee-4539-b503-fdf6e4849682" />
      - Двадцатая эпоха (основное обучение):
        <img width="2400" height="600" alt="debug_prediction_check_20" src="https://github.com/user-attachments/assets/f3db3dcd-5dc7-4c2b-9e9a-4fc122f32099" />
      - Двадцать пятая эпоха (основное обучение):
        <img width="2400" height="600" alt="debug_prediction_check_25" src="https://github.com/user-attachments/assets/e335eb81-68e2-4d09-b6be-8de8093158f2" />
      - Тридцатая эпоха (основное обучение):
        <img width="2400" height="600" alt="debug_prediction_check_30" src="https://github.com/user-attachments/assets/2a834559-9886-4977-b61d-e95ebed8f63c" />
      - Тридцать пятая эпоха (первый этап дообучения), просто дообучаем на большем кол-ве эпох:
        <img width="2400" height="600" alt="debug_prediction_check_35" src="https://github.com/user-attachments/assets/fd01fda2-9933-4353-b9c9-5c86e8c59a4d" />
      - Сороковая эпоха (первый этап дообучения):
        <img width="2400" height="600" alt="debug_prediction_check_40" src="https://github.com/user-attachments/assets/53afff77-a8c0-4db0-b7a1-1611adc88431" />
      - Сорок пятая эпоха (второй этап дообучения) повысил скорость обучения с 0.001 до 0.005, увеличил веса dice метрики, понизил iou, увеличил вес dice_loss, аугментация стала более агрессивной:
        <img width="2400" height="600" alt="debug_prediction_check_45" src="https://github.com/user-attachments/assets/f157648e-6dbf-4c1b-8f74-3f74c672a758" />
      - Пятидесятая эпоха (второй этап дообучения):
        <img width="2400" height="600" alt="debug_prediction_check_50" src="https://github.com/user-attachments/assets/f49ec63a-1265-4b19-8be8-ead0c9a5b1ac" />
      - Пятьдесят пятая эпоха (второй этап дообучения):
        <img width="2400" height="600" alt="debug_prediction_check_55" src="https://github.com/user-attachments/assets/1fba7235-8b9e-40e7-b391-2e7199af19c1" />
      - Шестидесятая эпоха (второй этап дообучения):
        <img width="2400" height="600" alt="debug_prediction_check_60" src="https://github.com/user-attachments/assets/6a918f8c-6453-4ec3-991d-d271c7279a01" />
      - Шестидесят пятая эпоха (второй этап дообучения):
        <img width="2400" height="600" alt="debug_prediction_check_65" src="https://github.com/user-attachments/assets/237d0dea-fe56-4022-b9f7-6a03aed5dc87" />
      - Семидесятая эпоха (второй этап дообучения) (заключительная):
        <img width="2400" height="600" alt="debug_prediction_check_70" src="https://github.com/user-attachments/assets/8fc37a4c-f0af-4778-86aa-666699968e79" />
        
Посмотрим на сколько хороша модель просто обученая, дообученная на одном этапе и финальная модель:

  - Просто обученная модель:
    <img width="2400" height="600" alt="image" src="https://github.com/user-attachments/assets/eab07565-6bc8-4188-a1bb-751c4a3e3cdd" />
  - Модель с одним дообучением:
    <img width="2400" height="600" alt="debug_prediction_best_old" src="https://github.com/user-attachments/assets/47b0d025-a3c8-4428-b96f-6b63729aa253" />
  - Финальная модель:
    <img width="2400" height="600" alt="debug_prediction_best_final" src="https://github.com/user-attachments/assets/013210b7-4559-484d-a303-705576510dea" />

### Что следует сделать в сл. релизе:
Смотря на эту модель хочется плакать т.к. она вообще не умеет обощать и запоминать информацию, обучение идет тяжело, так что в сл. релизе необходимо все реализовать архитектуру с включенной в нее resnet18.

# Ссылка на Goole Disc с весами и моделями
https://drive.google.com/drive/folders/1Hq9SDdW2r078CjMxRNCnomWsy5gpi8Jk?usp=drive_link




