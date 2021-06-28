## Тестовое задание на позицию Computer Vision Engineer в OSAI

### Метод и Результаты:
В результате обучения был получен macro-F1 score, равный 0.953972

Результат, безусловно, можно улучшить, однако, задание этого не требует.

Для обучения был выбран Resnet50. Особой причины этому нет. Просто это хорошая надежная модель, которая есть в торче, и ей всегда можно довериться.
Loss - CategoricalCrossEntropy, оптимайзер SGD.

### Запуск:

+ Устанавливем nvidia-docker и настраиваем его
+ Билдим докерфайл 
```docker build --tag osai_test:1.0 dockerfile/```
+ Запуск докера
```docker run --gpus=all --ipc=host -v папка_репозитория:/workspace -it osai_test:1.0 bash```

#### Скачивание файлов:
+ Для скачивания нужно использовать скрипт download_multiprocess.py

```python3 download_multiprocess.py --input_dir train```

```python3 download_multiprocess.py --input_dir test```

#### Обучение
+ Если нужно, меняем параметры в ```ut/config.py```
+ Для обучения скрипт train.py (веса будут сохраняться каждую эпоху в папку weights/)

```python3 train.py --path_to_train_csv train_images/downloaded.csv --path_to_test_csv test_images/downloaded.csv --batch_size 32```

+ Тест
```python3 demo.py -image test_images/formula1/72811.jpeg -model best_model.pt```






