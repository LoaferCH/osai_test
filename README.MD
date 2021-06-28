## Тестовое задание на позицию Computer Vision Engineer в OSAI

+ Устанавливем nvidia-docker и настраиваем его
+ Билдим докерфайл docker build --tag osai_test:1.0 dockerfile/   
+ Запуск докера

```docker run --gpus=all --ipc=host -v /media/sasha/d4556992-6bd6-482d-9f2d-d5bc5d9a00ad4/delete_me/osai:/workspace -it osai_test:1.0 bash```

#### Скачивание файлов:
+ Для скачивания нужно использовать скрипт download_multiprocess.py

```python3 download_multiprocess.py --input_dir train```

```python3 download_multiprocess.py --input_dir test```

#### Обучение
+ Для обучения скрипт train.py

```python3 train.py --path_to_train_csv train_images/downloaded.csv --path_to_test_csv test_images/downloaded.csv --batch_size 32```


### Результаты:




