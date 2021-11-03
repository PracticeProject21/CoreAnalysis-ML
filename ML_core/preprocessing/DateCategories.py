import numpy as np
import pandas as pd

root = 'C:/Users/tolik/information_technology/third_year/practice_project/CoreAnalysis-ML'
data = pd.read_csv('{}/data_for_study/data.csv'.format(root))


all_photos = data['photo_id'].unique().tolist()
all_tasks = data['task_id'].unique().tolist()

ultraviolet_photos = data[data['photo_type']=='УФ']['photo_id'].unique().tolist()
daylight_photos = data[data['photo_type']=='ДС']['photo_id'].unique().tolist()

# оставим часто встречающиеся породы
delete = data[data['segment_value'] \
                    .isin(['Глинисто-кремнистая порода',
                           'Уголь',
                           'Аргиллит углистый', 'Алевролит',
                           'Карбонатная порода','Известняк',
                           'Глина аргиллитоподобная'])] \
                    ['photo_id'].unique().tolist()

# оставим только часто встречающиеся породы
photo_for_train = list(set(all_photos) - set(delete))
ultra_for_model = ultraviolet_photos
day_for_model = [i for i in daylight_photos if i in photo_for_train]

# Сделаем из текущих масок one-hot encoded маски
colors_for_ultraviolet = {
        'Отсутствует': 20,
        'Насыщенное': 21,
        'Карбонатное': 22
}
colors_for_daylight = {
        'Переслаивание пород': 70,
        'Алевролит глинистый': 71,
        'Песчаник': 72,
        'Аргиллит': 73,
        'Глинисто-кремнистая порода': 74,
        'Песчаник глинистый': 75,
        'Уголь': 76,
        'Аргиллит углистый': 77,
        'Алевролит': 78,
        'Карбонатная порода': 79,
        'Известняк': 80,
        'Глина аргиллитоподобная': 81,
        'Разлом': 82,
        'Проба': 83}

# Всего 3 класса для УФ и 14 классов для ДС ==> всего 14 цветов
labels_colors = np.array([(0, 0, 0),        # 1 Переслаивание / отсутствует
                          (128, 0, 128),    # 2 Алевролит глинистый / насыщенное
                          (250, 233, 0),    # 3 Песчаник / Карбонатное
                          (0, 128, 90),     # 4 Аргиллит
                          (192, 128, 0),    # 5 Глинисто-кремнистая порода
                          (227, 178, 248),  # 6 Песчаник глинистый
                          (128, 128, 128),  # 7 Уголь
                          (0, 250, 221),    # 8 Аргиллит углистый
                          (64, 64, 192),    # 9 Алевролит
                          (255, 5, 255),    # 10 Карбонатная порода
                          (230, 5, 20),     # 11 Известняк
                          (124, 0, 30),     # 12 Глина аргиллитоподобная
                          (111, 247, 0),    # 13 Разлом
                          (0, 206, 247)])   # 14 Проба