# подключение библиотек
import numpy as np
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from itertools import combinations

# Путь к массивам
path = 'data\\'

# Тестирование по "Алгоритму идентификации A"
# По наибольшему количеству побед в примерах класса
# model - модель
# x, y - тестовые массивы
# win_count - количество классов-победителей для публикации
# caption - заголовок отчета
def testing_report_a(model, x, y, win_count=3, caption=''):
    print(caption)
    # предсказание для всех примеров
    predict = model.predict_proba(x)
    # количество классов
    class_count = np.unique(y).shape[0]
    # проход по классам и генерация словаря для каждого класса
    n = 0
    for i in range(class_count):
        # выборка предсказания по классу
        predict_cl = predict[np.where(y == i)]
        # победители по примерам
        winners = predict_cl.argmax(axis=1)
        # подсчет побед для классов
        winners_counts = []
        for cl in range(class_count):
            winners_counts.append(winners[winners==cl].shape[0])
        # максимум и список победителей
        max_val = max(winners_counts)
        win_list = []
        for j in range(class_count):
            if winners_counts[j] == max_val:
                win_list.append(j)
        # кортеж из классов победителей
        win_list = tuple(win_list)
        # признак распознавания
        if (i == win_list[0]) and (len(win_list) == 1):
            forecast = 'распознан'
            n += 1
        else:
            forecast = 'не распознан'
        # печать
        print('Класс', '{: >2}'.format(i), '-',
              '{: >12}'.format(forecast), '| Победитель:', end='')
        print('{: >3}'.format(max_val), win_list)
    print('\nИтог тестирования:', n, 'из', class_count, 'распознаны верно...')

# Тестирование по "Алгоритму идентификации B"
# По наибольшим значениям сумм вероятностей в примерах класса
# model - модель
# x, y - тестовые массивы
# win_count - количество классов-победителей для публикации
# caption - заголовок отчета
def testing_report_b(model, x, y, win_count=3, caption=''):
    print(caption)
    # предсказание для всех примеров
    predict = model.predict_proba(x)
    # количество классов
    class_count = np.unique(y).shape[0]
    # проход по классам и генерация словаря для каждого класса
    n = 0
    for i in range(class_count):
        # инициируем словарь
        d = {}
        # выборка предсказания по классу
        predict_cl = predict[np.where(y == i)]
        # суммирование вероятностей в список
        xl = list(predict_cl.sum(axis=0))
        # выделение лучших вероятностей
        max_vals = sorted(xl, reverse=True)[:win_count]
        # номера лучших классов с вероятностями
        for j in range(win_count):
            d[xl.index(max_vals[j])] = max_vals[j]
        # оценка прогноза
        if i == list(d.keys())[0]:
            forecast = 'распознан'
            n += 1
        else:
            forecast = 'не распознан'
        # печать
        print('Класс', '{: >2}'.format(i), '-',
              '{: >12}'.format(forecast), '| Вероятности: ', end='')
        for k in d:
            print('{: >10}'.format(str(round(d[k], 2))+' ('+str(k)+')'), end='  ')
        print()
    print('\nИтог тестирования:', n, 'из', class_count, 'распознаны верно...')

# Тестирование по "Алгоритму идентификации C"
# Первый этап - отбор лучших классов по наибольшим значениям сумм вероятностей
# в примерах класса, второй этап - определение класса по количеству побед
# между лучшими классами
# model - модель
# x, y - тестовые массивы
# win_count - количество лучших классов - победителей первого этапа
# caption - заголовок отчета
def testing_report_c(model, x, y, win_count=3, caption=''):
    print(caption)
    # предсказание для всех примеров
    predict = model.predict_proba(x)
    # количество классов
    class_count = np.unique(y).shape[0]
    # проход по классам и генерация словаря для каждого класса
    n = 0
    for i in range(class_count):
        # выборка предсказания по классу
        predict_cl = predict[np.where(y == i)]
        # суммирование вероятностей в список
        xl = list(predict_cl.sum(axis=0))
        # выделение лучших вероятностей
        max_vals = sorted(xl, reverse=True)[:win_count]
        # номера классов в призерах
        win_cl = [xl.index(max_vals[j]) for j in range(win_count)]
        # вытаскиваем столбцы призеров
        predict_win = np.vstack([predict_cl[:,win_cl[j]] for j in range(win_count)])
        # находим классы победителей по примерам среди призеров
        cls = predict_win.argmax(axis=0)
        # подсчитываем победителей финала по количеству побед в финале
        win_cl_val = [cls[cls==i].shape[0] for i in range(win_count)]
        # максимум и список победителей
        max_val = max(win_cl_val)
        win_list = []
        for j in range(win_count):
            if win_cl_val[j] == max_val:
                win_list.append(win_cl[j])
        # кортеж из классов победителей
        win_list = tuple(win_list)
        # признак распознавания
        if (i == win_list[0]) and (len(win_list) == 1):
            forecast = 'распознан'
            n += 1
        else:
            forecast = 'не распознан'
        # печать
        print('Класс', '{: >2}'.format(i), '-',
              '{: >12}'.format(forecast), '| Полуфинал:', end=' ')
        print('{: >12}'.format(str(tuple(win_cl))), '| Финал:', end=' ')
        print('{: >12}'.format(str(win_cl_val)), end='')
        print('{: >6}'.format(str(win_list)))
    print('\nИтог тестирования:', n, 'из', class_count, 'распознаны верно...')

# Тестирование по "Алгоритму идентификации D"
# Первый этап - отбор лучших классов по наибольшим значениям сумм вероятностей
# в примерах класса, второй этап - определение класса на основе подсчета
# количества побед попарно между лучшими классами
# model - модель
# x, y - тестовые массивы
# win_count - количество лучших классов - победителей первого этапа
# caption - заголовок отчета
def testing_report_d(model, x, y, win_count=3, caption=''):
    print(caption)
    # предсказание для всех примеров
    predict = model.predict_proba(x)
    # количество классов
    class_count = np.unique(y).shape[0]
    # проход по классам и генерация словаря для каждого класса
    n = 0
    for i in range(class_count):
        # выборка предсказания по классу
        predict_cl = predict[np.where(y == i)]
        # суммирование вероятностей в список
        xl = list(predict_cl.sum(axis=0))
        # выделение лучших вероятностей
        max_vals = sorted(xl, reverse=True)[:win_count]
        # номера классов в полуфинале
        win_cl = [xl.index(max_vals[j]) for j in range(win_count)]
        # пары финалистов для сопоставления
        win_pairs = list(combinations(win_cl, 2))
        # обнуляем счет в финальных сравнениях пар
        rab_dict = {}
        for cl in win_cl:
            rab_dict[cl] = 0
        # сравниваем пары
        for win_pair in win_pairs:
            # вытаскиваем столбцы призеров
            predict_pair = np.vstack([predict_cl[:,win_pair[0]], predict_cl[:,win_pair[1]]])
            # находим классы победителей в паре
            cls = predict_pair.argmax(axis=0)
            p1, p2 = cls[cls==0].shape[0], cls[cls==1].shape[0]
            if p1 > p2:
                rab_dict[win_pair[0]] += 2
            elif p2 > p1:
                rab_dict[win_pair[1]] += 2
            else:
                rab_dict[win_pair[0]] += 1
                rab_dict[win_pair[1]] += 1
        # максимум и список победителей
        max_val = max(list(rab_dict.values()))
        win_list = []
        for k in rab_dict:
            if rab_dict[k] == max_val:
                win_list.append(k)
        # кортеж из классов победителей
        win_list = tuple(win_list)
        # признак распознавания
        if (i == win_list[0]) and (len(win_list) == 1):
            forecast = 'распознан'
            n += 1
        else:
            forecast = 'не распознан'
        # печать
        print('Класс', '{: >2}'.format(i), '-',
              '{: >12}'.format(forecast), '| Полуфинал:', end=' ')
        print('{: >12}'.format(str(tuple(win_cl))), '| Финал:', end=' ')
        print('{: >6}'.format(str(win_list)))
    print('\nИтог тестирования:', n, 'из', class_count, 'распознаны верно...')

# Тестирование по "Алгоритму идентификации E"
# Первый этап - отбор лучших классов по наибольшим значениям сумм вероятностей
# в примерах класса, второй этап - определение класса на основе подсчета разницы
# побед и поражений по примерам попарно между лучшими классами
# model - модель
# x, y - тестовые массивы
# win_count - количество лучших классов - победителей первого этапа
# caption - заголовок отчета
def testing_report_e(model, x, y, win_count=3, caption=''):
    print(caption)
    # предсказание для всех примеров
    predict = model.predict_proba(x)
    # количество классов
    class_count = np.unique(y).shape[0]
    # проход по классам и генерация словаря для каждого класса
    n = 0
    for i in range(class_count):
        # выборка предсказания по классу
        predict_cl = predict[np.where(y == i)]
        # суммирование вероятностей в список
        xl = list(predict_cl.sum(axis=0))
        # выделение лучших вероятностей
        max_vals = sorted(xl, reverse=True)[:win_count]
        # номера классов в полуфинале
        win_cl = [xl.index(max_vals[j]) for j in range(win_count)]
        # пары финалистов для сопоставления
        win_pairs = list(combinations(win_cl, 2))
        # обнуляем счет в финальных сравнениях пар
        rab_dict = {}
        for cl in win_cl:
            rab_dict[cl] = 0
        # сравниваем пары
        for win_pair in win_pairs:
            # вытаскиваем столбцы призеров
            predict_pair = np.vstack([predict_cl[:,win_pair[0]], predict_cl[:,win_pair[1]]])
            # находим классы победителей попарно
            cls = predict_pair.argmax(axis=0)
            p1, p2 = cls[cls==0].shape[0], cls[cls==1].shape[0]
            rab_dict[win_pair[0]] += p1-p2
            rab_dict[win_pair[1]] += p2-p1
        # максимум и список победителей
        max_val = max(list(rab_dict.values()))
        win_list = []
        for k in rab_dict:
            if rab_dict[k] == max_val:
                win_list.append(k)
        # кортеж из классов победителей
        win_list = tuple(win_list)
        # признак распознавания
        if (i == win_list[0]) and (len(win_list) == 1):
            forecast = 'распознан'
            n += 1
        else:
            forecast = 'не распознан'
        # печать
        print('Класс', '{: >2}'.format(i), '-',
              '{: >12}'.format(forecast), '| Полуфинал:', end=' ')
        print('{: >12}'.format(str(tuple(win_cl))), '| Финал:', end=' ')
        print('{: >6}'.format(str(win_list)))
    print('\nИтог тестирования:', n, 'из', class_count, 'распознаны верно...')

# Обучение модели классификатора
# train_x, train_y
# file_name - файл для сохранения модели
# (если сохраненная модель существует, то она считывается)
def get_model_classifier(file_name, train_x, train_y):
    # попытка считать модель из файла
    try:
        print('\tПопытка прочитать модель из файла...', end='')
        model = CatBoostClassifier()
        model.load_model(file_name)
        print('загружена!')
    except:
        print('ошибка загрузки!')
        # подсчет количества классов
        classes = np.unique(train_y)
        class_count = classes.shape[0]
        # подсчет весов
        weights = compute_class_weight(class_weight='balanced',
                                       classes=classes, y=train_y)
        class_weights = dict(zip(classes, weights))
        # объект классификатора
        model = CatBoostClassifier(iterations = 1000,
                                   class_weights=class_weights,
                                   #verbose=0
                                   )
        # Обучение
        print('\tОбучающих примеров:', train_x.shape[0])
        print('\tКлассов:', class_count)
        print('\tПроцесс обучения...', end='')
        model.fit(train_x, train_y)
        print('Выполнено')
        print('\tСохранение модели...', end='')
        model.save_model(file_name)
        print('Выполнено')
    return model

# обучающие данные
train_data = {
    'True_Train': (np.load(path + 'train_x.npy'),
                   np.load(path + 'train_y.npy')),
    'OriginalGenerator_Train': (np.load(path + 'train_x_gan1.npy'),
                                np.load(path + 'train_y_gan1.npy')),
    'GANGenerator_Train': (np.load(path + 'train_x_gan2.npy'),
                           np.load(path + 'train_y_gan2.npy'))
}

# тестовые данные
test_data = {
    'тест-1': (np.load(path + 'test1_x.npy'),
               np.load(path + 'test1_y.npy')),
    'тест-2': (np.load(path + 'test2_x.npy'),
               np.load(path + 'test2_y.npy'))
}

# модели
models = {
    'True_Train': None,
    'OriginalGenerator_Train': None,
    'GANGenerator_Train': None
}

# обучение или чтение моделей
for key in models:
    print('Модель на основе:', key)
    models[key] = get_model_classifier(path + key + '.cbm',
                                       train_data[key][0],
                                       train_data[key][1])

# привязка названий к функциям алгоритмов
switch = {
        'A': testing_report_a,
        'B': testing_report_b,
        'C': testing_report_c,
        'D': testing_report_d,
        'E': testing_report_e,
    }

# ================ параметры тестирования и вывод результата ========================

# выбор алгоритма идентификации 'A', 'B', 'C', 'D', 'E'
alg_choice = 'E'

# выбор теста 'тест-1', 'тест-2'
test_choice = 'тест-2'

# выбор модели 'True_Train', 'OriginalGenerator_Train', 'GANGenerator_Train'
model_choice = 'True_Train'

# выбор параметра win_count
win_count_choice = 3

# вывод результата в зависимости от параметров
switch.get(alg_choice)(models[model_choice], test_data[test_choice][0],
                       test_data[test_choice][1], win_count_choice)