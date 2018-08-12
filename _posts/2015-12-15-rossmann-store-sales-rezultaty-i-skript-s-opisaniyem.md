---
layout: post
title: Rossmann Store Sales результаты и скрипт с описанием 
date: 2015-12-15 23:52:00 +0700
categories: article
tags: kaggle pandas python 
author: Гуров Павел
---

![rossman-1.png](/assets/img/rossman-1.png){:class="img-fluid"}

Вчера завершился конкурс [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) от Kaggle. Это второе мое участие в подобных мероприятиях. Участие в таком конкурсе дает очень много полезных навыков и знаний. Можно сравнить с большой задачей, которую ставят для страны (отправить человека на луну), которая тянет за собой развитие науки.

Чтобы некоторые знания все же не потерялись, я решил выложить скрипт, который который обеспечил мне 1468 место из 3423. В 50% я попал :)

``` python
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from datetime import datetime
from sklearn import cross_validation

import xgboost as xgbb

# преобразуем строки в числа, чтобы можно было скармливать алгоритмам
def dataPrepare(data):
    data['StateHoliday'] = data['StateHoliday'].map({
        '0': 0,
        'a': 1,
        'b': 2,
        'c': 3,
        0: 0
    }).astype(int)

    data['Open'] = data['Open'].fillna(1) #CHECK
    return data

train_file = 'train.csv'
test_file = 'test.csv'
output_file = 'predictions.csv'


# Читаем данные, добавляем  недостающие столбцы
# индекс у нас тут - время
# Парсим дату
train = pd.read_csv(train_file, index_col='Date', parse_dates=['Date'])
train['Date'] = train.index
train['Week'] = train.index.week
train['Year'] = train.index.year
train['Month'] = train.index.month
train['Quarter'] = train.index.quarter
train = dataPrepare(train)

test = pd.read_csv(test_file, parse_dates=['Date'])

test = dataPrepare(test)
# dt - для работы с datetime
test['Week'] = test.Date.dt.week
test['Year'] = test.Date.dt.year
test['Month'] = test.Date.dt.month
test['Quarter'] = test.Date.dt.quarter


stories = [1,3,7,8,9]#set(test.Store.values)

stories = set(test.Store.values)

# Столбцы для предсказаний
predictors = [
    'DayOfWeek',
    # 'Date',
    'Week',
    'Year',
    'Month',
    'Quarter',
    # 'Sales2',
    # 'Customers',
    'Open',
    'Promo',
    'StateHoliday',
    'SchoolHoliday'
]

# Параметры для XGBoost алгоритма
params = {"objective": "reg:linear",
    "eta": 0.02,
    "booster": "gbtree",
    "max_depth": 10,
     "min_child_weight": 15,
    "silent": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.7}
num_trees=1000

# Эта функци для дискредитации данных продаж.
# Берётся распределение продаж и делится на равновероятные участки. 
# Каждой продаже присваивается номер.
def saleCode(x, division):
    for i, value in enumerate(division[:-1]):
        if value - 1 < x < division[i + 1] + 1:
            return i
            break

print('Start group')
# Отличный алгоритм подсмотренный мною у лидеров.
# Группируются записи по трём столбцам и делается предположение, 
# что медиана будет продолжаться и в предсказываемых месяцах

columns = ['Store', 'DayOfWeek', 'Promo']
medians = train.groupby(columns)['Sales'].median()
medians = medians.reset_index()

test2 = pd.merge(test, medians, on = columns, how = 'left')
assert(len(test2) == len(test))

test2.loc[ test2.Open == 0, 'Sales' ] = 0

test['Sales'] = test2['Sales']
print('Group stories finished')

test.loc[:, 'Sales_rf'] = 0
test.loc[:, 'Sales_xgb'] = 0

# Дальше по каждому магазину рассчет отдельно
for s in stories:

    # Первый алгоритм - распределение и RandomForest
    f1_full = train[train.Store==s]
    f1 = DataFrame(f1_full[f1_full.Open==1])
    count,division = np.histogram(f1.Sales.values, 7)

    storeTrue = {}
    for i, value in enumerate(division[:-1]):
        storeTrue[i] = np.mean([value, division[i+1]])

    f1.loc[:, 'Sales2'] = f1.apply(lambda row: saleCode(row['Sales'], division), axis=1)
    alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=3, min_samples_leaf=3)
    alg.fit(f1[predictors], f1['Sales2'])
    t1_full = test[test.Store==s]
    t1 = t1_full[t1_full.Open==1]
    predictions = alg.predict(t1[predictors])
    test.loc[(test.Store==s) & (test.Open==1), 'Sales_rf'] = list(map(lambda x: storeTrue[x], predictions))
    test.loc[(test.Store==s) & (test.Open==0), 'Sales_rf'] = 0
    print ('Random forest store done: ' + str(s))


    # Второй алгоритм - XGBoost
    train1 = train[train.Store==s]
    test1 = test[test.Store==s]
    gbm1 = xgbb.train(params, xgbb.DMatrix(train1[predictors], train1["Sales"]), num_trees)
    test_probs = gbm1.predict(xgbb.DMatrix(test.loc[test.Store==s, predictors]))
    test.loc[test.Store==s, 'Sales_xgb'] = test_probs
    print ('XGBoost store done: ' + str(s))


test.loc[test.Open==0, 'Sales_xgb'] = 0

# Среднее значение для для всех трёх алгоритмов 
test.loc[:, 'Sales'] = test['Sales']/3.0 + test['Sales_rf']/3.0 + test['Sales_xgb']/3.0

test.loc[test.Open==0, 'Sales'] = 0

test.head(12)
test.describe()

assert(test2.Sales.isnull().sum() == 0)

test[[ 'Id', 'Sales' ]].to_csv(output_file, index = False)

print("Up the leaderboard!")
``` 

*Перенёс сюда из [жж](https://gurovpavel.livejournal.com/2400.html)*
