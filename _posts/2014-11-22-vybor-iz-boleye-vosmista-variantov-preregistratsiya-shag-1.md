---
layout: post
title: Выбор из более восьмиста вариантов (пререгистрация, шаг 1)
date: 2014-11-22 23:04:00 +0700
categories: article
tags:  openprovider домены пререгистрации
author: Гуров Павел
---
После принятия решения организации ICANN о регистрации дополнительных доменных зон, перед регистрарами встала задача поддержки большого количества новых доменов. Так как ожидалось, что регистрировать доменные имена в новых зонах будет большое количество, то были предприняты некоторые меры. Одной из таких мер была возможность пререгистрировать домены до официального запуска доменной зоны в общий доступ.

Реселлерам нужно было предоставить интерфейс для массовой регистрации доменов в более чем 800 доменных зонах.

**Реализация**

Весь процесс регистрации был разбит на два шага. Страница первого шага состоит из двух частей:

* поля ввода доменов и доменных имен;
* блока выбора доменных зон.

В поле ввода нужно указывать список доменных имен для регистрации. Так же есть возможность указывать доменные имена с доменными зонами. Такие варианты будут автоматически выбраны из блока доменных зон. Пример ввода:

```
example
mysite
example.london
```

Чтобы помочь ориентироваться реселлеру в большом количестве доменных зон, все зоны были разбиты на категории. Каждая категория, кроме первой, схлопнута. В первой категории все популярные домены выбраны по умолчанию.

![2796_original](/assets/img/2796_original.png){:class="img-fluid"}

Количество выбранных вариантов отображается в заголовке. Есть возможность выбрать все варианты в блоке. Для IDN вариантов вскобках даются пояснения.

![2935_original](/assets/img/2935_original.png){:class="img-fluid"}

UPD: Запуск пререгистрации позволил нам вырваться на 13 место по пререгистрациям среди других реселлеров. Но позже мы нашли один недостаток у текущего интерфейса: отсутствовал поиск по существующим доменам.

*Перенёс сюда из [жж](https://gurovpavel.livejournal.com/1750.html)*
