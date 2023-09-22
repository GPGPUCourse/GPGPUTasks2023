В этом репозитории предложены задания для курса по вычислениям на видеокартах 2023.

[Остальные задания](https://github.com/GPGPUCourse/GPGPUTasks2023/).

Сдача задания текстом через пулл-реквест в эту ветку, созданием файла ANSWER.md.

# Задание 2. Теоретическое задание: параллелизуемость/code divergence/memory coalesced access

Ниже три небольших задачи. Рекомендуется хотя бы начать делать каждое задание к лекции 15 сентября, чтобы задать вопросы если они будут и обсудить задания. **Дедлайн**: 23:59 24 сентября.

**1)** Пусть на вход дан сигнал x[n], а на выход нужно дать два сигнала y1[n] и y2[n]:

```
 y1[n] = x[n - 1] + x[n] + x[n + 1]
 y2[n] = y2[n - 2] + y2[n - 1] + x[n]
```

Какой из двух сигналов будет проще и быстрее реализовать в модели массового параллелизма на GPU и почему?

Решение: первый вызов будет исполняться быстрее, потому что он только читает данные, которые другие потоки не меняют.
Во втором случае, для y2[n], нам нужно дождаться выполнения y2[n - 2] и y2[n - 1] и получается, что мы сильно замедляемся.


**2)** Предположим что размер warp/wavefront равен 32 и рабочая группа делится
 на warp/wavefront-ы таким образом что внутри warp/wavefront
 номер WorkItem по оси x меняется чаще всего, затем по оси y и затем по оси z.

Напоминание: инструкция исполняется (пусть и отмаскированно) в каждом потоке warp/wavefront если хотя бы один поток выполняет эту инструкцию неотмаскированно. Если не все потоки выполняют эту инструкцию неотмаскированно - происходит т.н. code divergence.

Пусть размер рабочей группы (32, 32, 1)

```
int idx = get_local_id(1) + get_local_size(1) * get_local_id(0);
if (idx % 32 < 16)
    foo();
else
    bar();
```

Произойдет ли code divergence? Почему?

Code divergence не произвойдет, потому что так как у нас workItem по оси х меняется чаще и get_local_size(0) равен размеру warp,
то у нас при каждом новом вызове kernel, будет get_local_id(0) одинаковый для всех потоков. И так как get_local_size(1) == 32
То при каждом вызове будет выполняться одна конкретная ветка условия для всех потоков.
Поэтому code divergence не произойдет.

**3)** Как и в прошлом задании предположим что размер warp/wavefront равен 32 и рабочая группа делится
 на warp/wavefront-ы таким образом что внутри warp/wavefront
 номер WorkItem по оси x меняется чаще всего, затем по оси y и затем по оси z.

Пусть размер рабочей группы (32, 32, 1).
Пусть data - указатель на массив float-данных в глобальной видеопамяти идеально выравненный (выравнен по 128 байтам, т.е. data % 128 == 0). И пусть размер кеш линии - 128 байт.

(a)
```
data[get_local_id(0) + get_local_size(0) * get_local_id(1)] = 1.0f;
```

Будет ли данное обращение к памяти coalesced? Сколько кеш линий записей произойдет в одной рабочей группе?

Мы обращаемся к элементам с соседними индексами.
За одно обращение мы подгрузим сразу 32 элемента в кеш, поэтому обращение к памяти coalesced.
Произойдет 32 записи кеш линий.

(b)
```
data[get_local_id(1) + get_local_size(1) * get_local_id(0)] = 1.0f;
```

Будет ли данное обращение к памяти coalesced? Сколько кеш линий записей произойдет в одной рабочей группе?

Здесь мы обращаемся к элементам через 31 элемент, длины кеша не хватит, чтобы туда попало несколько вызываемых элементов.
Обращений будет 32*32=1024 и столько же будет записей в кеш.

(c)
```
data[1 + get_local_id(0) + get_local_size(0) * get_local_id(1)] = 1.0f;
```

Будет ли данное обращение к памяти coalesced? Сколько кеш линий записей произойдет в одной рабочей группе?

При обращении нам не хватит длины одной кеш линии, чтобы захватить все элементы, поэтому мы получаем в 2 раза больше подгрузок, чем в первом случае.
Итого 64 подгрузки.