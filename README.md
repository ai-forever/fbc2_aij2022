# FusionBrain Challenge 2.0

## Общее описание задачи

В рамках данной задачи предлагается построить единую multitask-модель, которая бы успешно решала подзадачи в двух модальностях (визуальной и текстовой), принимая на вход описания подзадач, выраженные на естественном русском языке, например: "сгенерируй изображение", "опиши изображение", "ответь на вопрос" и т.д. В состав входит 12 подзадач, из которых 6 известны участникам с момента начала Конкурса (открытые подзадачи), а 6 неизвестны (скрытые подзадачи) и представляют собой частные случаи открытых задач (имеют некоторые отличительные особенности в постановке). Основная задача участников заключается в построении и обучении единой мультимодальной мультизадачной архитектуры, которая позволила бы получить максимальные значения метрик для каждой отдельной подзадачи и, как следствие, достичь максимального значения интегральной метрики на 12 подзадачах. Далее в правилах проведения конкурса будет приведено подробное описание открытых задач и соответствующих метрик качества.

## Общий формат решения

### Содержимое контейнера

В проверяющую систему необходимо отправить код алгоритма, запакованный в ZIP-архив. Решения запускаются в изолированном окружении при помощи Docker. Время и ресурсы во время тестирования ограничены. Участнику нет необходимости разбираться с технологией Docker.

В корне архива обязательно должен быть файл `metadata.json` следующего содержания:
```
{
    "image": "cr.msk.sbercloud.ru/aijcontest2022_official/fusion:0.0.1",
    "entrypoint": "python3 run.py"
}
```

Здесь `image` – поле с названием docker-образа, в котором будет запускаться решение, `entrypoint` – команда, при помощи которой запускается скрипт инференса. Для каждой команды создается уникальная директория, куда распаковывается решение.

Для запуска решений можно использовать существующее окружение:

* ```cr.msk.sbercloud.ru/aijcontest2022_official/fusion:0.0.1``` - [Dockerfile и requirements](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/docker_FB2.zip) для данного образа.

При необходимости вы можете подготовить свой образ, добавив в него необходимое ПО и библиотеки (см. [инструкцию по созданию docker-образов](https://dsworks.s3pd01.sbercloud.ru/static/champ/aij22/%D0%98%D0%BD%D1%81%D1%82%D1%80%D1%83%D0%BA%D1%86%D0%B8%D1%8F%20%D0%BF%D0%BE%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B5%20%D1%81%D0%BE%20sbercloud.pdf)); для использования его необходимо будет опубликовать на ```sbercloud```. Кастомные образы должны быть наследованы от базовых образов ```sbercloud``` (см. [базовые образы](https://docs.sbercloud.ru/aicloud/mlspace/concepts/environments__basic-images-for-training.html)). При создании кастомного образа необходимо присвоить ему индивидуальное название и тэг (например, ```my_custom_fusionchallenge:0.0.5```).

### Baseline решение

Baseline решение основано на модели [RUDOLPH](https://github.com/ai-forever/ru-dolph). RUDOLPH – это мультизадачная модель-декодер, способная решать ряд задач внутри двух модальностей: текст и изображение, что соответсвует условиям соревнования FBC2. 

Существует три версии модели RUDOLPH: 350M, 1.3B, 2.7B. В качестве базовой модели мы использовали модель RUDOLPH 2.7B, дообученную для решения шести открытых задач FBC2 ([RUDOLPH-2.7B-FBC2](https://huggingface.co/sberbank-ai/RUDOLPH-2.7B-FBC2)). Описание наборов данных, используемых для дообучения приведены ниже.

Пример архива с решением размещен по ссылке [_sample\_submission.zip_](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/submission_350.zip).

### Структура данных

Данные представляют собой словарь: значение каждого ключа – список, содержимое которого зависит от данных, необходимых для постановки конкретной подзадачи: поле `type` указывает тип данных (`text` или `image`), `content` – непосредственное содержимое (например, вопрос или варианты ответа на естественном языке, путь к файлу с изображением). В общем виде формат выглядит следующим образом:
```
{"0": [{"type": ..., "content": ...}, ...], "1": [...], ...} 
```

### Формат решения

Данные для совершения предсказания объединяют все подзадачи и включают в себя:

* Файл _input.json_. Это словарь формата `{"0": [{"type": "text", "content": "Ответь на вопрос по изображению. На чём едет человек справа на картинке?"}, {"type": "image", "content": "images/00000.jpg"}, {"type": "text", "content": "варианты: a) на велосипеде; b) на лошади; c) на машине; d) на самокате"}], "1": [{"type": "text", "content": "Дай описание изображения."}, {"type": "image", "content": "images/00005.jpg"}], "2": [{"type": "text", "content": "Сгенерируй изображение по текстовому запросу."}, {"type": "text", "content": "Пара человек сидит в тени. Велосипед, оставленный на велопарковке, стоит на солнце."}], ...}`. Ключами являются id примеров, значениями – список словарей, описанный в пункте **Структура данных**.
* Папка _images_, в которой хранятся изображения, по которым нужно сделать предсказания. Внутри лежат файлы формата 00000.jpg, 00001.jpg, ....

Входные данные _input.json_ и _images_ находятся в директории _input_ (директория расположена на один уровень выше той, в которой запускается решение, то есть в _../input_, см. пример в файле config/fusion_inference.yaml [_sample\_submission.zip_](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/sample_submission.zip)).

**Обратите внимание**: в некоторых подзадачах непосредственное описание задания ("ответь на вопрос", "выбери правильный вариант" и т.д.) может выноситься в отдельный словарь `{"type": "text", "content": "Ответь на вопрос по изображению."}`, объединяться с другим текстовым полем `{"type": "text", "content": "Ответь на вопрос по изображению. На чём едет человек справа на картинке?"}`, или вовсе отсутствовать `{"type": "text", "content": "На чём едет человек справа на картинке?"}`.

Модель участника должна сделать предсказания для всех входных примеров из файла _input.json_ и сгенерировать выходные данные, которые включают в себя:

* Файл _predictions.json_. Это словарь формата `{"0": [{"type": "text", "content": "a)"}, {"type": "image", "content": "images/image_0.jpg"}], "1": [{"type": "text", "content": "две большие птицы летают над озером."}], "2": [{"type": "image", "content": "images/image_2.jpg"}], ...}`. Ключами являются id примеров из _input.json_, значениями – список словарей, где поле `type` указывает тип данных, которые сгенерировала модель (`text` или `image`), `content` – непосредственное содержимое (например, текстовый ответ, путь к файлу с сгенерированным изображением).
* Папка _images_. Это набор изображений, сгенерированных моделью для некоторых (или всех) примеров из входного файла _input.json_. Внутри лежат файлы формата image_0.jpg, image_1.jpg ....

**Обратите внимание:** выходные данные, предсказанные моделью (_prediction.json_ и папка с сгенерированными изображениями _images_) должны сохраняться в папку _output_ в директории запуска решения (т.е. в _./output_, см. пример в файле config/fusion_inference.yaml [_sample\_submission.zip_](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/sample_submission.zip)).
    
Файл правильных ответов _ground\_truth.json_, который будет использоваться для оценки качества модели во время запуска в контейнере — это словарь, который имеет следующий формат: `{"0": [{"type": "text", "content": ["a", "на велосипеде", "a на велосипеде"]}], "1": [{"type": "text", "content": ["две птицы летают над водой", "птицы летают", "два сокола летят над озером"]}], "2": [{"type": "image", "content": "gt_images/image_2.jpg"}], ...}`. 

После генерации предсказаний обрабатывающая система разбивает файл с предсказаниями _predictions.json_ и файл с правильными ответами _ground\_truth.json_ на несколько файлов для каждой из подзадач, сравнивает полученные файлы и выводит значения метрик для каждой из подзадач и значение интегральной метрики.

## Ограничения

В течение одних суток Участник или Команда Участников может загрузить для оценки не более 3 (трёх) решений. Учитываются только валидные попытки, получившие численную оценку. Если при расчете хотя бы одной метрики возникло исключение, решение считается невалидным, и счетчик попыток не уменьшается.

Контейнер с решением запускается в следующих условиях:
* 100 Гб оперативной памяти
* 3 vCPU
* 1 GPU Tesla A100 (80 Гб)
* время на выполнение решения: 5 часов
* решение не имеет доступа к ресурсам интернета
* максимальный размер упакованного и распакованного архива с решением: 15 Гб
* максимальный размер используемого Docker-образа: 25 Гб.

Мы предоставляем участникам возможность получить доступ к вычислительным ресурсам Кристофари для обучения модели. Количество ресурсов ограничено. Для получения доступа необходимо отправить заявку на адрес Christofari_AIJContest_2022@sberbank.ru с описанием того, как именно планируется использовать вычислительные ресурсы.

# Данные для обучения

Обучающие датасеты по всем открытым подзадачам могут быть загружены по [ссылке](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/fb_train_2022.tar.gz).

Ниже приведено подробное описание обучающей выборки для каждой открытой подзадачи.

# Подзадача 1 - Text QA (TextQA)

## Описание 

Задание на понимание прочитанного текста; для успешного решения модель должна уметь находить причинно-следственные связи в тексте. Каждый семпл состоит из текста, вопросов к нему и вариантов ответа (опционально). В корпусе присутствуют вопросы нескольких типов, для успешного ответа на вопрос модель должна уметь устанавливать причинно-следственные связи, разрешать кореференции, а также определять правильную последовательность действий, учитывая временную информацию. Если к входному тексту дан только вопрос, то модель должна сгенерировать подходящий ответ; если же к входному тексту, помимо вопроса, есть несколько вариантов ответа, то модель должна выбрать верный (может быть 4 варианта ответа и только один из них правильный).

## Данные

**Train** 

В качестве обучающего набора данных предлагается использовать датасет Sber Question Answering Dataset ([SberQuAD](https://huggingface.co/datasets/sberquad)). Набор данных SberQuAD был предложен для проверки способности модели отвечать на вопросы по входному тексту. Датасет содержит абзац текста из Википедии, вопрос, заданный к этому абзацу и часть текста из входного абзаца, которая является ответом на заданный вопрос. Датасет содержит 45328 обучающих, 5036 валидационных и 23936 тестовых примеров.

Дополнительно можно использовать датасет Russian Multi-Sentence Reading Comprehension ([MuSeRC](https://russiansuperglue.com/tasks/task_info/MuSeRC)). Набор данных MuSeRC был предложен для проверки способности модели отвечать на вопросы по входному тексту. Датасет содержит около 6000 вопросов, заданных к более чем 800 текстовым абзацам. Стоит отметить, что исходная разметка датасета MuSeRC сделана для задачи бинарной классификации, что не соответствует предлагаемой подзадаче **TextQA**, для ее использования необходимо привести данные из датасета к требуемому формату, чтобы была возможность использовать их в обучении модели.

Для удобства мы подготовили обучающий датасет для задачи TextQA (размещен в соответсвующей папке [архива](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/fb_train_2022.tar.gz)) в том же формате, в котором хранятся Test public и Test private. Обучающие данные содержат примеры из train/val частей датасета SberQuAD.

### Формат тестовых данных для TextQA

**Test public** 

Публичный лидерборд формируется по результатам проверки предсказаний моделей на наборе данных, соответствующему задаче TextQA, собранном и размеченным организаторами самостоятельно.

В тестовых наборах для подзадачи TextQA встречаются вопросы в **генеративном** формате и формате с **выбором правильного варианта ответа** из предложенных.

Формат входных семплов для задачи TextQA в _input.json_ выглядит следующим образом:

```
{
"0": [{"type": "text", "content": "Мужская сборная команда Норвегии по биатлону в рамках этапа Кубка мира в немецком Оберхофе выиграла эстафетную гонку. Вторыми стали французы, а бронзу получила немецкая команда. Российские биатлонисты не смогли побороться даже за четвертое место, отстав от норвежцев более чем на две минуты. Это худший результат сборной России в текущем сезоне. Четвёртыми в Оберхофе стали австрийцы. Напомним, что днем ранее российские биатлонистки выиграли свою эстафету. В составе сборной России выступали Анна Богалий-Титовец, Анна Булыгина, Ольга Медведцева и Светлана Слепцова. Они опередили своих основных соперниц - немок - всего на 0,3 секунды."}, {"type": "text", "content": "На сколько секунд сборная России опередила ближайших соперниц?"}, {"type": "text", "content": "a) на 0,3 секунды; b) на секунду; c) на 0,5 секунд"}],
"1": [{"type": "text", "content": "Определи правильный ответ по тексту."}, {"type": "text", "content": "Улица Авиаконструктора Сухого (название с 2004 года) — улица в Северном административном округе города Москвы на территории Хорошёвского района. Пролегает от Проектируемого проезда № 6161 до пересечения с юго-восточной частью взлётно-посадочной полосы Ходынского Поля. Название утверждено 30 марта 2004 года в честь знаменитого советского авиаконструктора Павла Осиповича Сухого (1895—1975)."}, {"type": "text", "content": "Какая улица пролегает от Проектируемого проезда № 6161 до пересечения с юго-восточной частью взлётно-посадочной полосы Ходынского Поля?"}],
}
```

Формат истинных ответов для TextQA в _ground\_truth.json_:

```
{
"0": {"type": "text", "content": ["a", "на 0,3 секунды", "a на 0,3 секунды"]},
"1": {"type": "text", "content": ["Авиаконструктора Сухого"]},
…
}
```

**Test private** 

Приватный тестовый датасет скрыт от участников. Его формат аналогичен публичной тестовой выборке.

### Формат решения

Верным ответом для подзадачи TextQA считается фрагмент входного текста, который отвечает на заданный вопрос (для генеративного формата), и символ-маркер верного ответа, либо непосредственный текст правильного ответа, либо символ-маркер и последующий текст ответа (для формата с выбором правильного варианта). Предсказания для задачи типа TextQA имеют следующий формат `{"0": [{"type": "text", "content": "_ответ на вопрос 0_"}, ...], "1": [{"type": "text", "content": "_ответ на вопрос 1_"}, ...]}`

# Подзадача 2 - Mathematical QA (MathQA)

## Описание 

Данная подзадача проверяет способность модели выполнять простейшие арифметические действия, необходимые для решения линейных уравнений или систем линейных уравнений, а также производить операции сравнения. Задание состоит из математического примера, который описан на естественном языке; используемые арифметические операции: сложение, вычитание, умножение и деление. Неизвестные переменные могут выражаться любыми латинскими буквами. Если к входной математической задаче дан только вопрос, то модель должна сгенерировать подходящий ответ; если же к входной математической задаче, помимо вопроса, есть несколько вариантов ответа, то модель должна выбрать верный (может быть несколько вариантов ответа и только один из них правильный).

## Данные

**Train** 

Для обучения модели можно собрать набор данных, содержащий математические примеры, сгенерированные с помощью библиотеки [Mathematics Dataset](https://github.com/deepmind/mathematics_dataset/tree/master/mathematics_dataset) и переведенные на русский язык (в некоторых примерах будет полезно аугментировать данные так, чтобы символы арифметических выражений ('+', '-', '*', '/', '=') были записаны словами 'плюс', 'минус' и т.д.). При генерации обучающих примеров особое внимание стоит обратить на модули для линейных уравнений и систем линейных уравнений с 1-2 неизвестными переменными (algebra ('linear_1d', 'linear_2d')).

Нами был подготовлен обучающий датасет для задачи MathQA (размещен в соответсвующей папке [архива](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/fb_train_2022.tar.gz)) в том же формате, в котором хранятся Test public и Test private. Обучающие данные были сгенерированы с использованием библиотеки Mathematics Dataset, аугментировани и переведен с помощью автоматического переводчика.

### Формат тестовых данных для MathQA

**Test public** 

В тестовом наборе могут присутствовать задания на решение линейных уравнений и систем линейных уравнений, а также задачи на сравнение двух величин. Сам текст математического выражения записан на естественном языке, в связи с чем, помимо математических символов, в тексте могут присутствовать выражения вида: "если к _x_ прибавить _y_, то ...", "_a_ равно 7 минус _b_" и т.д.

Задача MathQA предполагает формат выбора правильного варианта ответа из предложенных, таким образом, во входных данных всегда присутствует список вариантов ответа с соответствующими им маркерами (в качестве маркера выступают буквы латинского алфавита с закрывающей скобкой: a), b), c) ...). Правильным считается либо символ-маркер верного ответа, либо непосредственный текст правильного ответа, либо символ-маркер и последующий текст ответа.

Каждый входной семпл для задачи MathQA содержит следующие данные: текст задания (опционально), математический пример, 4 варианта ответа, один из которых является правильным.

Формат входных семплов для задачи MathQA в _input.json_ выглядит следующим образом:

```
{
"0": [{"type": "text", "content": "Решите линейное уравнение."}, {"type": "text", "content": "6 умножить на y – 54 это -72, для y."}, {"type": "text", "content": "a) -4; b) -20; c) -5; d) -3"}],
...
}
```

Формат истинных ответов для MathQA в _ground\_truth.json_ выглядит следующим образом:

```
{
"0": {"type": "text", "content": ["d", "-3", "d -3"]},
…
}
```

**Test private** 

Приватный тестовый датасет скрыт от участников. Его формат аналогичен публичной тестовой выборке.

## Формат решения

В качестве правильного ответа для примеров, относящихся к задаче MathQA, модель должна сгенерировать либо символ-маркер правильного варианта, либо текст правильного варианта, либо их комбинацию. По аналогии с TextQA с выбором правильного ответа, любой формат вывода из перечисленных выше будет считаться правильным ответом. 

В общем случае, словарь ответов для вопросов типа MathQA имеет следующий формат `{"0": [{"type": "text", "content": "_ответ на вопрос 0_"}, ...]}`.

# Подзадача 3 - Image Generation (ImageGeneration)

## Описание 

Эта подзадача подразумевает генерацию изображений на основе текстовых описаний на русском языке. Ответом на подзадачу является изображение, чье содержание соответствует входному текстовому описанию. 

## Данные

**Train** 

Для обучения предлагается использовать набор данных [COCO](https://cocodataset.org), содержащий пары "текст-картинка"; все текстовые описания могут быть переведены на русский язык с помощью переводчика.

Для задачи ImageGeneration мы также подготовили обучающий датасет (размещен в соответсвующей папке [архива](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/fb_train_2022.tar.gz)) в том же формате, в котором хранятся Test public и Test private. Обучающие данные были переведены с помощью автоматического переводчика.

### Формат тестовых данных для ImageGeneration

**Test public** 

Тестовый набор данных содержит текстовую формулировку решаемой подзадачи, а также запрос с описанием картинки, по которому нужно сгенерировать изображение. В ground_truth указаны пути до реальных картинок. 

Формат входных семплов для задачи ImageGeneration в _input.json_ выглядит следующим образом:

```
{
"0": [{"type": "text", "content": " Сгенерируй изображение по тексту."}, {"type": "text", "content": "Торт со светло-бежевой глазурью, украшенный фигуркой медведя, держащий связку воздушных шариков."}],
...
}
```

Формат истинных ответов для ImageGeneration в _ground\_truth.json_ выглядит следующим образом:

```
{
"0": [{"type": "image", "content": "images/00000.jpg"}],
...
}
```

**Test private** 

Приватный тестовый датасет скрыт от участников. Его формат аналогичен публичной тестовой выборке.

## Формат решения

Изображения, сгенерированные моделью должны быть расположены по пути _images_, путь к ним должен быть указан в соответсвующем поле выходного словаря `{"0": [{"type": "image", "content": "images/_имя изображения 0_"}, ...}`.

# Подзадача 4 - Image Captioning (ImageCaptioning)

## Описание 

Данная подзадача подразумевает генерацию текстовых описаний на русском языке к изображениям. Ответом на подзадачу является текстовая строка, содержащая текстовое описание входного изображения.

## Данные

**Train** 

Для обучения предлагается использовать набор данных [COCO](https://cocodataset.org), содержащий пары "текст-картинка"; все текстовые описания могут быть переведены на русский язык с помощью переводчика. 

Для задачи ImageCaptioning нами подготовлен обучающий датасет (размещен в соответсвующей папке [архива](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/fb_train_2022.tar.gz)) в том же формате, в котором хранятся Test public и Test private. Обучающие данные были переведены с помощью автоматического переводчика.

### Формат тестовых данных для ImageCaptioning

**Test public** 

Тестовый набор данных содержит текстовую формулировку решаемой подзадачи, а также изображение, для которого нужно сгенерировать описание.

Формат входных семплов для задачи ImageCaptioning в _input.json_ выглядит следующим образом:

```
{
"0": [{"type": "text", "content": "Что представлено на картинке?"}, {"type": "image", "content": "images/00000.jpg"}],
...
}
```

Пример изображения:

`images/00000.jpg`: <img src="https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/images_web/image-caption.png" width=250>

Формат истинных ответов для ImageCaptioning в _ground\_truth.json_ выглядит следующим образом:

```
{
"0": [{"type": "text", "content": ["Торт со светло-бежевой глазурью, украшенный фигуркой медведя, держащий связку воздушных шариков.", "Детский торт, покрытый светлой глазурью с украшением в виде медвежонка."]}],
...
}
```

**Test private** 

Приватный тестовый датасет скрыт от участников. Его формат аналогичен публичной тестовой выборке.

## Формат решения

В качестве валидного решения модель должна сгенерировать описание входного изображения `{"0": [{"type": "text", "content": "_описание изображения 0_"}, ...], ...}`.

По аналогии с подзадачами TextQA и MathQA, файл _ground_truth.json_ содержит несколько возможных описаний для входного изображения; берется максимальное значение метрики, полученное для всех вариантов.

# Подзадача 5 - Visual QA (VisualQA)

## Описание 

Подзадача предполагает, что обученная модель способна формировать ответ на вопрос по изображению. В этой подзадаче на вход модели подаётся пара вида "текстовый вопрос – картинка", а выходом является соответствующий текстовый ответ. Если к входному изображению дан только вопрос, то модель должна сгенерировать подходящий ответ; если же к входному тексту, помимо изображения, есть несколько вариантов ответа, то модель должна выбрать верный (может быть 4 варианта ответа и только один из них правильный).

**Train** 

Для обучения предлагается использовать набор данных [Visual Genome](https://visualgenome.org/api/v0/api_home.html), в котором содержатся пары "вопрос-ответ" по картинкам. Все вопросы и ответы могут быть переведены с помощью переводчика. 

Для задачи VisualQA был подготовлен обучающий датасет (размещен в соответсвующей папке [архива](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/fb_train_2022.tar.gz)) в том же формате, в котором хранятся Test public и Test private. Обучающие данные были переведены с помощью автоматического переводчика.

### Формат тестовых данных для VisualQA

**Test public** 

Тестовый набор данных содержит текстовую формулировку решаемой подзадачи (опционально), вопрос, а также изображение, по которому задается вопрос.

Формат входных семплов для задачи VisualQA в _input.json_ выглядит следующим образом:

```
{
"0": [{"type": "text", "content": "Ответь на вопрос по картинке."}, {"type": "text", "content": "Какого цвета плакат на ограждении?"}, {"type": "image", "content": "images/00000.jpg"}], 
...
}
```

`images/00000.jpg`: <img src="https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/images_web/image-2.png" width="250">


Формат истинных ответов для VisualQA в _ground_truth.json_ выглядит следующим образом:

```
{
"0": [{"type": "text", "content": ["желтого", "ярко-желтого"]}], 
...
}
```

**Test private** 

Приватный тестовый датасет скрыт от участников. Его формат аналогичен публичной тестовой выборке.

## Формат решения

В качестве валидного решения модель должна сгенерировать ответ на вопрос по изображению `{"0": [{"type": "text", "content": "_ответ на вопрос 0_"}, ...], ...}`.

# Подзадача 6 - Text Recognition in the Wild (TRitW)

## Описание 

Задание на распознавание текста в городской или иной подобной местности (вывески, дорожные знаки, рекламные объявления и т.п.). Данные представляют собой фотографии объектов с изображенным на них текстом. Ответом на подзадачу является текстовая строка, содержащая все распознанные на входном изображении кириллические и латинские буквы, а также распространенные специальные символы.

**Train** 

Для обучения предлагается новый датасет – [START](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/titw_dataset.zip) (**S**yn**T**hesized and **A**nnotated dataset for **T**ext **R**ecognition), состоящий из двух частей с синтетичекскими и реальными данными. Синтетический набор данных содержит 140 000 изображений с наложенными на них надписями на русском языке. Изображения взяты из открытого датасета [COCO](https://cocodataset.org). На изображения добавлен текст (текст расположен в разных частях изображения, повернут под разными углами, имеет различные цвета и прозрачность и т.д.). Набор реальных данных содержит 37 385 фотографий городской среды, на которых присутствуют надписи на русском языке. Модель обучается распознавать текст, присутствующий на изображении.

### Формат тестовых данных для TRitW

**Test public** 

Тестовый набор данных содержит текстовую формулировку решаемой подзадачи, а также изображение, на котором нужно распознать текст.

Формат входных семплов для задачи TRitW в _input.json_ выглядит следующим образом:

```
{
"0": [{"type": "text", "content": "Распознай текст на картинке."}, {"type": "image", "content": "images/00000.jpg"}],
...
}
```

`images/00000.jpg`: <img src="https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/images_web/image-4.jpeg" width="250">

Формат истинных ответов для TRitW в _ground_truth.json_ выглядит следующим образом:

```
{
"0": [{"type": "text", "content": ["хор"]}],
...
}
```

**Test private** 

Приватный тестовый датасет скрыт от участников. Его формат аналогичен публичной тестовой выборке.

## Формат решения

В качестве валидного решения модель должна сгенерировать текст, распознанный на изображении `{"0": [{"type": "text", "content": "_текст на изображении 0_"}, ...], ...}`.

# Метрики качества

С метрикой качества для каждой из подзадач можно ознакомиться по [ссылке](https://github.com/ai-forever/fbc2_aij2022/blob/main/Metrics.md).

### Общие правила для расчета метрик

* При подсчете метрик для задач, для которых ожидается текстовый выход (например, для открытых задач к данному типу относятся TextQA, MathQA, VisualQA, ImageCaptioning, TRitW), ответ считывается из **первого** словаря, где ключу ```"type"``` соответствует значение ```"text"```.
* Если для подзадач с текстовым выходом помимо текстового ответа модель сгенерировала изображение, это не считается ошибкой, а словарь для изображения (```{"type": "image", "content": ...```) игнорируется при подсчете метрик.
* В большинстве случаев, для подзадач с текстовым выходом, файл правильных ответов (_ground\_truth.json_) для Test public и Test private содержит несколько правильных вариантов ответа для каждого входного примера (например, разные варианты правильного ответа, синонимы к правильному ответу и т.д.). При подсчете метрик для задач с текстовым выходом, предсказание, сгенерированное моделью, сравнивается с каждым вариантом ответа из _ground_truth.json_ и сохраняется максимальное рассчитанное значение метрики (подробнее смотрите в описании подзадач).
* Перед расчетом метрик производится постпроцессинг сгенерированных предсказаний: приведение к нижнему регистру, удаление знаков пунктуации, удаление лишних пробелов, приведение слов к нормальной форме.
* При подсчете метрик для подзадач, где выходом является изображение (например, Image Generation), ответ считывается из первого словаря, где ключу ```"type"``` соответствует значение ```"image"```. Если помимо изображения модель сгенерировала какой-либо текст, это не считается ошибкой, а словарь для текстового выхода (```{"type": "text", "content": ...```) игнорируется при подсчете метрик.

### Ground truth

Файл с правильными ответами (ground truth) для Test public и Test private содержит несколько правильных вариантов ответа для каждого входного примера. При подсчете метрик для задач с текстовым выходом, предсказание, сгенерированное моделью, сравнивается с каждым вариантом ответа из ground truth и сохраняется максимальное рассчитанное значение метрики.

Пример:

predictions.json

```
{"0": [{"type": "text", "content": "a) 5"}]}
```

ground_truth.json

```
{"0": [{"type": "text", "content": ["a", "a 5", "5"]}]}
```

В данном случае максимальное значение метрики будет получено при сопоставлении предсказанного ответа _"a) 5"_ с истинным вариантом _"a 5"_. 


# Test sample set

По шести открытым задачам был подготовлен _test sample set_, который содержит по 20 примеров для каждой задачи. Формат файлов sample_input.json и sample_output.json полностью согласован с форматом соответствующих файлов из _Public test_ и _Private test_. Вы можете использовать данные файлы для проверки корректности вашего решения перед отправкой.

В **Test sample set** также присутствует файл _sample_ids.json_, в котором содержится словарь отображения id примера в тип задачи, к которому он относится (```{"sample_id": task_id, ...} ```). 

_task_id_ принимает значения от 0 до 5, ниже приведено соответствие между _task_id_ и названиями задач:

* 0: TextQA
* 1: MathQA
* 2: ImageGeneration
* 3: ImageCaptioning
* 4: VisualQA
* 5: TRitW

Обратите внимание что в _public_ и _private test_ нет никакого указания на тип задачи.

**Test sample set** размещен на платформе по [ссылке](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/sample_test.zip).

# Интегральная метрика

Итоговая оценка мультизадачной модели по задаче FusionBrain Challenge 2.0 складывается из значений метрик качества по отдельным подзадачам. 

$$S = \frac{1}{15}\sum_{j \in \\ [open_t]}{S_{j}} + \frac{1}{10}\sum_{k \in [hidden_t]}{S_{k}},$$

где $[open_t] = \\{ TextQA, MathQA, ImageGeneration, ImageCaptioning, VisualQA, TRitW \\}$ – это множество открытых подзадач, $[hidden_t] = \\{ Hidden1,\ Hidden2,\ Hidden3,\ Hidden4,\ Hidden5,\ Hidden6 \\}$ – это множество скрытых подзадач.

Если конкретный участник получил самый высокий балл по каждой отдельной подзадаче, то итоговое значение метрики $S$ будет максимальным среди всех участников.

Для расчета результатов на Test private участник определяет 3 решения по своему выбору, для каждого из которых производится расчет метрики $S$ на приватных данных. Лучшее значение метрики $S$ заносится в лидерборд на данных Test private. Если Участник не выбрал 3 решения для финальной проверки, то автоматически выбираются топ-3 решения по значению интегральной метрики на Test public.

Победителями соревнования будут считаться решения, занявшие 1-3 места в лидерборде на данных Test private путём вычисления значения интегральной метрики $S$. При этом, для каждой из открытых подзадач определена нижняя граница соответствующих метрик (границы определены исходя из условия 75% от качества baseline решения):

* TextQA: 0,2
* MathQA: 0,25
* ImageGeneration: 0,21
* ImageCaptioning: 0,15
* VisualQA: 0,24
* TRitW: 0,13

Для того чтобы претендовать на призовое место, значения метрик, полученные Участником для каждой из открытых подзадач, не должны быть меньше заданных нижних границ.

# Призовой фонд

## Основные номинации за создание multitask-модели**

**Первое место:** 1 000 000 рублей  
**Второе место:** 700 000 рублей   
**Третье место:** 400 000 рублей

## Специальная номинация
**Самое экологичное решение:** 100 000 рублей

## Специальная номинация "Самое экологичное решение"

### Описание номинации 

В рамках решаемой задачи вводится специальная номинация «Самое экологичное решение». Ее основная цель – мотивировать участников соревнования к поиску изящных и нестандартных решений, оптимальных с вычислительной точки зрения. Метрикой данной номинации является объем косвенного выделения углекислого газа (СО2) в процессе обучения модели, представленной участником в качестве решения. Метрика должна быть рассчитана с помощью библиотеки [eco2AI](https://github.com/sb-ai-lab/Eco2AI).

Для участия в номинации необходимо выполнить следующие условия:
+ Решение попало в топ по основной метрике задачи (подробнее далее)
+ Участник передал данные (прикрепил к архиву при загрузке решения на платформу) по выделенному СО2 во время обучения модели (файл </your_filename/>.scv посчитанный с помощью eco2aAI)
+ Участник прислал на почту AIJContest_2022_org@sberbank.ru презентацию в свободной форме с описанием структуры своей модели и процесса ее обучения, включая часть про расчёт СО2 и подбор вычислительных ресурсов. Цель презентации - показать каким образом были получены результаты участника, убедиться в честности их получения. Срок отправки до 11 ноября 2022.

Отдельным плюсом для участника при проверке презентации будет наличие двух графиков: графика прироста СО2 на каждой эпохе обучения и графика зависимости лосса модели от выделенного СО2. Пример графиков можно найти в статье "eco2AI: carbon emissions tracking of machine learning models as the first step towards sustainable AI". Для удобства построения аналогичных графиков в библиотеке реализованы следующие методы: .start_training .new_epoch  .stop_training. Ссылка на [колаб ноутбук с гайдом](https://github.com/sb-ai-lab/Eco2AI#3).

Ниже представлена техническая информация, необходимая участникам для использования eco2AI в рамках конкурса:
+ в параметре «project_name» метода Tracker указать название решаемой задачи
+ в параметре «experiment_description» объекта класса Tracker указать краткое описание модели (количество параметров и тд.)
+ метод tracker.start() должен использоваться в самом начале исполняемого кода, сразу после импорта библиотек. метод tracker.stop() должен использоваться в самом конце исполняемого кода
+ если осуществляется финальный этап обучения, дополнительно указать в поле «experiment_description» ключевую фразу “TargetValue”

Поскольку участники могут запускать обучение своих моделей многократно, в качестве ответа в номинации из файла emission.csv будет взято последнее значение "CO_2  emissions", у которого в графе "experiment_description" будет присутствовать ключевая фраза “TargetValue”. 
Организаторы оставляют за собой право отказать в участии в номинации «Самое экологичное решение» в случае несоблюдения вышеизложенных инструкций или обнаружения фальсификации данных. В случае если несколько решений будут иметь одинаковую метрику СО2, победитель будет определяться решением комиссии на основании предоставленных материалов участников.

### Описание метрики номинации "Самое экологичное решение"
В качестве метрики номинации используется значение "CO_2 emissions" – косвенное выделение СО2 в процессе обучения модели. Расчёт осуществляется по формуле ниже:

$$ CO_2 =(E_{CPU}+E_{GPU})*CO_{2coef}$$

где $E_{CPU}$ - энергия, затраченная ЦПУ, $E_{GPU}$ - энергия, затраченная ГПУ, $CO_{2coef}$ – региональный коэффициент эмиссии  CO2. Подробное описание методики расчета можно найти в статье "eco2AI: carbon emissions tracking of machine learning models as the first step towards sustainable AI".

Таким образом, СО2 эффективность решения складывается из трех составляющих:

1.   эффективность решения (влияет на время обучения)
2.   энергоэффективность ГПУ и ЦПУ устройств 
3.   регион из которого берутся вычислительные ресурсы

Значения коэффициентов эмиссии и мощности устройств можно найти на сайте https://github.com/sb-ai-lab/Eco2AI. Для того чтобы произвести расчет с меньшим региональным коэффициентом эмиссии вы можете использовать гугл колаб или найти другие сервисы. Обязательно расскажите о своем поиске в презентации.

### Подведение итогов и награждение в номинации "Самое экологичное решение"
Победителем в номинации будет считаться решение, показавшее наименьшее значение "CO_2  emissions" в процессе обучения модели и попавшее в топ решений, при условии выполнения всех инструкций из пункта "Описание номинации". В топ попадают решения, преодолевшие пороговое значение  $nomination$ по основной метрике, определяемое по следующей формуле:

$$nomination=mean(Main_{metrics} )+std( Main_{metrics} )$$

<!-- nomination=mean(Main_{metrics} )-std( Main_{metrics} )$$ -->

где $mean(Main_{metrics} )$ – среднее значение массива значений метрик, превзошедших baseline метрику задачи, $std( Main_{metrics} )$ – стандартное отклонение массива значений метрик, превзошедших baseline метрику задачи.

Победитель номинации "Самое экологичное решение" получает приз в размере 100 000 рублей.

После проведения соревнования победителю в дополнительной номинации необходимо будет защитить свое решение в формате презентации, в которой участнику предстоит рассказать о процессе обучения своей модели и использованных для этого вычислительных ресурсов.
