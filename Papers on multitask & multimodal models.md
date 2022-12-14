# Статьи по мультизадачным и мультимодальным моделям

Создание мультизадачных и мультимодальных моделей в настоящее время – активно развивающая сфера исследований, которая позволяет экономить вычислительные ресурсы и время, необходимое для обучения, а также впоследствии, возможно, приведет к созданию сильного искусственного интеллекта. 

1. [OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/pdf/2202.03052.pdf) (```Li et al., 2022```) — в статье описывается sequence-to-sequence фреймворк OFA, который позволяет решать в визуальной и текстовой модальностях множество разнообразных задач (кроссмодальных и унимодальных), среди которых генерация изображений (image generation), создание описаний изображений (image captioning), классификация изображений (image classification), языковое моделирование (language modelling), генерация ответов на вопросы по изображениям (VQA) и др. Авторы выделяют 3 главных качества, присущих модели: 

* независимость от задач: все задачи для предобучения и файнтюна формулируются в едином sequence-to-sequence формате и содержат инструкции на естественном языке (например, “Which region does the text x^t describe?”, “Does the image describe x^t?” и др.)
* независимость от модальности: модель построена на основе архитектуры трансформер, которая уже продемонстрировала способность успешно обрабатывать данные, относящиеся к различным модальностям (Pretrained Transformers As Universal Computation Engines; ```Lu et al., 2001```); для решения последующих (downstream) задач не добавляются обучаемые компоненты, которые были бы специфичны для конкретной модальности и/или задачи
* полнота решаемых задач: предобучение происходит на множестве унимодальных и кроссмодальных задач.

<p align="center">
  <img src="https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/images_git/ofa.png" width="60%">
</p>

Для того, чтобы модель могла обрабатывать данные, относящиеся к различным модальностям, текст, изображения и объекты на них дискретизируются и кодируются токенами унифицированного, единого для всех задач словаря. В частности, для задач, связанных с определением области на изображении (например, visual grounding), в словарь вводятся специальные локационные токены ⟨x1, y1, x2, y2⟩. Изображения обрабатываются следующим образом: после приведения к единому разрешению (оно разное для разных размеров модели; в базовой версии – 384x384) они делятся на 16x16 патчей, каждый из которых пропускается затем через 3 первых блока модуля ResNet для получения визуальных признаков. Тексты обрабатываются стандартным образом – с помощью BPE. 

Сама модель имеет архитектуру вида «кодировщик-декодировщик» (encoder-decoder) – используются стандартные модули трансформера (конфигурация аналогична BART) с некоторыми модификациями. В качестве данных для предобучения используются 20 млн пар «картинка – текст» из общедоступных датасетов. Тот факт, что во время предобучения модели в постановке задач содержатся инструкции на естественном языке, делает возможным инференс в режиме zero-shot (на новых задачах и без обучающих примеров) – качество работы модели на таких задачах, однако, в значительной степени зависит от того, как именно будут сформулированы эти инструкции (авторы указывают, что для достижения наилучшего результата приходится искать подходящий шаблон инструкции из большого пула кандидатов). Модель так же может адаптироваться к данным, относящимся к другому домену, нежели те, на которых она обучалась. 

2. [A Generalist Agent](https://arxiv.org/pdf/2205.06175.pdf) (Gato, ```Reed et al., 2022```) – это мультимодальная модель, способная решать более 600 различных задач. Такое впечатляющее количество достигается за счет большого числа игровых задач (подобных Atari), в которых модель выполняет роль агента и генерирует действия по входному контексту, помимо игровых задач, модель решает задачу языкового моделирования (language modeling) и ряд image-to-text задач (image captioning, visual question answering). Gato может работать с разнообразными входными данными: текстами и изображениями, стандартными для моделей глубинного обучения, а также с непрерывными и дискретными числовыми признаками. 

По принципу действия Gato похожа на генеративные языковые модели и работает как «декодировщик». Выходная последовательность генерируется Gato токен за токеном по известной входной последовательности. Лосс считается по текстовым, некоторым дискретным и непрерывным токенам (например, по тем, которые кодируют действие агента); токены изображения маскируются и не вносят вклад в расчет лосса, следовательно Gato не может генерировать изображения. Несмотря на это, архитектурных ограничений на генерацию изображений нет, внесение некоторых изменений в процесс обучения может позволить модели также создавать изображения по входной последовательности.

Данные, относящиеся к разным модальностям, с которыми может работать модель, кодируются в единую последовательность токенов после соответствующей процедуры токенизации. Текстовые данные кодируются с использованием SentencePiece (```Kudo and Richardson, 2018```) токенизатора, непрерывные признаки сначала нормализуются по мю-закону, затем дискретизируются разбиением на 1024 бина; дискретные признаки кодируются соответствующим им целым числом (при этом, соответствующие id токенов в словаре сдвигаются так, чтобы не было пересечения со словарем для текстовых токенов). Изображения кодируются по следующей схеме: сначала они разбиваются на патчи размером 16x16, далее значения внутри каждого патча нормализуются. Для получения визуальных признаков патчи проходят через один блок модели ResNet (```He et al., 2016```) с добавлением позиционных эмбеддингов.

В процессе обучения модель работает с несколькими задачами одновременно. Токенизированные последовательности сэмплов, относящиеся к различным задачам, объединяются в батчи, и модель учится генерировать выход для всего батча (т.е. одновременно для разных задач).

<p align="center">
  <img src="https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/images_git/gato.png" width="60%">
</p>

Для того чтобы иметь представление о решаемой задаче, авторы предлагают использовать так называемый prompt сonditioning. На вход модели подается краткий сценарий решаемой задачи, например 5 пар изображений и соответствующих им текстов, далее следует изображение, для которого модели нужно сгенерировать описание. На этапе предобучения подобные сценарии поведения добавлялись в начало 25% входных последовательностей. Такая постановка позволяет модели адаптировать текущее поведение под входной контекст и конкретную решаемую задачу.

Архитектурные особенности модели: Gato имеет 1.2 миллиарда обучаемых параметров, включает в себя 24 nсандартных декодер-слоев, размер эмбеддингов – 2048, размер скрытого слоя – 8196. Датасеты для обучения содержат примерно 1.5T токенов, при этом, 85% всего обучающего датасета составляют данные, относящиеся к игровым задачам, около 15% относятся к задачам типа image-text и text-text.

3. [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/pdf/2204.14198.pdf) (Flamingo, ```Alayrac et al., 2022```) - мультимодальная модель DeepMind, способная генерировать текстовое описание фото, видео и звуков. Модель превосходит предыдущие state-of-the-art модели во многих открытых визуальных и текстовых задачах. Flamingo способна в zero-shot режиме давать ответы по различным задачам, опираясь всего на несколько input/output примеров (VQA, captioning, классификация с выбором ответа из множества).

<p align="center">
  <img src="https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/images_git/flamingo2.png" width="70%">
</p>

 Архитектура Flamingo представляет собой комбинацию предобученных моделей - Vision Encoder и Language Model, веса которых замораживаются во время обучения. 
 * Vision Encoder представляет собой модель Normalizer-Free Resnet, обученную на парах текст-картинка с использованием contrastive loss, подобно CLIP. Энкодер извлекает семантические пространственные признаки, такие как цвет, форма, расположение объекта и др. 
 * Language Model - авторегрессионная языковая модель Chinchilla 70B, обеспечивающая сильные генеративные способности языка и доступ к большому количеству знаний. 
 
 Чтобы объединить предобученные модели используются:
 * Perceiver Resampler - трансформер, преобразующий эмбеддинги из Visual Encoder в эмбеддинги меньшего фиксированного размера.
 * Cross attention layers - встраиваются между замороженными обученными слоями LM. Эти слои позволяют объединять визуальную информацию и текстовые признаки, получая на выходе текстовые токены.
 
Flamingo может обрабатывать за раз несколько различных пар изображение/текст или кадров из видео/текст. Связываются они следующим оборазом. Вводится специальная функция, которая присваивает каждой позиции текста индекс последнего изображения/видео, появляющегося перед текущей позицией. Таким образом, получается, что каждый текстовый токен смотрит только на токены картинки, которая появилась непосредственно перед ним.  Это работает лучше, чем если бы текстовый токен напрямую связывался со всеми предыдущими изображениями. Несмотря на то, что модель может обращаться в определенный момент времени только к одному изображению, по прежнему остается зависимость от других изображений в последовательности через self-attention в декодере текста. Преимущество состоит в том, что такая схема легко может обобщаться на любое количество изображений.

Flamingo обучается на 3 типах датасетов - пары изображение/текст, пары видео/текст и чередующийся набор изображений/текстов, полученных с веб-страниц. Во время обучения в качестве лосс функции используется взвешенная сумма negative log likelihood по типам датасетов, поскольку разные датасеты имеют разные свойства (качество текста, степень сооответствия текст/картинка), и это нужно учитывать при расчете лосс функции.
 
4. [UniT: Multimodal Multitask Learning with a Unified Transformer](https://arxiv.org/pdf/2102.10772.pdf) (Unified Transformer (UniT), ```Hu, Singh, 2021```), который является частью фреймворка [MMF](https://github.com/facebookresearch/mmf) для построения мультимодальных моделей. В экспериментах UniT одновременно обучается на 8 датасетах (MS-COCO, VG, VQAv2, SNLI-VE, QNLI, MNLI-mm, QQP, SST-2) по 7 задачам (в текстовом, визуальном и совместном текстово-визуальном доменах). Модель имеет архитектуру типа «кодировщик-декодировщик» (encoder-decoder): для каждого типа данных (модальностей, в данном случае – текстовой и визуальной) используется свой кодировщик; декодировщик при этом единый для всех задач. Выход декодировщика отправляется в специфическую для конкретного задания «голову» (для всех задач, кроме object detection, она представляет собой двуслойный перцептрон), которая и выдаёт финальное предсказание:

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/unit.png" width="60%">
</p>

При каждой итерации во время обучения для формирования батча выбирается задание и соответствующий ему датасет; для каждого задания заранее задаётся вероятность сэмплирования.

5. [Attention Bottlenecks for Multimodal Fusion](https://arxiv.org/abs/2107.00135) (Multimodal Bottleneck Transformer (MBT), `Nagrani, Arsha et al., 2021`) - оригинальный подход к работе с мультимодальными входами (в основном видео и аудио) со средним уровнем слияния.

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/mid_fuse.png" width="60%">
</p>
Почти все слои обрабатываются трансформатором отдельно, и только рядом с верхним (2-4 слоя)  производится слияние:
    -Шаг А: нейроны модаальности<sub>1</sub> соединияются с небольшим количеством B так называемых мультимодальных bottlenecks (авторы брали B=4) и, затем реализуется self-attention механизм, потом
    -Шаг B: нейроны модаальности<sub>2</sub> соединияются с небольшим количеством B  мультимодальных bottlenecks (соответствующих выходу шагу A) и, затем реализуется self-attention механизм (мдальности<sub>2</sub> + мультимодальные bottlenecks).

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/mbt.png" width="60%">
</p>
Самое интересное, что обмен мультимодальной информацией происходит через очень узкий информационный bottleneck одновременно без добавления вычислительной сложности.
В качестве трансформера авторы используют ViT-B архитектуру.

6. [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206) and [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795) (`Jaegle, Andrew et al., 2021`) - the novel method of dealing with attentions for multi-modal data with linear complexity on either input size or output size. Две основные идеи:
- Iterative attention, когда один и тот же вход может быть передан на разную глубину - как процедура RNN (авторы утверждают, что эта идея лежит в основе работы человеческого мозга) - где параметры трансформеров могут быть общими, и

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/perceiver.png" width="60%">
</p>

- Cross-attention, когда либо запрос является латентным потоком и ключевые значения являются входными данными (для подачи данных в модель), либо запрос является выходной структурой (например, список позиций пикселей и задание<sub>id</sub>)

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/perceiverIO.png" width="60%">
</p>
Здесь важно отметить, как вводить информацию вход или выход). Авторы предлагают различные схемы кодирования позиции (включая признаки Фурье) или даже выученную кодировку (так что нет необходимости в явной токенизации).


<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/perceiverIO_emb.png" width="60%">
</p>

Архитектура латентного трансформера GPT-2. 

7. [Multi-Task Deep Neural Network](https://github.com/namisan/mt-dnn) (MT-DNN, ```Liu, He et al., 2019```) – единая модель, созданная для решения различных задач в области понимания естественного языка (NLU). Нижние слои модели едины для всех задач, верхние слои специфичны для каждого типа задания. В качестве кодировщика используется многослойный двунаправленный кодировщик Трансформера – однако, в отличие от BERT, MT-DNN выучивает репрезентации не только с помощью предобучения, но и с помощью мультизадачных целевых функций. 

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/mt-dnn.png" width="60%">
</p>

Процедура обучения MT-DNN состоит из двух стадий: предобучения и мультизадачного обучения. Во время мультизадачной фазы в каждую эпоху выбирается мини-батч (среди всех 9 заданий GLUE) – и веса модели обновляются согласно целевой функции, использующейся для конкретного задания. Такой подход аппроксимативно оптимизирует сумму целевых функций для всех задач. Авторы подчеркивают, что мультизадачное обучение имеет преимущества за счет эффекта регуляризации (меньший риск переобучения на конкретной задаче) – благодаря этому выученные репрезентации данных получаются более универсальными. MT-DNN обучалась на датасетах SNLI, SciTail и GLUE, решая 4 типа заданий: классификация единичных предложений, классификация пар текстов, оценка близости текстов, ранжирование текстов по релевантности.

8. [OmniNet: A unified architecture for multi-modal multi-task learning](https://arxiv.org/pdf/1907.07804.pdf) (`Pramanik et al., 2020`) — модель [OmniNet](https://github.com/subho406/OmniNet) включает два блока «периферических сетей» (peripheral networks): один кодирует картинки и видео, другой — текстовые данные. В качестве кодировщика изображений и видео используется предобученная сверточная нейросеть ResNet-152. Для кодирования текстов применяются предобученные подсловные эмбеддинги, полученные с помощью BPE. Закодированные данные конкатенируются с эмбеддингами типа данных и поступают в центральный блок (Central Neural Processor). Данные, у которых есть временная размерность, проходят через блок кодировщика; данные, у которых временной размерности нет, подвергаются преобразованию размерности (reshape). Все полученные матрицы сохраняются в кэш, который поступает в блок декодера:

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/omninet.png" width="70%">
</p>

Предусмотрены эмбеддинги заданий, несколько наборов выходных эмбеддинов для разных типов выходных данных и отдельные классификаторы для разных заданий. Мультизадачность достигается с помощью подхода HogWild: помимо глобальной копии модели, создаются локальные копии для каждого задания; вычисленные локальные градиенты асинхронно копируются в глобальную модель, затем происходит обновление её весов. Модель обучалась на задачах частеречной разметки, генерации ответов на вопросы по изображению, генерации подписей к картинкам и распознавания действий на видео.

9. [12-in-1: Multi-Task Vision and Language Representation Learning](https://arxiv.org/pdf/1912.02315.pdf) (`Lu, Goswami et al., 2020`) — в статье описывается модель, основанная на архитектуре [ViLBERT](https://arxiv.org/pdf/1908.02265.pdf) (`Lu et al., 2019`), в которой текстовые и визуальные входные данные взаимодействуют посредством слоёв co-attention (механизма "взаимного внимания"):

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/vilbert.png" width="80%">
</p>

Изображение и текст, к которым добавляется соответствующий заданию токен, кодируются с помощью двух блоков, архитектура которых подобна BERT. Эти модули предобучены на задачах предсказания маскированного элемента и предсказания наличия связи между входными данными. Авторы использовали 12 датасетов с задачами, основанных на изображениях и текстах. Эти задачи можно разделить на следующие группы: выбор подходящего ответа на вопрос по изображению, выбор изображения, соответствующего описанию, выбор фрагмента изображения, соответствующего описанию, проверка, соответствуют ли друг другу изображение и текст Для решения этих задач обучено шесть голов модели. При обучении используется предложенный авторами метод stop-and-go, позволяющий приоставливать использование датасетов небольшого объема на большее количество итераций и продолжать использовать большие датасеты. 

10. [M3P: Learning Universal Representations via Multitask Multilingual Multimodal Pre-training](https://arxiv.org/pdf/2006.02635.pdf) (`Ni et al., 2021`) — модель, в которой мультилингвальное предобучение и мультимодальное предобучение комбинируются с помощью мультизадачности в едином фреймворке:

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/m3p.png" width="80%">
</p>

Мультизадачность интегрирована в стадию предобучения для одновременной оптимизации всех выбранных целевых функций. Сначала модель предобучается на задаче предсказания маскированного токена, при этом используются три типа входных данных: текст на одном языке + изображения; текст, в котором происходит смена языка (Multimodal Code-switched Training); сочетание первых двух случаев. Далее модель дообучается на двух мультилингвальных датасетах, используемых для задачи поиска изображений: Multi30K — расширенная версия Flickr30K (английские, немецкие, французские и чешские подписи) and MS-COCO (английские, китайские и японские подписи). 

11. [HyperGrid Transformers: Towards A Single Model for Multiple Tasks](https://openreview.net/pdf?id=hiq1rHO8pNT) (`Tay et al., 2021`) – авторы предлагают подход, при котором мультизадачное обучение модели обеспечивается благодаря использованию декомпозиционной гиперсети (сети, которая генерирует веса для основной модели), которая выучивает grid-wise проекции, позволяющие выделять отдельные регионы в матрице весов в зависимости от задачи, обеспечивая специализацию подсети. Для построения подобной гиперсети используются локальная (зависящая от примера и задачи) и глобальная (независимая от задачи) проекции: композиция локального и глобального векторов формирует матрицу гейтирования, которая затем расширяется (повтором) до размера матрицы весов Трансформера для получения специфичной для каждого задания матрицы весов:

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/hypergrid.png" width="80%">
</p>

Авторы проводили эксперименты на бенчмарках GLUE и SuperGLUE (задачи NLP и NLU). В качестве основной модели используется T5, к ней добавляются слои HyperGrid. Результаты демонстрируют, что качество единой модели (Avg 85.0 на GLUE и 73.6 на SuperGLUE) не намного уступает качеству отдельных моделей, дообученных под конкретные задачи (Avg 85.7 на GLUE и 74.8 на SuperGLUE), однако обучение единой архитектуры требует изменения в 16 раз меньшего количества параметров.  

12. [AdapterHub: A Framework for Adapting Transformers](https://arxiv.org/pdf/2007.07779v1.pdf) (`Pfeiffer et al., 2020`) – в статье описывается новый подход к дообучению базовых моделей (foundation models): вместо того, чтобы полностью дообучать тяжеловесные модели, предлагается использовать «адаптеры» – легковесные слои, которые включаются в каждый слой большой предобученной модели; именно эти слои обучаются в процессе дообучения, в то время как веса трансформера остаются неизменными. Таким образом, с помощью адаптеров происходит кодирование специфичных для конкретного задания репрезентаций в слоях предобученной модели:

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/adapters.png" width="60%">
</p>

Адаптеры располагаются инкапсулированно, имеют модульную природу, что позволяет легко сочетать их друг с другом: наслаивать или менять динамически в процессе дообучения, комбинируя несколько источников релевантной информации. Авторы отмечают, что стандартные подходы к мультизадачному обучению (multi-task learning, MTL) имеют следующие недостатки: «катастрофическое забывание», когда информация, выученная на предыдущих стадиях обучения «перезаписывается»; падение качества предсказаний для множества заданий при добавлении новых; сложности с балансировкой задач. 
Инкапсулированность адаптеров позволяет им выучивать репрезентации для разных задач, совместимые друг с другом, – несколько адаптеров могут быть скомбинированы, например, с помощью механизма внимания ([MAD-X: An Adapter-Based Framework for
Multi-Task Cross-Lingual Transfer](https://arxiv.org/pdf/2005.00052.pdf) `Pfeiffer et al., 2020`). Адаптеры обучаются отдельно друг от друга, благодаря чему отпадает необходимость использовать эвристики при сэмплировании датасетов для разных задач; разделение двух процессов – извлечения знания и его объединения – позволяет легко добавлять новые задачи и решает проблему «катастрофического забывания». Фреймворк [AdapterHub](https://adapterhub.ml/), построенный поверх библиотеки Transformers от Hugging Face, позволяет легко экспериментировать с адаптерами. 

13. [Conditionally Adaptive Multi-Task Learning: Improving Transfer Learning in NLP Using Fewer Parameters & Less Data](https://arxiv.org/abs/2009.09139) (CA-MTL, `Pilaut et al., 2020`) - для решения одновременно нескольких NLP задач в классическом Multi-Task сеттинге, авторы предлагают модифицированную архитектуру трансформеров: обучаются специфичные для задачи эмбеддинги и специальный слой Conditional Alignment, выравнивающий входные эмбеддинги для соответствующей задачи, а также в половине слоёв предобученной модели заморожены, в то время как в другой половине добавлены обучаемы специфичные к задаче Conditional Attention, Conditional Layer Normalization и аналогичный идее адаптеров Conditional Bottleneck модули. Такое архитектурное решение позволяет в отличие от адаптеров обучаться одновременно на нескольких датасетах, а не обучать каждый адаптер по отдельности, при этом оставляя большую часть параметров модели общей. Для решения проблемы, связанной с неоднородностью данных в Multi-Task датасетах, авторы предлагают сэмплировать в батч только те примеры, в которых модель наименее уверенна в своих предсказаниях (т.е. наибольшая энтропия Шэннона), эта идея называется Multi-Task Uncertainty Sampling. В итоге предложенная архитектура позвоялет добиться лучших результатов на GLUE и SuperGLUE датасетах, чем файн-тюнинг для каждой отдельной задачи (т.е. обучая всего 1.12x, а не 24x параметров) и показывает отличные результаты в zero-shot и few-shot сеттинге.

<p align="center">
  <img src="https://i.ibb.co/0t3LVY3/CA-MTL.png" width="60%">
</p>

