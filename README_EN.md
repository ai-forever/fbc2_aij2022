
# FusionBrain Challenge 2.0


## General task description

As part of this task, it is proposed to build a single multitask model that would successfully complete sub-tasks in two modalities (visual and textual) after receiving sub-task descriptions expressed in natural Russian language, e.g. "generate an image", "describe an image", "answer a question", etc. at the input. It consists of 12 sub-tasks, 6 of which are known to participants before the beginning of the Contest (open sub-tasks), and 6 are unknown (hidden sub-tasks) that are connected with open sub-tasks but have some specific features in their setting. The key task of the participants is to build and train a single multimodal multitask architecture that would allow getting maximum metric values for each individual sub-task and, therefore, achieving the maximum value of integral metric for 12 sub-tasks.

## General solution format

### Container contents

The checking system should receive an algorithm code packed as a ZIP archive. Solutions shall be run in the isolated environment using the Docker. Time and resources during testing shall be limited. Participants do not have to dig into the Docker technology.

The archive root must contain a `metadata.json` file with the following contents:
```
{
    "image": "cr.msk.sbercloud.ru/aijcontest2022_official/fusion:0.0.1",
    "entrypoint": "python3 run.py"
}
```

Where `image` is a field with a name of docker image where the solution will be run, and `entrypoint` is a command used to run the inference script. There is a specific current directory for each team's solution.

It is possible to use the existing environment to run the solutions:

* ```cr.msk.sbercloud.ru/aijcontest2022_official/fusion:0.0.1``` - [Dockerfile and requirements](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/docker_FB2.zip) for this image.

If needed, you may prepare your own docker image by adding the required software and libraries to that (see [the instruction for creation of Docker images](https://dsworks.s3pd01.sbercloud.ru/static/champ/aij22/%D0%98%D0%BD%D1%81%D1%82%D1%80%D1%83%D0%BA%D1%86%D0%B8%D1%8F%20%D0%BF%D0%BE%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B5%20%D1%81%D0%BE%20sbercloud_en.pdf)); to use that, it will be necessary to publish that on ```sbercloud```. Custom images should be inherited from basic ```sbercloud``` images (see [basic images](https://docs.sbercloud.ru/aicloud/mlspace/concepts/environments__basic-images-for-training.html)). When creating a custom image, it is necessary to assign it an individual name and tag (for example, ```my_custom_fusionchallenge:0.0.5```).

### Baseline

Baseline is built using the [RUDOLPH](https://github.com/ai-forever/ru-dolph) model. RUDOLPH is a multi-tasking decoder-based model that can solve a range of tasks within two modalities (text and image) that corresponds the FBC2 rules. 

There are three versions of the RUDOLPH model: 350M, 1.3B, 2.7B. In the baseline, we used RUDOLPH 2.7B fine-tuned for 6 open tasks from FBC2 ([RUDOLPH-2.7B-FBC2](https://huggingface.co/sberbank-ai/RUDOLPH-2.7B-FBC2)). The description of training data for each open task is given below.

The code for baseline can be downloaded by [_sample\_submission.zip_](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/sample_submission.zip) link.

### Data structure

The data constitute a dictionary: the value of each key is a list whose contents depend on the data needed to set a specific task: the `type` field specifies the data type (`text` or `image`), while `content` shows the actual contents (for example, a question or answer options in natural language, a path to file with image). In general, the format looks as follows:
```
{"0": [{"type": ..., "content": ...}, ...], "1": [...], ...} 
```

### Solution format

The data for prediction combine all sub-tasks and include:

* The _input.json_ file. It is a dictionary in such format as `{"0": [{"type": "text", "content": "Ответь на вопрос по изображению. На чём едет человек справа на картинке?"}, {"type": "image", "content": "images/00000.jpg"}, {"type": "text", "content": "варианты: a) на велосипеде; b) на лошади; c) на машине; d) на самокате"}], "1": [{"type": "text", "content": "Дай описание изображения."}, {"type": "image", "content": "images/00005.jpg"}], "2": [{"type": "text", "content": "Сгенерируй на изображении."}, {"type": "text", "content": "Пара человек сидит в тени. Велосипед, оставленный на велопарковке, стоит на солнце."], ...}`. The keys are represented by IDs of examples, while values - by the list of dictionaries described in **Data structure**.
* The _images_ folder contains a set of images that should be used to make predictions. It includes files in such format as 00000.jpg, 00001.jpg, ....

Input data (_input.json_ and images folder) is located in the _input_ directory (it is located one level up the directory where decision script is run).

**Please note**: in some sub-tasks, the actual task description ("ответь на вопрос", "выбери правильный вариант", etc.) may be given in a separate dictionary `{"type": "text", "content": "Ответь на вопрос по изображению."}`, be combined with another text field `{"type": "text", "content": "Ответь на вопрос по изображению. На чём едет человек справа на картинке?"}`, or be absent `{"type": "text", "content": "На чём едет человек справа на картинке?"}`.

**Please note**: in some sub-tasks, the actual task description ("ответь на вопрос", "выбери правильный вариант", etc.) may be given in a separate dictionary `{"type": "text", "content": "Ответь на вопрос по изображению."}`, be combined with another text field `{"type": "text", "content": "Ответь на вопрос по изображению. На чём едет человек справа на картинке?"}`, or be absent `{"type": "text", "content": "На чём едет человек справа на картинке?"}`.

The participant's model should make predictions for all input data from the _input.json_ file and generate output data, including:

* The _predictions.json_ file. It is a dictionary in such format as `{"0": [{"type": "text", "content": "a)"}, {"type": "image", "content": "images/image_0.jpg"}], "1": [{"type": "text", "content": "две большие птицы летают над озером."}], "2": [{"type": "image", "content": "images/image_2.jpg"}], ...}`. Keys are represented by IDs of examples from _input.json_, while values - by the list of dictionaries, where the `type` field specifies a data type generated by the model (`text` or `image`), while `content` shows the actual contents (for example, a textual answer, a path to file with generated image).
* The images folder. It is a set of images generated by the model for some (or all) examples from the input file _input.json_. It includes files in such format as image_0.jpg, image_1.jpg ...

**Please note**: All the output data (such as _prediction.json_ and image folder _images_) should be located in the _output_ directory inside the working directory (i.e., _./output_).
    
The file of correct answers _ground\_truth.json_, which will be used to evaluate the model quality when running it in the container, is a dictionary in the following format: `{"0": [{"type": "text", "content": ["a на велосипеде", "a", "на велосипеде"]}], "1": [{"type": "text", "content": ["две птицы летают над водой", "птицы летают", "два сокола летят над озером"]}], "2": [{"type": "image", "content": "gt_images/image_2.jpg"}], ...}`. 

After that, the system shall break the file with predictions _predictions.json_ and the file with correct answers _ground\_truth.json_ into several files for each sub-task, compare the obtained files and display metric values for each sub-task and an integral metric value.

## Limitations

During one day, a Participant or a Team of Participants may upload no more than 3 (three) solutions for evaluation. At the same time, only valid attempts with numeric grades will be taken into account. In case of raised Exception during some metric calculation, an attempt will not be taken into consideration.

A container with the solution will be run under the following conditions:
* 100 Gb RAM
* 3 vCPU
* 1 GPU Tesla A100 (80 Gb)
* time to execute the solution: 5 hours
* the solution does not have access to internet resources;
* maximum size of packed and unpacked archive with the solution: 15 Gb
* maximum size of the Docker image used: 25 Gb

We provide the participants with an opportunity to get access to computing resources of Christofari to train their models. The resources are limited. To get access, it is necessary to send a request to Christofari_AIJContest_2022@sberbank.ru describing how exactly you are planning to use the computing resources.

# Sub-task 1 - Text QA (TextQA)

## Description 

Machine reading comprehension task; to be successful, the model should be able to find cause-and-effect relationships in the text. Each sample consists of a text, questions to that and answer options (optionally). The corpus includes questions of several types; to answer them successfully, the model should be able to infer cause-and-effect relationships, solve co-references, and determine the correct sequence of actions with allowance for temporary information. If only a question is given to the input text, the model should generate an appropriate answer; if there are also some answer options to the input text, the model should choose the correct one (there may be 4 answer options and only one of them is correct).

## Data

**Train** 

We used Sber Question Answering Dataset ([SberQuAD](https://huggingface.co/datasets/sberquad)) as a training dataset for TextQA. SberQuAD dataset was introduced for machine reading comprehension task. The dataset consists of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage. The dataset includes 45328 training, 5036 validation, and 23936 test samples.

As the additional source of training examples Russian Multi-Sentence Reading Comprehension ([MuSeRC](https://russiansuperglue.com/tasks/task_info/MuSeRC)) dataset can be used. The MuSeRC data set was proposed to check the model's ability to answer questions to the input text. The dataset contains about 6,000 questions to more than 800 text paragraphs. It is important to note that the initial tagging of the MuSeRC data set is made for the binary classification task, which does not correspond to the proposed sub-task **TextQA**. You should translate the data from the dataset to the required format, to have an opportunity to use them in the model training.

For the convenience, we prepared training dataset for TextQA sub-task (the data can be found in the corresponding folder of the [archive](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/fb_train_2022.tar.gz)) in the same format as Test public and Test private sets.

### TextQA test data format

**Test public** 

The public leader board is based on the results of model predictions checkup on the data set corresponding to the **TextQA** task, collected and tagged by the organizers.

Test sets for the **TextQA** sub-task include questions in the **generative** format and the format with **choosing the correct option**.

TextQA sample from _input.json_ has the following format:

```
{
"0": [{"type": "text", "content": "Мужская сборная команда Норвегии по биатлону в рамках этапа Кубка мира в немецком Оберхофе выиграла эстафетную гонку. Вторыми стали французы, а бронзу получила немецкая команда. Российские биатлонисты не смогли побороться даже за четвертое место, отстав от норвежцев более чем на две минуты. Это худший результат сборной России в текущем сезоне. Четвёртыми в Оберхофе стали австрийцы. Напомним, что днем ранее российские биатлонистки выиграли свою эстафету. В составе сборной России выступали Анна Богалий-Титовец, Анна Булыгина, Ольга Медведцева и Светлана Слепцова. Они опередили своих основных соперниц - немок - всего на 0,3 секунды."}, {"type": "text", "content": "На сколько секунд сборная России опередила ближайших соперниц?"}, {"type": "text", "content": "a) на 0,3 секунды; b) на секунду; c) на 0,5 секунд"}],
"1": [{"type": "text", "content": "Определи правильный ответ по тексту."}, {"type": "text", "content": "Улица Авиаконструктора Сухого (название с 2004 года) — улица в Северном административном округе города Москвы на территории Хорошёвского района. Пролегает от Проектируемого проезда № 6161 до пересечения с юго-восточной частью взлётно-посадочной полосы Ходынского Поля. Название утверждено 30 марта 2004 года в честь знаменитого советского авиаконструктора Павла Осиповича Сухого (1895—1975)."}, {"type": "text", "content": "Какая улица пролегает от Проектируемого проезда № 6161 до пересечения с юго-восточной частью взлётно-посадочной полосы Ходынского Поля?"}],
}
```

TextQA ground truth from _ground\ truth.json_ has the following format:

```
{
"0": {"type": "text", "content": ["a", "на 0,3 секунды", "a на 0,3 секунды"]},
"1": {"type": "text", "content": ["Авиаконструктора Сухого"]},
…
}
```

**Test private** 

The private testing dataset is hidden from the participants. Its format is similar to the public testing sample.

## Solution format

The correct answer for the TextQA sub-task is an input text fragment answering the question asked (for the generative format) and a correct answer marker, or an actual text of the correct answer, or a marker and a subsequent text of the answer (for the format with choosing the correct option). The dictionary for TextQA questions has the following format: `{"0": [{"type": "text", "content": "_answer to question 0_"}, ...]}`

# Sub-task 2 - Mathematical QA (MathQA)

## Description 

This sub-task checks the model's ability to carry out simplest arithmetic operations needed to solve linear equations or linear equation systems, and to conduct comparison operations. The task consists of a mathematical example described in natural language: arithmetic operations used are: addition, subtraction, multiplication, and division. Unknown variables may be expressed by any Latin letters. If only a question is given to the input mathematical problem, the model should generate an appropriate answer; if there are also some answer options to the input mathematical problem, the model should choose the correct one (there may be several answer options and only one of them is correct).

## Data

**Train** 

You can prepare training dataset, which contains mathematical examples generated with the use of the [Mathematics Dataset library](https://github.com/deepmind/mathematics_dataset/tree/master/mathematics_dataset) and translate them into Russian (you may find useful to augment data so that arithmetic symbols ('+', '-', '*', '/', '=') are written in words: 'плюс', 'минус', etc.). When generating training examples, it is recommended to pay attention to the modules for linear equations and linear equation systems with 1-2 unknown variables (algebra ('linear_1d', 'linear_2d')).

For the convenience, we prepared training dataset for TextQA sub-task (the data can be found in the corresponding folder of the [archive](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/fb_train_2022.tar.gz)) in the same format as Test public and Test private sets.

### MathQA test data format

**Test public** 

The testing data set contains the textual statement of mathematical task to be solved. The testing set may include tasks to solve linear equations and linear equation systems, as well as tasks to compare two values. The text of mathematical expression is written in natural language, and therefore, in addition to mathematical symbols, the text may include such expressions as  "если к _x_ прибавить _y_, то ...", "_a_ равно 7 минус _b_", etc.

The MathQA task provides for the format with choosing the correct answer option, and therefore, the input data always includes a list of answer options with corresponding markers (Latin letters with a closing bracket: a), b), c) ...).

MathQA sample from _input.json_ has the following format:

```
{
"0": [{"type": "text", "content": "Решите линейное уравнение."}, {"type": "text", "content": "6 умножить на y – 54 это -72, для y."}, {"type": "text", "content": "a) -4; b) -20; c) -5; d) -3"}],
...
}
```

MathQA ground truth from _ground\ truth.json_ has the following format:

```
{
"0": {"type": "text", "content": ["d", "-3", "d -3"]},
…
}
```

**Test private** 

The private testing dataset is hidden from the participants. Its format is similar to the public testing sample.

## Solution format

As a correct answer for examples belonging to the MathQA task, the model should generate either a correct option marker, or a correct option text, or their combination. In a similar manner to **TextQA** with choosing the right answer, any input format of the above will be considered to be the correct answer. 

In general, the dictionary for MathQA questions has the following format: `{"0": [{"type": "text", "content": "_answer to question 0_"}, ...]}`.

# Sub-task 3 - Image Generation (ImageGeneration)

## Description 

This sub-task provides for generation of images based on textual descriptions in Russian. An answer to the sub-task shall be an image with contents corresponding to the input textual description. 

## Data

**Train** 

It is proposed to use the [COCO data set](https://cocodataset.org) for training, which contains "text-picture" pairs; all textual descriptions can be translated into Russian by a translator.

The dataset should consist of two json files, one of which contains the input data for the task: task text, textual description of picture, and the other contains paths to real pictures.

For the convenience, we prepared training dataset for TextQA sub-task (the data can be found in the corresponding folder of the [archive](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/fb_train_2022.tar.gz)) in the same format as Test public and Test private sets.

### ImageGeneration test data format

**Test public** 

The testing data set contains the textual statement of sub-task to be solved, and a request, for which it is necessary to generate an image.

ImageGeneration sample from _input.json_ has the following format:

```
{
"0": [{"type": "text", "content": " Сгенерируй изображение по тексту."}, {"type": "text", "content": "Торт со светло-бежевой глазурью, украшенный фигуркой медведя, держащий связку воздушных шариков."}],
...
}
```

ImageGeneration ground truth from _ground\ truth.json_ has the following format:

```
{
"0": [{"type": "image", "content": "images/00000.jpg"}],
...
}
```

**Test private** 

The private testing dataset is hidden from the participants. Its format is similar to the public testing sample.

## Solution format

Images generated by the model should be in the _images_ folder, and the path to them should be indicated in the corresponding field of the output dictionary `{"0": [{"type": "image", "content": "images/_name of image 0_"}, ...}`.

# Sub-task 4 - Image Captioning (ImageCaptioning)

## Description 

This sub-task assesses the model's ability to generate textual descriptions of images in Russian. An answer to the sub-task shall be a text string containing a description of the input image contents.

## Data

**Train** 

It is proposed to use the [COCO data set](https://cocodataset.org) for training, which contains "text-picture" pairs; all textual descriptions can be translated into Russian by a translator.

For the convenience, we prepared training dataset for TextQA sub-task (the data can be found in the corresponding folder of the [archive](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/fb_train_2022.tar.gz)) in the same format as Test public and Test private sets.

### ImageCaptioning test data format

**Test public** 

The testing data set contains the textual statement of sub-task to be solved, and a request, for which it is necessary to generate an image. The format of testing examples is similar to the _input.json_ format of the training data set.

ImageCaptioning sample from _input.json_ has the following format:

```
{
"0": [{"type": "text", "content": "Что представлено на картинке?"}, {"type": "image", "content": "images/00000.jpg"}],
...
}
```

Image Example:

`images/00000.jpg`: <img src="https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/images_web/image-caption.png" width=250>

ImageCaptioning ground truth from _ground\ truth.json_ has the following format:

```
{
"0": [{"type": "text", "content": ["Торт со светло-бежевой глазурью, украшенный фигуркой медведя, держащий связку воздушных шариков.", "Детский торт, покрытый светлой глазурью с украшением в виде медвежонка."]}],
...
}
```

**Test private** 

The private testing dataset is hidden from the participants. Its format is similar to the public testing sample.

## Solution format

As a valid solution, the model should generate an input image description `{"0": [{"type": "text", "content": "_caption for image 0_"}, ...], ...}`.

In a similar manner to the TextQA and MathQA sub-tasks, the _ground_truth.json_ file contains several possible descriptions for the input image; we will consider the maximum metric for all options.

# Sub-task 5 - Visual QA (VisualQA)

## Description 

The sub-task provides that a trained model is able to generate an answer to the question based on an image. In this sub-task, the model receives a "textual question - picture" pair at the input, and it should return the corresponding textual answer at the output. If only a question is given to the input image, the model should generate an appropriate answer; if there are also some answer options to the input text, the model should choose the correct one (there may be 4 answer options and only one of them is correct).

## Data

**Train** 

It is proposed to use the [Visual Genome data set](https://visualgenome.org/api/v0/api_home.html) for training, which contains "question-answer" pairs with pictures. All questions and answers can be translated by a translator. 

For the convenience, we prepared training dataset for TextQA sub-task (the data can be found in the corresponding folder of the [archive](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/fb_train_2022.tar.gz)) in the same format as Test public and Test private sets.

### VisualQA test data format

**Test public** 

The testing data set contains the textual statement of sub-task to be solved (optionally), a question, and an image, for which the question is asked. The format of testing examples is similar to the _input.json_ format of the training data set.

VisualQA sample from _input.json_ has the following format:

```
{
"0": [{"type": "text", "content": "Ответь на вопрос по картинке."}, {"type": "text", "content": "Какого цвета плакат на ограждении?"}, {"type": "image", "content": "images/00000.jpg"}], 
...
}
```

Image example:

`images/00000.jpg`: <img src="https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/images_web/image-2.png" width="250">


VisualQA ground truth from _ground\ truth.json_ has the following format:

```
{
"0": [{"type": "text", "content": ["желтого", "ярко-желтого"]}], 
...
}
```

**Test private** 

The private testing dataset is hidden from the participants. Its format is similar to the public testing sample.

## Solution format

As a valid solution, the model should generate an answer to the question based on the image `{"0": [{"type": "text", "content": "_answer to question 0_"}, ...], ...}`.

# Sub-task 6 - Text Recognition in the Wild (TRitW)

## Description 

A task to recognize texts in urban or other similar environment (sign boards, road signs, advertisements, etc.) The data includes photos of objects with some text on them. An answer to the sub-task shall be a text string containing all Cyrillic and Latin letters, as well as generally accepted wildcards recognized in the input image.

**Train** 

There is a novel dataset START (**S**yn**T**hesized and **A**nnotated dataset for **T**ext **R**ecognition) consisting of the synthetic and real-world human-annotated data with text in Russian. Synthetic data includes 140 000 images taken from the open dataset [COCO](https://cocodataset.org). The images are supplemented with some text (the text is situated in various parts of the images, turned at various angles, has various colors and transparency, etc.). Real-world data consists of 40 312 urban photos with various text labels both in Russian (mainly) and English. Also, as additional data, you can use [dataset from SberIDP](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/titw_sberidp.zip), which contains 17,000 images with text on a monochrome background. The model shall be trained to recognize a text in the image.

### TRitW test data format

**Test public** 

The testing data set contains the textual statement of sub-task to be solved (optionally), a question, and an image, for which the question is asked. The format of testing examples is similar to the _input.json_ format of the training data set.

TRitW sample from _input.json_ has the following format:

```
{
"0": [{"type": "text", "content": "Распознай текст на картинке."}, {"type": "image", "content": "images/00000.jpg"}],
...
}
```

Image example:

`images/00000.jpg`: <img src="https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/images_web/image-4.jpeg" width="250">

TRitW ground truth from _ground\ truth.json_ has the following format:

```
{
"0": [{"type": "text", "content": ["хор"]}],
...
}
```

**Test private** 

The private testing dataset is hidden from the participants. Its format is similar to the public testing sample.

## Solution format

As a valid solution, the model should generate a text recognized in the image `{"0": [{"type": "text", "content": "_text in image 0_"}, ...], ...}`.

# Evaluation metrics

You can find the description of each evaluation metric by the [following link](https://github.com/ai-forever/fbc2_aij2022/blob/main/Metrics_en.md).

### General rules for metric calculation

* When calculating metrics for tasks with expected textual output (for example, among open tasks, this type includes TextQA, MathQA, VisualQA, ImageCaptioning, TRitW), the answer shall be read from the **first** dictionary, where the ```"type"``` key corresponds to the ```"text"``` value.
* If the model generates an image for sub-tasks with textual output, in addition to the textual answer, it shall not be considered as an error, and a dictionary for the image (```{"type": "image", "content": ...```) shall be ignored when calculating metrics.
* In most cases for sub-tasks with textual output, a correct answers file (_ground\_truth.json_) for Test public and Test private shall contain several correct answer options for each input example (e.g. different options of the correct answer, synonyms to the correct answer, etc.). When calculating metrics for tasks with textual output, a prediction generated by the model shall be compared to each answer option from _ground_truth.json_, and the maximum calculated metric value shall be saved (for details see the sub-task descriptions).
* Before calculation of metrics, the post-processing of generated predictions is conducted: reducing to lower case, deleting punctuation characters, reducing words to normal form.
* When calculating metrics for sub-tasks with image output (e.g. Image Generation), the answer shall be read from the first dictionary, where the ```"type"``` key corresponds to the ```"image"``` value. If the model generates some text in addition to the image, it shall not be considered as an error, and a dictionary for the textual output  (```{"type": "text", "content": ...```) shall be ignored when calculating metrics.

### Ground truth

A file with correct answers (ground truth) for Test public and Test private contains several correct answer options for each input example. When calculating metrics for tasks with textual output, a prediction generated by the model shall be compared to each answer option from ground truth, and the maximum calculated metric value shall be saved.

Example:

predictions.json

```
{"0": [{"type": "text", "content": "a) 5"}]}
```

ground_truth.json

```
{"0": [{"type": "text", "content": ["a", "a 5", "5"]}]}
```

In this case, the maximum metric value shall be obtained when comparing the predicted value of _"a) 5"_ with the true option of _"a 5"_. 

# Test sample set

For six open tasks, we prepared a _test sample set_ containing 20 examples for each task. The input_sample.json and output_sample.json file formats match the format of corresponding files from _public test_ and _private test_. You may use these files to check the correctness of your solution before sending.

**Test sample sets** contains _sample_ids.json_ file that is a mapping between sample id and task id (```{"sample_id": task_id, ...} ```).

_task_id_ can take value from 0 to 5; the mapping between _task_id_ and task name is given below:

* 0: TextQA
* 1: MathQA
* 2: ImageGeneration
* 3: ImageCaptioning
* 4: VisualQA
* 5: TRitW

Please note that _public_ and _private test_ do not contain any indication of task type.

**Test sample set** can be downloaded by the [following link](https://n-ws-f21jf.s3pd02.sbercloud.ru/b-ws-f21jf-ny6/FBC2/sample_test.zip).

# Integral Metric

The final grade of a multitask model for the FusionBrain Challenge 2.0 task shall be composed of quality metric values for individual sub-tasks. 

$$S = \frac{1}{15}\sum_{j \in \\ [open_t]}{S_{j}} + \frac{1}{10}\sum_{k \in [hidden_t]}{S_{k}},$$

where $[open_t] = \\{ TextQA, MathQA, ImageGeneration, ImageCaptioning, VisualQA, TRitW \\}$ is a set of open sub-tasks, $[hidden_t] = \\{ Hidden1,\ Hidden2,\ Hidden3,\ Hidden4,\ Hidden5,\ Hidden6 \\}$ is a set of hidden sub-tasks.

If the particular participant gets the highest score for each individual sub-task, the final value of the $S$ metric will be the maximum among all the participants.

To obtain the final value $S$ on Test private data, the particular participant should choose three submissions that will be launched on Test private data. The best value of $S$ is saved as a final metric. If the Participant does not choose three submisions to launch on Test private, the system will automatically proceed with three best submissions on Public test.

Solutions that take 1-3 places in the leaderboard based on Test private data by calculating the value of the integral metric $S$ will be considered the winners of the competition. Please note that there is a lower threshold determined for the open tasks (each threshold is defined as 75% of the corresponding baseline’s result):

* TextQA: 0.2
* MathQA: 0.25
* ImageGeneration: 0.21
* ImageCaptioning: 0.15
* VisualQA: 0.24
* TRitW: 0.13

In order to take 1-3 places, a Participant should get the values of the corresponding metrics higher or equal to their lower thresholds.

# Prize money

## Main categories for multitask model creation**

**First place:** RUB 1,000,000  
**Second place:** RUB 700,000   
**Third place:** RUB 400,000

## Special category
**The Most Sustainable Solution:** RUB 100,000

## Special category "The Most Sustainable Solution"

### Category description 

As part of the task to be solved, a special category is introduced, which is called "The Most Sustainable Solution". Its key goal is to motivate the competition participants to search for elegant and unconventional solutions that are optimal in terms of computing. The key metric in this category is the amount of indirect CO2 emission while training a model presented by the participant as a solution. The metric should be calculated using the [eco2AI](https://github.com/sb-ai-lab/Eco2AI). 

To take part in the category, it is necessary to meet the following conditions:
+ The solution entered the top in terms of the key metric (for more details see below)
+ The participant submitted the data (attached it to the archive when uploading the solution to the platform) on CO2 emissions during the model training (the </your_filename/>.scv file calculated using eco2aAI)
+ The participant sent a free form presentation describing the structure of the respective model and the process of its training, including a part about the CO2 calculation and selection of computing resources, to AIJContest_2022_org@sberbank.ru. The goal of this presentation is to show how the participant obtained such results and to make sure that they were obtained in good faith. The presentation should be sent before 11 November 2022.

An additional advantage for the participant when checking the presentation may be represented by two diagrams: a CO2 increase diagram for each training epoch and a diagram of model loss dependence on emitted CO2. For an example see "eco2AI: carbon emissions tracking of machine learning models as the first step towards sustainable AI". The following functions are implemented in the library to make the process of graphs drawing more convenient: start_training .new_epoch  .stop_training. See the [colab notebook](https://github.com/sb-ai-lab/Eco2AI#3) for more detail.

Below you may find the technical information that the participants would need to use eco2AI during the contest:
+ specify a name of the task to be solved in the "project_name" parameter of the Tracker method
+ specify the brief model description (the number of parameters) in the "experiment_description" parameter of the Tracker method
+ the tracker.start() method should be used in the very beginning of the executable code, right after the import of libraries. the tracker.stop() method should be used at the very end of the executable code
+ if the final training stage is conducted, specify the key phrase “TargetValue” in the "experiment_description" field

Since the participants may train their models many times, we shall take the last value of "CO_2 emissions" from the emission.csv file, with the key phrase “TargetValue” in the "experiment_description" column, as a reply in the category. 

The organizers reserve the right to deny participation in "The Most Sustainable Solution" category if the above instructions are not complied with or if it is found that the data is falsified. If several solutions have the same CO2 metric, the winner will be determined by the commission decision based on submitted materials.

### Description of "The Most Sustainable Solution" category metric
The category metric is represented by the "CO_2 emissions" value – the indirect CO2 emission while training a model. The calculation will be based on the following formula:

$$ CO_2 =(E_{CPU}+E_{GPU})*CO_{2coef}$$

where $E_{CPU}$ is the the energy consumed by CPU, $E_{GPU}$ is the the energy consumed by GPU, $CO_{2coef}$ is the regional CO2 emission coefficient. For detailed description of the calculation methodology see "eco2AI: carbon emissions tracking of machine learning models as the first step towards sustainable AI".

Therefore, the CO2 solution efficiency consists of three components:

1. solution efficiency (has an impact on training time)
2. energy efficiency of GPU and CPU devices 
3. region where computing resources are taken from

For values of emission and device power coefficients see https://github.com/sb-ai-lab/Eco2AI. To conduct calculations with lower regional emission coefficient, you may use Google Colab or find some other services. Please describe your search in the presentation.

### Announcement of results and distribution of awards in "The Most Sustainable Solution" category
The winner in this category will be a solution that showed the lowest value of "CO_2  emissions" in the course of model training and entered the top of solutions, provided that the respective participants followed all the instructions from the "Category description". The top shall include the solutions exceeding the threshold value  $nomination$ in terms of the key metric based on the following formula:

$$nomination=mean(Main_{metrics} )+std( Main_{metrics} )$$

<!-- nomination=mean(Main_{metrics} )-std( Main_{metrics} )$$ -->

where $mean(Main_{metrics} )$ is the mean value of the value array for metrics that exceeded the baseline metric for the task, $std( Main_{metrics} )$  is a standard deviation of the value array for metrics that exceeded the baseline metric for the task.

The winner in "The Most Sustainable Solution" category will get a prize in the amount of RUB 100,000.

After the competition, the winners in the additional category will have to defend their solution in the presentation format, where the participants will have to describe the process of their model training and computing resources used for that.
