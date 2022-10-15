# Evaluation metrics


## Sub-task 1 – Text QA

To evaluate the quality of textual tasks in the "question-answer" form, we use the **F1** metric.
The **F1** metric shall be calculated on the basis of all tokens in predicted (after post-processing of the generated text) and true answers and shall express the harmonic mean of precision and recall of predicted answers.
Standard formula for the metric: 

![image](https://latex.codecogs.com/svg.image?S_{TextQA}=F1&space;=&space;2\cdot&space;\frac{Recall&space;\cdot&space;Precision}{Recall&space;&plus;&space;Precision})

There and further through the text $S_i^{[o,h]}$ refers the metric for _i-th_ open (_o_) or hidden (_h_) task respectively.

For question answering tasks: 

![image](https://latex.codecogs.com/svg.image?\text{precision}&space;=&space;\frac{\char"0023&space;tokens_{common}}{\char"0023tokens_{predicted}},&space;\text{recall}&space;=&space;\frac{\char"0023&space;tokens_{common}}{\char"0023&space;tokens_{gt}},)

where ![image](https://latex.codecogs.com/svg.image?\char"0023&space;tokens_{common}) is the number of common tokens for the predicted and the true answer;
![image](https://latex.codecogs.com/svg.image?\char"0023&space;tokens_{predicted}) is the number of tokens in the predicted answer;
![image](https://latex.codecogs.com/svg.image?\char"0023&space;tokens_{gt}) is the number of tokens in the true answer.

## Sub-task 2 – Mathematical QA

To evaluate the quality of mathematical tasks in the "question-answer" form, we use the **Exact Match (EM)** metric $(S_{MathQA})$.

This metric shall receive the value of 1, if all tokens in the predicted answer (after post-processing of the generated text) are in complete accord with tokens in the true answer; otherwise the metric value shall be *EM = 0*. 

This metric is used for multiple choice mathematical tasks (MathQA), where the answer may be represented by only a letter corresponding to the correct answer option, or a number corresponding to the correct solution of the mathematical task.


## Sub-task 3 – Image Generation in a Wide Domain

To evaluate the quality of text-to-image tasks solution, we use two key metrics: **FID** and **CLIP**. These metrics are widely used in the tasks for image generation based on text.

**FID** is the Frechet distance between two multidimensional Gaussian distributions: $N(\mu,\Sigma)$ – a distribution of features for images created by the generator, and $N(\mu_{\omega},\Sigma_{\omega})$ – a distribution of features for real images used to train the network. As a neural network for getting features, we usually use Inception v3 trained on ImageNet. As a result, the metric may be calculated on the basis of mean values and co-variation of activations, when synthesized and real images are fed to the Inception network.

![image](https://latex.codecogs.com/svg.image?\text{FID}=|\mu&space;-&space;\mu_\omega|^{2}&plus;tr(\Sigma&space;&plus;&space;\Sigma_\omega-2(\Sigma\Sigma_\omega)^{\frac{1}{2}}).)

Instead of direct comparison of images, pixel by pixel, FID compares the mean value and the standard deviation for one of the deepest layers in the convolutional neural network. These layers are closer to output nodes corresponding to real world objects such as a certain dog breed or an airplane, and further from the shallow layers near the input image. As a result, they have a tendency to imitate the human perception of image similarity. The metric value is better, the closer it to 0; at the same time, the metric is unnormalized and values are very dependent on the size of sample used to measure it.
 
**CLIP score** is a metric that allows evaluating how much the visual presentation corresponds to the textual description. To calculate the metric, we use the neural network CLIP returning textual and visual embeddings for each "picture-text" pair: The obtained representations shall be compared on the basis of their cosine similarity. Cosine similarity reflects the measure of similarity between two vectors and is calculated by the following formula:

![image](https://latex.codecogs.com/svg.image?\text{similarity}=\frac{x_1&space;x_2}{max(\left\|&space;x_1\right\|_2&space;\boldsymbol{\cdot}&space;\left\|&space;x_2\right\|_2,\epsilon)},)

where $x_1$ is the textual embedding, $x_2$ is the picture embedding, $\epsilon$ is the arbitrarily small positive number entered to avoid the division by 0.

Cosine similarity equals to 1, if the corresponding textual and visual vector representations match, and to 0, if they are completely different. The common value of CLIP score shall be calculated as an averaged value of the metric calculated on the basis of all testing examples for this task.

The final metric for text-to-image task is a combination of two metrics - **FID** and **CLIP score**. It is calculated by the following formula:

![image](https://latex.codecogs.com/svg.image?S_{ImageGeneration}=\frac{1}{2}\left(CLIP_{score}&plus;\frac{200-min(200,X)}{200}\right))

## Sub-task 4 – Image Captioning

For the task to evaluate the quality of generated textual descriptions of images, we use such metrics as **METEOR** and **CLIP score**. 

**METEOR** is a metric based on the analysis of n-grams and focused on statistical and precise evaluation of initial texts. This metric uses the synonym comparison functions together with the exact match of words. 

First, the algorithm aligns the text between two sentences - a reference translation string and an input text string for evaluation. After that, it uses several stages of matching between machine and reference translation words to compare two strings:
1. Precise matching - identifying identical strings in reference and machine translations.
2. Matching of stems - stemming and identifying words with the same roots in reference and machine translations.
3. Matching of synonyms - identifying words being synonyms in accordance with RuWordNet.

Alignment is the multitude of matches between n-grams. Match has the following limitation: each n-gram in a candidate sentence should correspond to one or none n-gram in the reference sentence. If there are two alignments with the same number of matches, it is necessary to choose the one with the least number of intersections for matches. The stages of comparison with reference translations shall be conducted in succession, and at each of them only those n-grams that did not have matches at previous stages shall be added to the multitude of matches. After the last stage, the final value of precision for n-grams shall be calculated by the following formula:

![image](https://latex.codecogs.com/svg.image?\text{P}=\frac{m}{w_t},)

where $m$ is the number of n-grams in the machine translation that were found in the reference one, $w_t$ is the number of n-grams in the machine translation. 

The recall value for n-grams (common n-gram for reference translations) shall be calculated by the following formula:

![image](https://latex.codecogs.com/svg.image?\text{R}=\frac{m}{w_r},)

where $w_r$ is the number of n-grams in the reference translation.

As a result, METEOR shall be calculated as a combination of precision and recall, with the use of the harmonic mean formula, where the recall weight is 9 times more than the precision weight:

![image](https://latex.codecogs.com/svg.image?\text{METEOR}=\frac{10PR}{R&plus;9P}.)

The common value of the METEOR metric shall be calculated as its averaged value calculated on the basis of all testing examples for this task.

**CLIP score** is a metric that allows evaluating how much the textual description corresponds to the visual presentation. The metric shall be calculated in the same way as the metric used in the image generation task. 

The final metric for Image Captioning tasks is calculated as an average of the METEOR and CLIP score.

![image](https://latex.codecogs.com/svg.image?S_{ImageCaptioning}&space;=\frac{1}{2}&space;\cdot&space;(METEOR&plus;CLIP_{score})&space;)

## Sub-task 5 – Visual QA

For the VisualQA task, we use such metrics as **METEOR**. METEOR shall be calculated in the same way as the metric described in the image captioning task. However, there are some modifications:
1. It is taken into account, in which proportion the predicted numerical result differs from the real one. With this end in view, the smallest number shall be selected from a couple of predicted and reference results and divided by the biggest one. Therefore, if the numbers match, the metric for this pair equals to 1; otherwise, the metric shall be calculated proportionally.
2. Numerals shall be translated from the text format to the numeric one: "three" - 3.

![image](https://latex.codecogs.com/svg.image?S_{VisualQA}&space;=&space;METEOR.)

## Sub-task 6 – Text Recognition in the Wild

As the key metric to evaluate the participants' solutions, we use the normalized metric **1 - NED** (_NED_ - _Normalized Edit Disctance_) metric. It shall be calculated as follows:

![image](https://latex.codecogs.com/svg.image?\bg{blue}S_{TRitW}&space;=&space;1&space;-&space;NED&space;=&space;1&space;-&space;\frac{D(s_i,&space;\hat{s}_i)}{max(l_i,\hat{l}_i)},)

where $D(\cdot)$ indicates a Levenshtein distance between the predicted text $s_i$ and the ground truth one $\hat{s}_i$; $l_i$ and $\hat{l}_i$ denote the length of $s_i$ and $\hat{s}_i$, respectively.

The metrics values for each open task varies from 0 to 1, where 0 is the worst value and 1 is the best one.

## Hidden Tasks Metrics

For each hidden sub-task the corresponding hidden metric $S_i^{h}$ is calculated. $S_k \in [0,1]$, where 0 is the worst, and 1 is the best metric's value, $k \in \\{ Hidden1,\ Hidden2,\ Hidden3,\ Hidden4,\ Hidden5,\ Hidden6 \\}$.
