# Pretraining T5
Continually pretraining T5 on the code search dataset.

Pretrained models from [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) are made available from [Huggingface](https://huggingface.co/transformers/model_doc/t5.html). This is the code for continually pretraining T5 on the code search dataset. This code follows the same unsupervised pretraining objective followed by the original paper. Details of the T5 style pretraining can be found in the [paper](https://arxiv.org/abs/1910.10683).

### Dataset

The dataset can be found at

https://drive.google.com/file/d/1CdeKJY2vY1P0jzjoDk1hjJ2ZLDhq5hWg/view?usp=sharing

### Code

* **pretrain.py:** the code for continually pre-training T5 on the code search dataset

### Run

In order to run the code, first install the packages from requirements.txt 

~~~
pip install -r requirements.txt
~~~
You also have to install torch that is compatible with your CUDA version from (https://pytorch.org/)

Run the code with the following parameters setting:
~~~
python pretrain.py --input_length 128 --output_length 128 --num_train_epochs 3 --output_dir t5_pretraining --train_batch_size 32 --learning_rate 1e-3 --model t5-base
~~~

