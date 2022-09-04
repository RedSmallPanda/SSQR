# Query Reformulation

## Task Definition

Given a natural language query, the query reformulation task is to reformulate the query to make it more accurate.

### Code

  - **code/generate.py:** the code for reformulating a given code search query

### Run

Reformulate the queries in code search dataset:

```shell
cd code
python generate.py \
    --model=$model_checkpoint_dir$ \
    --output=$your_output_dir$ \
    --dataset=$the_code_search_dataset$ \
```



# Code Search

### Code

  - **code/model.py:** the model structure of CodeBERT
  - **code/run.py:** code for fine-tuning and running CodeBERT
  - **evaluator/evaluator.py:** code for evaluating the code search performance

### Data Download and Preprocess

```shell
unzip dataset.zip
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
unzip python.zip
python preprocess.py
rm -r python
rm -r *.pkl
rm python.zip
cd ..
```

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  251,820  |
| Test  |  19,210   |

## Finetune and Use CodeBERT as Search Engine

We also provide a pipeline that fine-tunes [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) on this task. 

### Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0


### Fine-tune

To fine-tune encoder-decoder on the dataset

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 2 \
    --block_size 256 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log
```


### Run

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 2 \
    --block_size 256 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee test.log
```

### Evaluate

```shell
python ../evaluator/evaluator.py -a ../dataset/test.jsonl  -p saved_models/predictions.jsonl 
```



## Results

The results on the test set for code search performance and human evaluation of different query reformulation approaches are shown as below:

| Method   |    MRR     | Naturalness | Informativeness |
| -------- | :--------: | ----------- | --------------- |
| Original |   0.2021   | 3.21        | 3.15            |
| SEQUER   |   0.2220   | 3.63        | 3.44            |
| LuSearch |   0.1441   | 2.63        | 3.17            |
| NLP2API  |   0.1478   | 2.80        | 3.50            |
| **SSQR** | **0.2221** | **3.83**    | **3.98**        |

