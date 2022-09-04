import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
import json
import random
import heapq
from scipy.special import entr
import operator

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup,T5Tokenizer, T5ForConditionalGeneration
from models import build_or_load_gen_model
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist


def generate_entr(logits):
    entropies = entr(logits.detach().numpy()).sum(axis=1)
    mean_entropy = np.mean(entropies)
    return mean_entropy

def reformulate(sentences, model, tokenizer):
    origin = []
    entr_target = []
    for idx, sentence in enumerate(sentences):
        sentence_split = sentence.split()
        sentences_with_mask = []
        entropies = []
        output_labels = []
        if sentence_split[-1][-1] == "?":
            sentence_split[-1] = sentence_split[-1][:-1]
        for i in range(1, len(sentence_split)+1):
            temp_split = list(sentence_split)
            temp_split.insert(i, "<extra_id_0>")
            sentences_with_mask.append(temp_split)
            new_sentence = " ".join(temp_split)
            input_ids = tokenizer(new_sentence, return_tensors="pt").input_ids  # Batch size 1
            outputs = model.generate(input_ids)
            output_labels.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
            label = '<extra_id_0> ' + tokenizer.decode(outputs[0], skip_special_tokens=True) + ' <extra_id_1>'
            labels = tokenizer(label, return_tensors='pt').input_ids
            outputs = model(input_ids=input_ids, labels=labels)
            mean_entropy = generate_entr(F.softmax(outputs.logits[0], dim=1))
            entropies.append(mean_entropy)
        max_entropy_index = np.argmax(entropies, axis=0)
        target_entr_sentence_split = sentences_with_mask[max_entropy_index]
        target_entr_sentence_split = [output_labels[max_entropy_index] if i == "<extra_id_0>" else i for i in
                                      target_entr_sentence_split]
        target_entr_sentence = " ".join(target_entr_sentence_split)
        origin.append(sentence)
        entr_target.append(target_entr_sentence)
    return origin, entr_target

def reformulate_json(sentences, model, tokenizer, entr_filename, res_num):
    idx = 0
    with open(entr_filename, 'w') as fe:
        for key in sentences:
            sentence = ' '.join(sentences[key]['docstring_tokens'])
            sentence_split = sentence.split()
            if len(sentence_split) < res_num:
                continue
            idx += 1
            sentences_with_mask = []
            entropies = []
            output_labels = []
            for i in range(1, len(sentence_split)+1):
                temp_split = list(sentence_split)
                temp_split.insert(i, "<extra_id_0>")
                sentences_with_mask.append(temp_split)
                new_sentence = " ".join(temp_split)
                input_ids = tokenizer(new_sentence, return_tensors="pt").input_ids  # Batch size 1
                outputs = model.generate(input_ids)
                output_labels.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
                label = '<extra_id_0> ' + tokenizer.decode(outputs[0], skip_special_tokens=True) + ' <extra_id_1>'
                labels = tokenizer(label, return_tensors='pt').input_ids
                outputs = model(input_ids=input_ids, labels=labels)
                mean_entropy = generate_entr(F.softmax(outputs.logits[0], dim=1))
                entropies.append(mean_entropy)
            entropies = np.array(entropies)
            min_entropy_index = np.argsort(entropies)[:res_num]
            target_entr_sentence_split = np.array(sentences_with_mask)[min_entropy_index]
            entr_output_labels = np.array(output_labels)[min_entropy_index]
            target_entr_res = []
            for entr_idx in range(len(target_entr_sentence_split)):
                curr_sentence = []
                for word in target_entr_sentence_split[entr_idx]:
                    if word != "<extra_id_0>":
                        curr_sentence.append(word)
                    else:
                        curr_sentence.append(entr_output_labels[entr_idx])
                target_entr_res.append(curr_sentence)
            target_entr_sentence = []
            for sen in target_entr_res:
                target_entr_sentence.append(" ".join(sen))
            for sen in target_entr_sentence:
                sentences[key]['docstring_tokens'] = sen.split()
                fe.write(json.dumps(sentences[key]) + '\n')

def load_data(input_path, output_path, model, tokenizer):
    records = []
    with open(input_path) as f:
        for line in f:
            js = json.loads(line.strip())
            records.append(js['sentence'])
    f.close()
    with open(output_path, 'w') as f1:
        origin, target = reformulate(records, model, tokenizer)
        for i in range(len(origin)):
            f1.write(origin[i] + ", " + target[i] + "\n")
    f1.close()

def read_json(filename):
    data = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            data[js['url']] = js
    return data


def count_len(sentences):
    len_count = {}
    for key in sentences:
        sentence = ' '.join(sentences[key]['docstring_tokens'])
        sentence_split = sentence.split()
        if len(sentence_split) not in len_count:
            len_count[len(sentence_split)] = 0
        len_count[len(sentence_split)] += 1
    len_count = sorted(len_count.items(), key=lambda x:x[0])
    print(len_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="The input path of the model checkpoint dir")
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="The input path of the code search dataset")
    parser.add_argument("--output", default=None, type=str, required=True,
                        help="The output path of result file")
    args = parser.parse_args()
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    # model = T5ForConditionalGeneration.from_pretrained('./pretrain/t5_pretraining/best_tfmr')
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    data = read_json(args.dataset)
    res_num = 3
    entr_filename = args.ouput
    reformulate_json(data, model, tokenizer, entr_filename, res_num)

if __name__ == "__main__":
    main()