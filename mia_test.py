import os
import sys
import json
import glob
import math
import torch
import random
import shutil
import argparse
import datasets
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from collections import defaultdict
from statistics import mean, harmonic_mean
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def mia_attacks(args, model, tokenizer):
    mia_data_path = '/home/amunis/Unlearning-sensitive-content-from-LLMs/new_evaluation/mia_test2/'
    member_file =  mia_data_path+ 'member.jsonl'
    nonmember_file = mia_data_path + 'nonmember.jsonl'
    losses = []
    accelerator = Accelerator()
    model.to(accelerator.device)

    for dataset, train_file in [('member', member_file), ('nonmember', nonmember_file)]:
        data_files = {}
        dataset_args = {}
        if train_file is not None:
            data_files["train"] = train_file
        raw_datasets = datasets.load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )
        train_dataset = raw_datasets["train"]

        output_dic = defaultdict(lambda :{'id': [], 'nll': []})

        with accelerator.split_between_processes(train_dataset, apply_padding=True) as data:
            for idx in tqdm(range(len(data['document']))):
                document = data["document"][idx]
                output_dic[accelerator.process_index]['id'].append(data["id"][idx])
                input_ids = tokenizer(
                    document,
                    return_tensors='pt'
                ).input_ids.to(model.device)

                target_ids = input_ids.clone()
                
                with torch.no_grad():
                    out = model(input_ids, labels=target_ids)
                    print(out.loss)
                    neg_log_likelihood = out.loss.item()
                    output_dic[accelerator.process_index]['nll'].append(neg_log_likelihood)

        accelerator.wait_for_everyone()
        
        output_df = pd.DataFrame.from_dict(output_dic[accelerator.process_index])
        losses.append(output_df['nll'])
    print(losses)
    compute_auc(list(losses[0]), list(losses[1]))

def compute_auc(member_loss, nonmember_loss):
    assert not np.any(np.isnan(member_loss))
    assert not np.any(np.isnan(nonmember_loss))
    combined_loss = member_loss + nonmember_loss 
    combined_loss = -1 * np.array(combined_loss)
    combined_labels = len(member_loss) * [1] + len(nonmember_loss) * [0]
    fp, tp, _ = roc_curve(combined_labels, combined_loss)
    print('fp: ', fp)
    print('---------')
    print('tp: ', tp)

    auc_score = float(auc(fp, tp))
    print('auc_score: ', auc_score)
def main():

    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    checkpoint_path = '/home/amunis/Unlearning-sensitive-content-from-LLMs/claudio_ce_epoch_3'

    # Set up accelerator
    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16, trust_remote_code = True) # .to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    args=None
    mia_attacks(args,model,tokenizer)

if __name__ == "__main__":
    main()