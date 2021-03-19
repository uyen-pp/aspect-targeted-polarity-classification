# -*- coding: utf-8 -*-
"""Thesis_phoBERTbase_aspect_regconition_v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HZIfYY8F-p_o32VfjHjCLHeNFagEuxtS

This is the first model for the first task of my thesis. The model is for Multi-label aspect classification using phoBert pretrained model for Vietnamese text.

We add a Dropout followed by a Fully connected layer for the purpose of Regularization and classification. The output of FC is (21,) tensor for 21 Binary Multilabel Encoding. The input vector used for classification is special vector as 1st position of phoBERT encoding (known as _cls)

The loss function used will be a combination of Binary Cross Entropy which is implemented as BCELogits Loss in PyTorch with Adam optimizer.
"""

# from google.colab import drive
# drive.mount('/gdrive')

import os


# YN Data proposed
YN_DATA_HOME = "C:\\Users\\Uyen\\Documents\\nlp\\YNdata"

# Clean YN Data dataframe pickle files
YN_CLEAN = os.path.join(YN_DATA_HOME, "splitted")

# YN Data transfrom to xml to be run_glue adaptable, for training
YN_AR = os.path.join(YN_DATA_HOME, "ar_dataset")

YN_ATSC = os.path.join(YN_DATA_HOME, "atsc_dataset")

# YN Data transfrom to xml to be run_glue adaptable, for testing
YN_TEST = os.path.join(YN_DATA_HOME, "test")

# Aspect regconition model
MODEL_DIR = "./AR_Model"

"""# 1. Installation"""

# Import numpy and pandas

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


import logging
import random
import sys
from typing import Optional

import numpy as np
import math

from datasets import load_dataset, load_metric
from dataclasses import dataclass, field
from sklearn import metrics

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AdamW,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
    HfArgumentParser,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from utils_glue import (
    compute_metrics, 
    convert_examples_to_features, 
    output_modes, 
    processors)

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

def validation(model, data_loader):
    
    model.eval()
    fin_targets=[]
    fin_predictions=[]
    for _, batch in enumerate(data_loader):
        input_ids, targets = batch['input_ids'].to(device), batch['labels'].to(device)
        outputs = model(input_ids)
        get_sigmoid = torch.nn.Sigmoid()
        predictions = (get_sigmoid(outputs.logits) >= 0.5)*1

        fin_targets.extend([int(l) for l in targets.cpu().detach().numpy().tolist()[0]])
        fin_predictions.extend(predictions.cpu().detach().numpy().tolist()[0])

    accuracy = metrics.accuracy_score(fin_targets, fin_predictions)
    logger.info(f"Accuracy Score = {accuracy}")
    f1_score_micro = metrics.f1_score(fin_targets, fin_predictions, average='micro')
    logger.info(f"F1 Score (Micro) = {f1_score_micro}")
    f1_score_macro = metrics.f1_score(fin_targets, fin_predictions, average='macro')
    logger.info(f"F1 Score (Macro) = {f1_score_macro}")
    f1_score = metrics.f1_score(fin_targets, fin_predictions, average=None)
    logger.info(f"F1 Score = {f1_score}")
    recall_score = metrics.recall_score(fin_targets, fin_predictions, average=None)
    logger.info(f"Recall Score = {recall_score}")
    precision = metrics.precision_score(fin_targets, fin_predictions, average=None)
    logger.info(f"Precision Score = {precision}")

    """# 2. Setup model arguments, data arguments, training arguments"""

def train(model, data_loader, last_checkpoint=None):
    # Optimizer: 
    if last_checkpoint is not None:
        optim = torch.load(os.path.join(last_checkpoint, "optimizer.pt"))
        optimizer.load_state_dict(optim)
    else:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Loss function 
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_steps), disable=False)
    completed_steps = 0
    logging_steps = training_args.logging_steps
    save_steps = training_args.save_steps

    for epoch in range(training_args.num_train_epochs):
        model.train()
        
        for step, batch in enumerate(data_loader):
            outputs = model(batch["input_ids"].to(device))
            logits = outputs["logits"]
            targets = batch["labels"].to(device)
            loss = loss_fn(logits.view(-1), targets.view(-1))
            if step%logging_steps==0:
                logger.info(f'Epoch: {epoch}, Step: {step}, Loss:  {loss}')
            if step%save_steps==0 and step!=0:
                save_checkpoint(model, optimizer, step) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.update(1)

# Data arguments
device = torch.device("cuda")
max_sequence_length = 256
task_name = "ar"
train_file = os.path.join(YN_AR, "train.csv")
validation_file = os.path.join(YN_AR, "dev.csv") 
load_dataset_script = "C:\\Users\\Uyen\\Documents\\nlp\\thesis\\aspect-targeted-polarity-classification\\finetuning_and_classification\\load_dataset_ar.py"
padding = "max_length"

# Training arguments
training_args = TrainingArguments(
    output_dir = MODEL_DIR,
    overwrite_output_dir = True,
    do_eval = True,
    seed = 282,
    warmup_steps = 100,
    num_train_epochs=1,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    logging_steps = 10,
    save_steps = 200
)

# Model Arguments
model_args = ModelArguments(
    model_name_or_path=MODEL_DIR,
    use_fast_tokenizer=False
)


def main():

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")

    set_seed(training_args.seed)
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    """# 3. Load data"""

    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_sequence_length, truncation=True)
        result["labels"] = examples["labels"]
        return result

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    
    # Preprocessing the datasets
    sentence1_key, sentence2_key = ("sentence", None)

    data_files = {"train": train_file, "validation": validation_file}

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    train_dataset, eval_dataset = None, None
    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    data_collator = default_data_collator

    datasets = load_dataset(path=load_dataset_script, data_files=data_files)

    if training_args.do_train:
        train_dataset = load_dataset(path=load_dataset_script, data_dir=YN_AR, split="train")
        num_labels = train_dataset.num_labels
        train_dataset = train_dataset.map(preprocess_function, batched=True, load_from_cache_file=True)
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size = training_args.per_device_train_batch_size)

    if training_args.do_eval:
        eval_dataset = load_dataset(path=load_dataset_script, data_dir=YN_AR, split="validation")
        eval_dataset = eval_dataset.map(preprocess_function, batched=True, load_from_cache_file=True)
        # Log a few random samples from the validation set:
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Examples: {eval_dataset[index]}.")
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size)

    
    """# 4. Load Model"""
    num_labels = (
        train_dataset.features['labels'].length if train_dataset is not None 
        else eval_dataset.features['labels'].length)

    # Detecting last checkpoint.
    def get_last_checkpoint(folder):
        content = os.listdir(folder)
        checkpoints = [
            path
            for path in content
            if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
        ]
        print(checkpoints)
        if len(checkpoints) == 0:
            return
        return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))

    def save_checkpoint(model, optimizer, step):
        # Save model checkpoint
        checkpoint_folder = os.path.join(MODEL_DIR, f"checkpoint-{step}")

        model.save_pretrained(checkpoint_folder)
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_folder, "optimizer.pt"))

    import re
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

    last_checkpoint = None
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        logger.info(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

    config = AutoConfig.from_pretrained(
        last_checkpoint if last_checkpoint is not None else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        last_checkpoint if last_checkpoint is not None else model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )

    model.to(device)

    """# 5. Train"""
    if training_args.do_train:
        total_batch_size = training_args.per_device_train_batch_size * training_args.n_gpu

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps = {training_args.max_steps}")
        logger.info(f"  Save after steps = {training_args.save_steps}")
        logger.info(f"  Log after steps = {training_args.logging_steps}")
        logger.info(f"  Training device = {model.device}")

        train(model, last_checkpoint, train_dataloader)
        

    """# 6. Evaluation"""
    if training_args.do_eval:
        logger.info("***** Running evaluate *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_eval_batch_size}")
        logger.info(f"  Device = {model.device}")

        validation(model, eval_dataloader)

if __name__ == "__main__":
    main()