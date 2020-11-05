# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run GPT2 small on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open
import pickle


import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from gpt2sqa.file_utils import PYTORCH_PRETRAINED_GPT2_CACHE, WEIGHTS_NAME, CONFIG_NAME
from gpt2sqa.modeling_gpt2 import GPT2ModelForQuestionAnswering
from transformers.optimization import AdamW
from transformers import GPT2Tokenizer
from gpt2sqa.squad.squad_example import InputFeatures
from gpt2sqa.squad.utils import convert_examples_to_features, read_squad_examples, get_final_text, write_predictions, _check_is_max_context, _get_best_indexes, _compute_softmax, RawResult
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_name", default="ruGPT3Medium",
                        type=str, help="ruGPT3Small or ruGPT3Medium or ruGPT3Large")
    parser.add_argument("--with_negative", default=False,
                        type=str, help="version 2 with negative or not")
    # Other parameters
    parser.add_argument("--train_file", default=None, type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=1000, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    args = parser.parse_args()
    print(args)
    os.environ['XLA_USE_BF16'] = '1'
    os.environ['TRIM_GRAPH_SIZE'] = '10000000'

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = GPT2Tokenizer.from_pretrained(
        "sberbank-ai/rugpt3large_based_on_gpt2")

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = read_squad_examples(
            input_file=args.train_file, is_training=True, version_2_with_negative=args.with_negative)
        num_train_optimization_steps = int(len(
            train_examples) / args.train_batch_size / args.gradient_accumulation_steps / xm.xrt_world_size()) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = GPT2ModelForQuestionAnswering.from_pretrained(
        cache_dir=PYTORCH_PRETRAINED_GPT2_CACHE, pretrained_model_name_or_path=args.model_name)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    
    if args.do_train:
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True)
        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor(
            [f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor(
            [f.end_position for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)

        WRAPPED_MODEL = xmp.MpModelWrapper(model)

        def train_fn(index, train_data):
            device = xm.xla_device()
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True)
            train_dataloader = DataLoader(
                train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=1, drop_last=True)
            global_step = 0
            losses = []
            w_model = WRAPPED_MODEL.to(device)

            # Prepare optimizer
            param_optimizer = list(w_model.named_parameters())

            # hack to remove pooler, which is not used
            # thus it produce None grad that break apex
            param_optimizer = [
                n for n in param_optimizer if 'pooler' not in n[0]]

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.learning_rate * xm.xrt_world_size())
            
            w_model.train()
            for epoch in range(int(args.num_train_epochs)):
                if xm.is_master_ordinal():
                    print(f"\n\nEpoch n.{epoch + 1}/{args.num_train_epochs} started")
                device_loader = pl.ParallelLoader(train_dataloader, [device]).per_device_loader(device)
                pbar = tqdm(device_loader, disable=not xm.is_master_ordinal(), desc="Epoch progress", position=0, leave=True)
                for step, batch in enumerate(device_loader):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    loss = w_model(input_ids, segment_ids, input_mask,
                                start_positions, end_positions)
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    losses.append(loss.item())
                    # print(f"loss={loss.item()}")
                    optimizer.zero_grad()
                    loss.backward()
                    pbar.update(1)
                    if step % 10 == 0 and xm.is_master_ordinal() and step:
                        pbar.set_description(desc=f'loss : {np.mean(losses[10:])}')
                    if xm.is_master_ordinal() and ((step + 1) % 1000 == 0) or step + 1 == len(device_loader):
                        print(f"\navg loss for 1000 steps = {np.mean(losses)}")
                        losses = []
                    xm.optimizer_step(optimizer)
                    global_step += 1
                xm.rendezvous('save_model')
                w_model.to("cpu")
                if xm.is_master_ordinal():
                    print("\nSaving model...")
                    print(w_model == model)
                    # torch_xla.utils.serialization.save(model, args.output_dir, master_only=True, global_master=True)
                    # Save a trained model, configuration and tokenizer
                    model_to_save = w_model.module if hasattr(
                        model, 'module') else w_model  # Only save the model it-self
                    
                    # If we save using the predefined names, we can load using `from_pretrained`
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                    # xm.save(model, output_model_file, master_only=True, global_master=False)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(args.output_dir)
        # train_fn(device_loader)
        xmp.spawn(train_fn, args=(train_data, ), nprocs=1, start_method='fork')
        

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        # Load a trained model and vocabulary that you have fine-tuned
        model = GPT2ModelForQuestionAnswering.from_pretrained(args.output_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(args.output_dir)
    else:
        model = GPT2ModelForQuestionAnswering.from_pretrained(
            cache_dir=PYTORCH_PRETRAINED_GPT2_CACHE, pretrained_model_name_or_path=args.model_name)
    device = xm.xla_device()
    model.to(device)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_squad_examples(
            input_file=args.predict_file, is_training=False, )
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False,)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(
            all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0], position=0, leave=True):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits = model(
                    input_ids, segment_ids, input_mask)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))
        output_prediction_file = os.path.join(
            args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(
            args.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(
            args.output_dir, "null_odds.json")
        write_predictions(eval_examples, eval_features, all_results,
                          args.n_best_size, args.max_answer_length,
                          args.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                          True, args.null_score_diff_threshold)


if __name__ == "__main__":
    main()
