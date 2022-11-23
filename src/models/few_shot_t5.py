from __future__ import absolute_import, division, print_function
import time
import os
import sys
import random
import math
import json
import logging

import numpy as np
import argparse
import mlflow
import spacy
import torch
import pprint
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, \
    get_linear_schedule_with_warmup, Adafactor

from utils import load_examples, generate_features_t5, \
    generate_dataset_t5, my_collate_t5, evaluate_all, evaluate_dev

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
pp = pprint.PrettyPrinter(indent=4)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S %Z')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UN OHCHR Detection')
    parser.add_argument('--hidden_dim', type=int, default=160,
                        help='hidden dimention of lstm')
    parser.add_argument('--lower_case', action='store_true', help='lower case')
    parser.add_argument('--train_file',
                        default='',
                        help='path to training bio file')
    parser.add_argument('--dev_file',
                        default='',
                        help='path to dev bio file')
    parser.add_argument('--test_file',
                        default='',
                        help='path to test bio file')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--checkpoint_name', default='t5_un',
                        help='name of checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gup id, set to -1 if use cpu mode')
    parser.add_argument('--lr', type=float, default=0.015,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05,
                        help='decay ratio of learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for sgd')
    parser.add_argument('--epoch', type=int, default=20,
                        help='number of epoches')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='grad clip at')
    parser.add_argument('--output_dir', default='tmp/t5_un',
                        help='path of output dir')
    parser.add_argument("--model_dir", default="t5_un/epoch0-step0", type=str,
                        help="eval/test model")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate "
                             "before performing a backward/update pass.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear "
                             "learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on per epoch")
    parser.add_argument("--eval_metric", default='average', type=str)
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization, "
                             "set it as None if no seed is used")
    parser.add_argument('--add_prefix', action='store_true',
                        help='add prefix for each task')
    parser.add_argument('--use_metric', action='store_true',
                        help='use metric on dev for model selection, '
                             'otherwise train the model '
                             'with certain number of steps')
    parser.add_argument('--lower_input', action='store_true',
                        help='use uncased model if True')
    parser.add_argument('--model_name', default='basic parameters',
                        help='name of the run')
    parser.add_argument('--dataset_name', default='basic parameters',
                        help='name of the run')
    parser.add_argument('--context_filter', action='store_true',
                        help='filter context')
    parser.add_argument('--replicate', action='store_true',
                        help='replicate the reported scores')
    args = parser.parse_args()

    # set up run name for MLflow
    args.run_name = args.model_name + '-' + args.dataset_name

    # set seed
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu >= 0:
            torch.cuda.manual_seed_all(args.seed)

    # set gpu
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    # initialize model
    model_name = 't5-large'
    model_class = T5ForConditionalGeneration
    tokenizer_mame = T5Tokenizer
    config_name = T5Config
    config = config_name.from_pretrained(model_name)
    tokenizer = tokenizer_mame.from_pretrained(model_name)
    model = model_class.from_pretrained(
        model_name, cache_dir='./pre-trained-model-cache')

    logger.info(f'tokenizer.pad_token_id is {tokenizer.pad_token_id}')
    logger.info(f"Vocab size is {len(tokenizer.get_vocab())}")

    vocab = tokenizer.get_vocab()
    args.dropout = config.dropout_rate
    args.hidden_dim = config.d_model

    # load data
    logger.info(f"loading data ...")

    train_examples = load_examples(args.train_file)
    dev_examples = load_examples(args.dev_file)
    test_examples = load_examples(args.test_file, split_doc=True, max_len=600)

    # generate data loaders
    logger.info(f"generate data loaders ...")
    train_features = generate_features_t5(train_examples, tokenizer,
                                          add_prefix=args.add_prefix,
                                          max_len=512,
                                          context_filter=args.context_filter)
    dev_features = generate_features_t5(dev_examples, tokenizer,
                                        add_prefix=args.add_prefix,
                                        max_len=512,
                                        context_filter=args.context_filter)
    test_features = generate_features_t5(test_examples, tokenizer,
                                         add_prefix=args.add_prefix,
                                         max_len=512,
                                         context_filter=args.context_filter,
                                         split_doc=True, top_sentence=None,
                                         replicate=args.replicate)

    train_dataset = generate_dataset_t5(train_features)
    dev_dataset = generate_dataset_t5(dev_features)
    test_dataset = generate_dataset_t5(test_features)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=my_collate_t5)

    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=my_collate_t5)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=my_collate_t5)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu >= 0:
            torch.cuda.manual_seed_all(args.seed)

    # define optimizer
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': args.lr_decay},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    tot_len = len(train_dataloader)
    logger.info(f"There are {tot_len} train batches in total.")
    num_train_optimization_steps = math.ceil(
        tot_len / args.gradient_accumulation_steps) * args.epoch
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = Adafactor(
        optimizer_grouped_parameters,
        lr=args.lr,
        clip_threshold=args.clip_grad,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )

    # set up optimizer scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_optimization_steps)

    # compute the steps for applying evaluation on dev set
    eval_step = max(1, len(train_dataloader) // (
            args.gradient_accumulation_steps * args.eval_per_epoch))
    logger.info(f'Evaluation is applied on dev set every {eval_step} steps')

    if args.gpu >= 0:
        model.cuda()

    tr_loss = 0

    # counter for trained samples
    nb_tr_examples = 0

    # counter for trained steps
    nb_tr_steps = 0

    # counter for optimizer update
    global_step = 0

    start_time = time.time()
    best_result = None

    best_dev_f1 = float('-inf')
    saved_params = {"learning_rate": args.lr,
                    "batch_size": args.batch_size,
                    "epoch": args.epoch,
                    "lr_decay": args.lr_decay}
    model.train()

    optimizer.zero_grad()
    for epoch in range(args.epoch):
        epoch_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Start epoch #{epoch}/{args.epoch} (lr = {current_lr})...")

        for step, batch in enumerate(train_dataloader):
            batch_input_ids, batch_input_masks, \
            batch_output_ids, batch_feature_idx = batch

            if args.gpu >= 0:
                batch_input_ids = batch_input_ids.cuda()
                batch_input_masks = batch_input_masks.cuda()
                batch_output_ids = batch_output_ids.cuda()

            loss = model(input_ids=batch_input_ids,
                         attention_mask=batch_input_masks,
                         labels=batch_output_ids).loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            tr_loss += loss.item()
            epoch_loss += loss.item()
            nb_tr_examples += batch_input_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            # update optimizer every {gradient_accumulation_steps} steps
            if nb_tr_steps % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # do eval on dev every {eval_step} optimizer updates
                if global_step % eval_step == 0:
                    save_model = False

                    result, preds, preds_victims = evaluate_dev(
                        dev_dataloader,
                        dev_examples, dev_features,
                        tokenizer, model)

                    model.train()

                    result['global_step'] = global_step
                    result['epoch'] = epoch
                    result['batch_size'] = args.batch_size

                    dev_flag = False
                    if args.use_metric and ((best_result is None) or (
                            result[args.eval_metric] >
                            best_result[args.eval_metric])):
                        dev_flag = True
                    elif not args.use_metric:
                        dev_flag = True

                    if dev_flag:
                        if not os.path.exists(args.output_dir):
                            os.makedirs(args.output_dir)

                        best_result = result
                        save_model = True

                        # save prediction results
                        f_out = open(
                            os.path.join(args.output_dir, 'dev_results.csv'),
                            'w')
                        for line in preds:
                            f_out.write('%s' % line)
                        f_out.close()
                        f_out = open(os.path.join(args.output_dir,
                                                  'dev_results_victims.csv'),
                                     'w')
                        for line in preds_victims:
                            f_out.write('%s' % line)
                        f_out.close()

                        logged_metrics = {
                            "perpetrator pre": result["perpetrator pre"],
                            "perpetrator rec": result["perpetrator rec"],
                            "perpetrator f1": result["perpetrator f1"],
                            "victim pre": result["victim pre"],
                            "victim rec": result["victim rec"],
                            "victim f1": result["victim f1"],
                            "victim loose pre": result["victim loose pre"],
                            "victim loose rec": result["victim loose rec"],
                            "victim loose f1": result["victim loose f1"],
                            "age acc": result["age acc"],
                            "population acc": result["population acc"],
                            "sex acc": result["sex acc"],
                            "type acc": result["type acc"],
                            "city acc": result["city acc"],
                            "region acc": result["region acc"],
                            "country acc": result["country acc"],
                            "date acc": result["date acc"],
                            "month acc": result["month acc"],
                            "year acc": result["year acc"],
                            "perpetrator type acc": result[
                                "perpetrator type acc"],
                            "violation type acc": result[
                                "violation type acc"],
                            "violation type loose acc": result[
                                "violation type loose acc"],
                            "violation type pre": result[
                                "violation type pre"],
                            "violation type rec": result[
                                "violation type rec"],
                            "violation type f1": result[
                                "violation type f1"],
                            "average score": result["average"]
                        }

                        logger.info(
                            f'Epoch: {epoch}/{args.epoch}, '
                            f'Step: {nb_tr_steps % len(train_dataloader)}'
                            f' / {len(train_dataloader)}, '
                            f'used_time = {time.time() - start_time:.2f}s, '
                            f'loss = {tr_loss / nb_tr_steps:.6f}')

                        logger.info(
                            f"!!! Best dev {args.eval_metric} "
                            f"(lr={optimizer.param_groups[0]['lr']:.10f}): "
                            f"perpetrator: "
                            f"p: {result['perpetrator pre']:.2f} "
                            f"r: {result['perpetrator rec']:.2f} "
                            f"f1: {result['perpetrator f1']:.2f}, "
                            f"victim exact match: "
                            f"p: {result['victim pre']:.2f} "
                            f"r: {result['victim rec']:.2f} "
                            f"f1: {result['victim f1']:.2f}, "
                            f"victim loose match: "
                            f"p: {result['victim loose pre']:.2f} "
                            f"r: {result['victim loose rec']:.2f} "
                            f"f1: {result['victim loose f1']:.2f}, "
                            f"age acc: {result['age acc']:.2f} "
                            f"population acc: "
                            f"{result['population acc']:.2f} "
                            f"sex acc {result['sex acc']:.2f} "
                            f"type acc {result['type acc']:.2f} "
                            f"city acc {result['city acc']:.2f} "
                            f"region acc {result['region acc']:.2f} "
                            f"country acc {result['country acc']:.2f} "
                            f"date acc {result['date acc']:.2f} "
                            f"month acc {result['month acc']:.2f} "
                            f"year acc {result['year acc']:.2f}  "
                            f"perpetrator type acc "
                            f"{result['perpetrator type acc']:.2f} "
                            f"violation type acc "
                            f"{result['violation type acc']:.2f} "
                            f"violation type loose acc "
                            f"{result['violation type loose acc']:.2f} "
                            f"violation type pre "
                            f"{result['violation type pre']:.2f} "
                            f"violation type rec "
                            f"{result['violation type rec']:.2f} "
                            f"violation type f1 "
                            f"{result['violation type f1']:.2f}")

                    if save_model:
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        subdir = './pretrained_model'
                        if not os.path.exists(subdir):
                            os.makedirs(subdir)
                        output_model_file = os.path.join(
                            subdir, "pytorch_model.bin")
                        output_config_file = os.path.join(
                            subdir, "config.json")
                        torch.save(model_to_save.state_dict(),
                                   output_model_file)
                        model_to_save.config.to_json_file(
                            output_config_file)
                        tokenizer.save_vocabulary(subdir)

    model_name = './pretrained_model'
    model_class = T5ForConditionalGeneration
    tokenizer_mame = T5Tokenizer
    config_name = T5Config
    config = config_name.from_pretrained(model_name,
                                         local_files_only=True)
    tokenizer = tokenizer_mame.from_pretrained(model_name,
                                               local_files_only=True)
    model = model_class.from_pretrained(model_name, local_files_only=True)
    if args.gpu >= 0:
        model.cuda()
    result, test_preds, test_preds_victims = evaluate_all(
        test_dataloader,
        test_examples, test_features,
        tokenizer, model, num_beams=2)

    logger.info(
        f"perpetrator: "
        f"p: {result['perpetrator pre']:.2f} "
        f"r: {result['perpetrator rec']:.2f} "
        f"f1: {result['perpetrator f1']:.2f}, "
        f"victim exact match: "
        f"p: {result['victim pre']:.2f} "
        f"r: {result['victim rec']:.2f} "
        f"f1: {result['victim f1']:.2f}, "
        f"victim loose match: "
        f"p: {result['victim loose pre']:.2f} "
        f"r: {result['victim loose rec']:.2f} "
        f"f1: {result['victim loose f1']:.2f}, "
        f"age acc: {result['age acc']:.2f} "
        f"population acc: "
        f"{result['population acc']:.2f} "
        f"sex acc {result['sex acc']:.2f} "
        f"type acc {result['type acc']:.2f} "
        f"city acc {result['city acc']:.2f} "
        f"region acc {result['region acc']:.2f} "
        f"country acc {result['country acc']:.2f} "
        f"date acc {result['date acc']:.2f} "
        f"month acc {result['month acc']:.2f} "
        f"year acc {result['year acc']:.2f}  "
        f"perpetrator type acc "
        f"{result['perpetrator type acc']:.2f} "
        f"violation type acc "
        f"{result['violation type acc']:.2f} "
        f"violation type loose acc "
        f"{result['violation type loose acc']:.2f} "
        f"violation type pre "
        f"{result['violation type pre']:.2f} "
        f"violation type rec "
        f"{result['violation type rec']:.2f} "
        f"violation type f1 "
        f"{result['violation type f1']:.2f}")

    f_out = open(os.path.join(args.output_dir,
                              'test_results.csv'), 'w')
    for line in test_preds:
        f_out.write('%s' % line)
    f_out.close()
    f_out = open(os.path.join(args.output_dir,
                              'test_results_victims.csv'), 'w')
    for line in test_preds_victims:
        f_out.write('%s' % line)
    f_out.close()
    logger.info(
        f'the visualization results are saved in {args.output_dir}')

