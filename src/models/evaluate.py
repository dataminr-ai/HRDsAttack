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
    generate_dataset_t5, my_collate_t5, evaluate_all

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
pp = pprint.PrettyPrinter(indent=4)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S %Z')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UN OHCHR Detection')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gup id, set to -1 if use cpu mode')
    parser.add_argument('--test_file',
                        default='src/data/dev.json',
                        help='path to test bio file')
    parser.add_argument('--add_prefix', action='store_true',
                        help='add prefix for each task')
    parser.add_argument('--context_filter', action='store_true',
                        help='filter context')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--output_dir', default='tmp/t5_un',
                        help='path of output dir')
    parser.add_argument('--model_dir', default='./pretrained_model',
                        help='path of output dir')
    parser.add_argument('--replicate', action='store_true',
                        help='replicate the reported scores')
    parser.add_argument('--fusion', action='store_true',
                        help='use paragraph-based fusion or not')

    args = parser.parse_args()

    model_class = T5ForConditionalGeneration
    tokenizer_mame = T5Tokenizer
    config_name = T5Config
    config = config_name.from_pretrained(args.model_dir,
                                         local_files_only=True)
    tokenizer = tokenizer_mame.from_pretrained(args.model_dir,
                                               local_files_only=True)
    model = model_class.from_pretrained(args.model_dir, local_files_only=True)

    if args.fusion:
        test_examples = load_examples(args.test_file, split_doc=True)
        test_features = generate_features_t5(test_examples, tokenizer,
                                             add_prefix=args.add_prefix,
                                             max_len=512,
                                             context_filter=args.context_filter,
                                             split_doc=True, top_sentence=None,
                                             replicate=args.replicate)
    else:
        test_examples = load_examples(args.test_file)
        test_features = generate_features_t5(test_examples, tokenizer,
                                             add_prefix=args.add_prefix,
                                             max_len=512,
                                             context_filter=args.context_filter)
    test_dataset = generate_dataset_t5(test_features)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=my_collate_t5)

    if args.gpu >= 0:
        model.cuda()
    if args.fusion:
        result, test_preds, test_preds_victims = evaluate_all(
            test_dataloader,
            test_examples, test_features,
            tokenizer, model, num_beams=2)
    else:
        result, test_preds, test_preds_victims = evaluate_all(
            test_dataloader,
            test_examples, test_features,
            tokenizer, model, num_beams=1)

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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
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
