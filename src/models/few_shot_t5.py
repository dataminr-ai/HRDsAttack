from __future__ import absolute_import, division, print_function
import time
import os
import random
import math
import json
import logging

import numpy as np
import argparse
import mlflow
import torch
from torch.utils.data import Dataset
import pprint

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, \
    get_linear_schedule_with_warmup, Adafactor

pp = pprint.PrettyPrinter(indent=4)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S %Z')
logger = logging.getLogger(__name__)


class T5Dataset(Dataset):
    """
    Customized Pytorch Dataset class
    """

    def __init__(self, all_input_ids, all_mask_ids,
                 all_target_ids, all_feature_idx):
        """
        :param all_input_ids: input token ids
        :param all_mask_ids:  input mask ids
        :param all_target_ids: output token ids
        :param all_feature_idx: index of the data point
        """
        self.all_input_ids = all_input_ids
        self.all_mask_ids = all_mask_ids
        self.all_target_ids = all_target_ids
        self.all_feature_idx = all_feature_idx

    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.all_mask_ids[idx], \
               self.all_target_ids[idx], self.all_feature_idx[idx]

    def __len__(self):
        return len(self.all_input_ids)


def load_examples(input_doc, filter_name=True, training=True):
    """
    load data from files
    :param input_doc: path of the data file
    :param filter_name: True if filter out the Unknown victims
    :return: list of dictionaries
    """
    type_set = set()
    name_set = set()
    examples = []
    count_filtered = 0
    data = []
    # map for normalized month string
    month_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May',
                 6: 'June', 7: 'July', 8: 'August', 9: 'September',
                 10: 'October', 11: 'November', 12: 'December'}
    month_set = set([month_map[x] for x in month_map])
    month_set.add(None)

    # test
    # for doc in os.listdir(
    #         '/dbfs/mnt/s3fs-bucket/data/dlu/un_ohchr/data/annotated_processed_data/'):
    #     if 'full_batch' not in doc:
    #         continue
    #     if training and doc == 'full_batch_1.json':
    #         continue
    #     elif (not training) and doc != 'full_batch_1.json':
    #         continue
    #     with open(os.path.join(
    #             '/dbfs/mnt/s3fs-bucket/data/dlu/un_ohchr/data/annotated_processed_data/',
    #             doc)) as fin:
    with open(input_doc) as fin:
        data = json.load(fin)
        # data = data + cur_data

    for sample in data:
        # initialize class values as None
        date, month, year = None, None, None
        city, region, country = None, None, None
        perpetrator_type = None
        violation_types = None

        # some samples only have partial annotations
        # if the annotations contain city information, it's full annotation
        # otherwise it's partial annotation
        if 'city' not in sample['merged_sample']:
            annotation_type = 'part'
        else:
            annotation_type = 'full'

        sample_id = sample['sample']['Input.GLOBALEVENTID']
        context = sample['sample']['Input.article_interface']
        perpetrator_mention = sample['merged_sample']['perpetrator_mention']
        publish_date = sample['sample']['Input.publish_date']
        if annotation_type == 'full':
            # normalize annotation and correct annotation errors for city
            city = sample['merged_sample']['city']
            if city and city.lower() in ['n/a', 'none']:
                city = None
            if city == 'City':
                city = 'Pune'

            # normalize annotation and correct annotation errors for region
            region = sample['merged_sample']['region']
            if region and region.lower() in ['n/a', 'none']:
                region = None

            # normalize annotation and correct annotation errors for country
            country = sample['merged_sample']['country']
            if country and country.lower() in ['n/a', 'none']:
                country = None

            date = sample['merged_sample']['date']

            # normalize annotation and correct annotation errors for month
            month = sample['merged_sample']['month']
            if month and month.lower() == 'decenmber':
                month = 'December'
            if month and month.lower() == 'ocober':
                month = 'October'
            if month and month.lower() in ['n//a', 'none']:
                month = None

            # if month and month.strip().isdigit() and 1 <= int(
            #         float(month.strip())) <= 12:
            #     month = month_map[int(float(month))]
            # assert month in month_set
            try:
                if month and 1 <= int(float(month.strip())) <= 12:
                    month = month_map[int(float(month))]
            except:
                month = month
            if month not in month_set:
                logger.info(f'month error: {month}')
                logger.info(f'{sample}')
            # normalize year and correct annotation errors for year
            year = sample['merged_sample']['year']
            if sample['sample']['Input.GLOBALEVENTID'] == 1004122033:
                date = '09'
                year = '2021'
                month = 'September'

            if sample['sample']['Input.GLOBALEVENTID'] == 885531401:
                date = '13'
                year = '1839'
            if sample['sample']['Input.GLOBALEVENTID'] == 876209412:
                month = 'March'
                year = '2015'

            if year:
                year = str(int(float(year)))
            if date:
                date = str(int(float(date)))

            perpetrator_type = sample['merged_sample']['perpetrator_type']
            violation_types = sample['merged_sample']['violation_types']

            # remove annotation conflicts for violation type, the unknown type
            # does not appear together with other types
            if len(violation_types) > 1 and 'unknown' in violation_types:
                violation_types.remove('unknown')

        victims = []
        for victim in sample['merged_sample']['victims']:
            # remove unknown victim names from data if filter_name is True
            if not victim['victim_name'] and filter_name:
                continue
            victim_name = victim['victim_name']
            name_set.add(victim_name)
            victim_age = victim['victim_age_group']
            victim_population_type = victim['victim_population_type']
            victim_sex_type = victim['victim_sex_type']
            victim_type = victim['victim_type']
            victims.append([victim_name, victim_age, victim_population_type,
                            victim_sex_type, victim_type])

            victim_type = list(set(victim_type))
            if len(victim_type) > 1 and 'insufficient information' \
                    in victim_type:
                # remove annotation conflicts, insufficient information does
                # not appear together with other victim types
                victim_type.remove('insufficient information')
                count_filtered += 1

            for x in victim_type:
                type_set.add(x)

        example = {
            'id': sample_id,
            'context': context,
            'perpetrator_mention': perpetrator_mention,
            'victims': victims,
            'annotation_type': annotation_type,
            'date': date,
            'month': month,
            'year': year,
            'city': city,
            'region': region,
            'country': country,
            'perpetrator_type': perpetrator_type,
            'violation_types': violation_types,
            'publish_date': publish_date,

        }
        examples.append(example)
    return examples


def input_ids_generate(context, question,
                       task_prefix, tokenizer,
                       max_len=512, add_prefix=True, publish_date=None):
    if add_prefix:
        if not publish_date:
            input_text = '%s question: %s context: %s </s>' % (
                task_prefix, question, context)
        else:
            input_text = \
                '%s question: %s publish date: %s context: %s </s>' % (
                    task_prefix, question, publish_date, context)
    else:
        if not publish_date:
            input_text = 'question: %s context: %s </s>' % (
                question, context)
        else:
            input_text = 'question: %s publish date: %s context: %s </s>' % (
                question, publish_date, context)
    input_encodings = tokenizer.encode_plus(
        input_text, max_length=max_len)
    input_ids = input_encodings['input_ids']
    input_masks = input_encodings['attention_mask']
    # truncated input text
    truncated_text = tokenizer.decode(input_ids)
    return input_ids, input_masks, truncated_text


def generate_features_t5(examples, tokenizer, add_prefix=False,
                         max_len=512, context_filter=False):
    """
    generate features from examples
    :param examples: input examples
    :param tokenizer: tokenizer of PLM backbone
    :param add_prefix: True if add task prefix to T5 input
    :param max_len: maximum length for T5 input
    :param context_filter: check if the input paraphrase contains answer
                           before generate question-answer pairs for T5
    :return: list of features
    """
    violation_all_list = ['arbitrary detention', 'enforced disappearance',
                          'kidnapping', 'killing', 'unknown', 'other',
                          'torture']
    features = []
    for example_id, example in enumerate(examples):
        context = example['context']
        perpetrator_mention = example['perpetrator_mention']
        victims = example['victims']
        annotation_type = example['annotation_type']
        date = example['date']
        month = example['month']
        year = example['year']
        city = example['city']
        region = example['region']
        country = example['country']
        perpetrator_type = example['perpetrator_type']
        violation_types = example['violation_types']
        publish_date = example['publish_date']

        # generate question answer pairs for 'Perpetrator Mention'
        task_prefix = 'detect perpetrator'
        question = 'Does it mention any perpetrator?'
        input_ids, input_masks, truncated_text_1 = \
            input_ids_generate(context, question, task_prefix, tokenizer,
                               max_len=max_len, add_prefix=add_prefix)

        if perpetrator_mention:
            answer = 'yes </s>'
        else:
            answer = 'no </s>'
        target_encodings = tokenizer.encode_plus(answer)
        target_ids = target_encodings['input_ids']
        target_masks = target_encodings['attention_mask']

        one_feature = {
            'example_id': example_id,
            'input_ids': input_ids,
            'input_mask': input_masks,
            'target_ids': target_ids,
            'target_mask': target_masks,
            'task': 'perpetrator',
            'victim': None
        }
        features.append(one_feature)

        # generate question answer pairs for 'Victim Name'

        # sort the victims as the orders they appear in the article
        sorted_victims = []
        for victim in victims:
            # only add the victim names present
            # in the truncated_text into answer
            if victim[0] and victim[0] in truncated_text_1:
                sorted_victims.append(
                    [victim[0], truncated_text_1.find(victim[0])])
        sorted_victims.sort(key=lambda x: x[1])

        task_prefix = 'extract victims'
        question = 'Who is the victims of the violation?'
        input_ids, input_masks, truncated_text = \
            input_ids_generate(context, question, task_prefix, tokenizer,
                               max_len=max_len, add_prefix=add_prefix)

        answer = '%s </s>' % (', '.join([x[0] for x in sorted_victims]))
        target_encodings = tokenizer.encode_plus(answer)
        target_ids = target_encodings['input_ids']
        target_masks = target_encodings['attention_mask']

        one_feature = {
            'example_id': example_id,
            'input_ids': input_ids,
            'input_mask': input_masks,
            'target_ids': target_ids,
            'target_mask': target_masks,
            'task': 'victim',
            'victim': None
        }
        features.append(one_feature)

        for victim in victims:
            if victim[0] and victim[0] in truncated_text_1:
                victim_name = victim[0]

                # age
                task_prefix = 'extract victim age'
                question = 'What is the age group of %s?' % victim_name
                input_ids, input_masks, truncated_text = \
                    input_ids_generate(context, question, task_prefix,
                                       tokenizer,
                                       max_len=max_len,
                                       add_prefix=add_prefix)

                answer = '%s </s>' % (victim[1])
                target_encodings = tokenizer.encode_plus(answer)
                target_ids = target_encodings['input_ids']
                target_masks = target_encodings['attention_mask']

                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_masks,
                    'target_ids': target_ids,
                    'target_mask': target_masks,
                    'task': 'victim_age',
                    'victim': victim_name
                }
                features.append(one_feature)

                # population type
                task_prefix = 'extract victim population type'
                question = 'What is the population type of %s?' % victim_name
                input_ids, input_masks, truncated_text = \
                    input_ids_generate(context, question, task_prefix,
                                       tokenizer,
                                       max_len=max_len,
                                       add_prefix=add_prefix)

                answer = '%s </s>' % (victim[2])
                target_encodings = tokenizer.encode_plus(answer)
                target_ids = target_encodings['input_ids']
                target_masks = target_encodings['attention_mask']

                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_masks,
                    'target_ids': target_ids,
                    'target_mask': target_masks,
                    'task': 'victim_population_type',
                    'victim': victim_name
                }

                features.append(one_feature)
                # sex
                task_prefix = 'extract victim sex'
                question = 'What is the sex of %s?' % victim_name
                input_ids, input_masks, truncated_text = \
                    input_ids_generate(context, question, task_prefix,
                                       tokenizer,
                                       max_len=max_len,
                                       add_prefix=add_prefix)

                answer = '%s </s>' % (victim[3])
                target_encodings = tokenizer.encode_plus(answer)
                target_ids = target_encodings['input_ids']
                target_masks = target_encodings['attention_mask']

                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_masks,
                    'target_ids': target_ids,
                    'target_mask': target_masks,
                    'task': 'victim_sex_type',
                    'victim': victim_name
                }
                features.append(one_feature)

                # type
                task_prefix = 'extract victim type'
                question = 'Is %s a trade unionist?' % victim_name
                input_ids, input_masks, truncated_text = \
                    input_ids_generate(context, question, task_prefix,
                                       tokenizer,
                                       max_len=max_len,
                                       add_prefix=add_prefix)

                type_yes_flag = False
                if 'trade unionist' in victim[4]:
                    type_yes_flag = True
                    answer = 'yes </s>'
                else:
                    answer = 'no </s>'
                target_encodings = tokenizer.encode_plus(answer)
                target_ids = target_encodings['input_ids']
                target_masks = target_encodings['attention_mask']
                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_masks,
                    'target_ids': target_ids,
                    'target_mask': target_masks,
                    'task': 'victim_type',
                    'victim': victim_name,
                    'victim_type': 'trade unionist'
                }
                features.append(one_feature)

                task_prefix = 'extract victim type'
                question = 'Is %s a journalist?' % victim_name

                input_ids, input_masks, truncated_text = \
                    input_ids_generate(context, question, task_prefix,
                                       tokenizer,
                                       max_len=max_len,
                                       add_prefix=add_prefix)

                if 'journalist' in victim[4]:
                    type_yes_flag = True
                    answer = 'yes </s>'
                else:
                    answer = 'no </s>'
                target_encodings = tokenizer.encode_plus(answer)
                target_ids = target_encodings['input_ids']
                target_masks = target_encodings['attention_mask']
                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_masks,
                    'target_ids': target_ids,
                    'target_mask': target_masks,
                    'task': 'victim_type',
                    'victim': victim_name,
                    'victim_type': 'journalist'
                }
                features.append(one_feature)

                task_prefix = 'extract victim type'
                question = 'Is %s a human rights defender?' % victim_name
                input_ids, input_masks, truncated_text = \
                    input_ids_generate(context, question, task_prefix,
                                       tokenizer,
                                       max_len=max_len,
                                       add_prefix=add_prefix)

                if 'human rights defender' in victim[4]:
                    type_yes_flag = True
                    answer = 'yes </s>'
                else:
                    answer = 'no </s>'
                target_encodings = tokenizer.encode_plus(answer)
                target_ids = target_encodings['input_ids']
                target_masks = target_encodings['attention_mask']
                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_masks,
                    'target_ids': target_ids,
                    'target_mask': target_masks,
                    'task': 'victim_type',
                    'victim': victim_name,
                    'victim_type': 'human rights defender'
                }
                features.append(one_feature)

        if annotation_type == 'full':
            # city
            task_prefix = 'extract violation city'
            question = 'In which city did the violation happen?'
            input_ids, input_masks, truncated_text = \
                input_ids_generate(context, question, task_prefix, tokenizer,
                                   max_len=max_len, add_prefix=add_prefix)

            if context_filter:
                if city and city in truncated_text:
                    answer = '%s </s>' % city
                else:
                    answer = 'None </s>'
            else:
                answer = '%s </s>' % city

            target_encodings = tokenizer.encode_plus(answer)
            target_ids = target_encodings['input_ids']
            target_masks = target_encodings['attention_mask']
            one_feature = {
                'example_id': example_id,
                'input_ids': input_ids,
                'input_mask': input_masks,
                'target_ids': target_ids,
                'target_mask': target_masks,
                'task': 'city',
                'victim': None
            }

            features.append(one_feature)

            # region
            task_prefix = 'extract violation region'
            question = 'In which region did the violation happen?'
            input_ids, input_masks, truncated_text = \
                input_ids_generate(context, question, task_prefix, tokenizer,
                                   max_len=max_len, add_prefix=add_prefix)

            if context_filter:
                if region and region in truncated_text:
                    answer = '%s </s>' % region
                else:
                    answer = 'None </s>'
            else:
                answer = '%s </s>' % region

            target_encodings = tokenizer.encode_plus(answer)
            target_ids = target_encodings['input_ids']
            target_masks = target_encodings['attention_mask']
            one_feature = {
                'example_id': example_id,
                'input_ids': input_ids,
                'input_mask': input_masks,
                'target_ids': target_ids,
                'target_mask': target_masks,
                'task': 'region',
                'victim': None
            }
            features.append(one_feature)

            # country
            task_prefix = 'extract violation country'
            question = 'In which country did the violation happen?'
            input_ids, input_masks, truncated_text = \
                input_ids_generate(context, question, task_prefix, tokenizer,
                                   max_len=max_len, add_prefix=add_prefix)

            if context_filter:
                if country and country in truncated_text:
                    answer = '%s </s>' % country
                else:
                    answer = 'None </s>'
            else:
                answer = '%s </s>' % country

            target_encodings = tokenizer.encode_plus(answer)
            target_ids = target_encodings['input_ids']
            target_masks = target_encodings['attention_mask']
            one_feature = {
                'example_id': example_id,
                'input_ids': input_ids,
                'input_mask': input_masks,
                'target_ids': target_ids,
                'target_mask': target_masks,
                'task': 'country',
                'victim': None
            }
            features.append(one_feature)

            # date
            task_prefix = 'extract violation date'
            question = 'On which date did the violation happen?'
            input_ids, input_masks, truncated_text = \
                input_ids_generate(context, question, task_prefix, tokenizer,
                                   max_len=max_len, add_prefix=add_prefix,
                                   publish_date=publish_date)

            answer = '%s </s>' % date
            target_encodings = tokenizer.encode_plus(answer)
            target_ids = target_encodings['input_ids']
            target_masks = target_encodings['attention_mask']
            one_feature = {
                'example_id': example_id,
                'input_ids': input_ids,
                'input_mask': input_masks,
                'target_ids': target_ids,
                'target_mask': target_masks,
                'task': 'date',
                'victim': None
            }
            features.append(one_feature)

            # month
            task_prefix = 'extract violation month'
            question = 'In which month did the violation happen?'
            input_ids, input_masks, truncated_text = \
                input_ids_generate(context, question, task_prefix, tokenizer,
                                   max_len=max_len, add_prefix=add_prefix,
                                   publish_date=publish_date)

            answer = '%s </s>' % month
            target_encodings = tokenizer.encode_plus(answer)
            target_ids = target_encodings['input_ids']
            target_masks = target_encodings['attention_mask']
            one_feature = {
                'example_id': example_id,
                'input_ids': input_ids,
                'input_mask': input_masks,
                'target_ids': target_ids,
                'target_mask': target_masks,
                'task': 'month',
                'victim': None
            }
            features.append(one_feature)

            # year
            task_prefix = 'extract violation year'
            question = 'In which year did the violation happen?'
            input_ids, input_masks, truncated_text = \
                input_ids_generate(context, question, task_prefix, tokenizer,
                                   max_len=max_len, add_prefix=add_prefix,
                                   publish_date=publish_date)

            answer = '%s </s>' % year
            # print([input_text, target_text])
            target_encodings = tokenizer.encode_plus(answer)
            target_ids = target_encodings['input_ids']
            target_masks = target_encodings['attention_mask']
            one_feature = {
                'example_id': example_id,
                'input_ids': input_ids,
                'input_mask': input_masks,
                'target_ids': target_ids,
                'target_mask': target_masks,
                'task': 'year',
                'victim': None
            }
            features.append(one_feature)

            # perpetrator_type
            task_prefix = 'extract perpetrator type'
            question = 'What is the type of the perpetrator?'
            input_ids, input_masks, truncated_text = \
                input_ids_generate(context, question, task_prefix, tokenizer,
                                   max_len=max_len, add_prefix=add_prefix)

            answer = '%s </s>' % perpetrator_type
            target_encodings = tokenizer.encode_plus(answer)
            target_ids = target_encodings['input_ids']
            target_masks = target_encodings['attention_mask']
            one_feature = {
                'example_id': example_id,
                'input_ids': input_ids,
                'input_mask': input_masks,
                'target_ids': target_ids,
                'target_mask': target_masks,
                'task': 'perpetrator_type',
                'victim': None
            }
            features.append(one_feature)

            for one_violation_type in violation_all_list:
                if one_violation_type == 'unknown':
                    continue
                violation_type_answer = 'yes' if \
                    one_violation_type in violation_types else 'no'
                task_prefix = 'extract violation type'
                question = 'Is there any %s violation mentioned in the text?' \
                           % one_violation_type
                input_ids, input_masks, truncated_text = \
                    input_ids_generate(context, question, task_prefix,
                                       tokenizer,
                                       max_len=max_len,
                                       add_prefix=add_prefix)

                answer = '%s </s>' % violation_type_answer
                target_encodings = tokenizer.encode_plus(answer)
                target_ids = target_encodings['input_ids']
                target_masks = target_encodings['attention_mask']
                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_masks,
                    'target_ids': target_ids,
                    'target_mask': target_masks,
                    'task': 'violation_types',
                    'violation type': one_violation_type
                }
                features.append(one_feature)

    return features


def generate_dataset_t5(data_features):
    """
    generate dataset class for input features
    """
    all_input_ids = []
    all_output_ids = []
    all_input_masks = []
    # all_target_mask_ids = []
    all_feature_idx = []
    for feature_idx, feature in enumerate(data_features):
        all_input_ids.append(feature['input_ids'])
        all_input_masks.append(feature['input_mask'])
        all_output_ids.append(feature['target_ids'])
        # all_target_mask_ids.append(feature['target_mask'])
        all_feature_idx.append(feature_idx)

    bucket_dataset = T5Dataset(all_input_ids, all_input_masks,
                               all_output_ids, all_feature_idx)
    return bucket_dataset


def my_collate_t5(one_batch):
    """
    collate function
    """
    list_input_ids = []
    list_input_masks = []
    list_output_ids = []
    # list_target_atten_mak = []
    list_feature_idx = []
    max_len = 0
    max_len_tgt = 0
    for idx, sample in enumerate(one_batch):
        input_ids = sample[0]
        output_ids = sample[2]
        max_len = max(max_len, len(input_ids))
        max_len_tgt = max(max_len_tgt, len(output_ids))

    for idx, sample in enumerate(one_batch):
        cur_len = len(sample[0])
        cur_len_tgt = len(sample[2])

        input_ids = sample[0] + [0] * (max_len - cur_len)
        list_input_ids.append(input_ids)

        input_masks = sample[1] + [0] * (max_len - cur_len)
        list_input_masks.append(input_masks)

        output_ids = sample[2] + [0] * (max_len_tgt - cur_len_tgt)
        list_output_ids.append(output_ids)

        # target_attention_mask = sample[3] + [0] * (max_len_tgt - cur_len_tgt)
        # list_target_atten_mak.append(target_attention_mask)

        list_feature_idx.append(sample[3])

    batch_input_ids = torch.LongTensor(list_input_ids)
    batch_input_masks = torch.LongTensor(list_input_masks)
    batch_output_ids = torch.LongTensor(list_output_ids)
    batch_output_ids[batch_output_ids == 0] = -100
    # list_target_atten_mak_tensor = torch.LongTensor(list_target_atten_mak)
    batch_feature_idx = torch.LongTensor(list_feature_idx)

    return batch_input_ids, batch_input_masks, \
           batch_output_ids, batch_feature_idx


def evaluate_test_squad(dataloader, examples, features, tokenizer, model,
                        gpu=True):
    """
    evaluation function
    """
    # all_results = []

    model.eval()

    sys_count = 0
    gold_count = 0
    overlap_count = 0
    overlap_count_fuzzy = 0

    correct_num = 0
    correct_perpetrator = 0
    sys_perpetrator = 0
    gold_perpetrator = 0
    overlap_perpetrator = 0

    gold_predictions = {}
    gold_victims_dic = {}
    sys_predictions = {}
    sys_victims_dic = {}
    fusion_flag_checker = {}

    for example_id, example in enumerate(examples):
        fusion_flag_checker[example_id] = {}
        gold_predictions[example_id] = example

        # retrieve ground truth since we evaluate on
        # gold victim names for victim attributes
        gold_victims_dic[example_id] = {}
        for victim in example['victims']:
            if victim[0]:
                gold_victims_dic[example_id][victim[0]] = {
                    'age': victim[1],
                    'population': victim[2],
                    'sex': victim[3],
                    'type': set(victim[4])
                }

    all_num = 0

    for step, batch in enumerate(dataloader):
        batch_input_ids, batch_input_masks, \
        batch_output_ids, batch_feature_idx = batch

        if gpu:
            batch_input_ids = batch_input_ids.cuda()
            batch_input_masks = batch_input_masks.cuda()

        with torch.no_grad():
            preds = model.generate(input_ids=batch_input_ids,
                                   attention_mask=batch_input_masks)

        for i, example_index in enumerate(batch_feature_idx):
            pre_answer = tokenizer.decode(preds[i], skip_special_tokens=True)

            eval_feature = features[example_index.item()]
            example_id = eval_feature['example_id']
            if example_id not in sys_predictions:
                sys_predictions[example_id] = {'perpetrator_mention': False,
                                               'victims': [],
                                               'date': None, 'month': None,
                                               'year': None,
                                               'city': None, 'region': None,
                                               'country': None,
                                               'perpetrator_type': None,
                                               'violation_types': set()
                                               }
            # input_sentx = examples[example_id]['text']
            perpetrator_label = examples[example_id]['perpetrator_mention']
            victims_label = [x[0] for x in examples[example_id]['victims']]

            if eval_feature['task'] == 'perpetrator':

                if pre_answer.strip().lower() == 'yes':
                    sys_predictions[example_id]['perpetrator_mention'] = True

            if eval_feature['task'] == 'perpetrator_type' and \
                    sys_predictions[example_id]['perpetrator_type'] in \
                    [None, 'None']:
                sys_predictions[example_id][
                    'perpetrator_type'] = pre_answer.strip()
            if eval_feature['task'] == 'city':
                # and sys_predictions[example_id]['city']  in [None, 'None']:
                if sys_predictions[example_id]['city'] is None:
                    sys_predictions[example_id]['city'] = pre_answer.strip()
                if sys_predictions[example_id]['city'] == 'None':
                    # pre_answer.strip() == 'None':
                    sys_predictions[example_id]['city'] = pre_answer.strip()
            #                     print(pre_answer.strip(), 'pre_answer.strip()')
            #                 sys_predictions[example_id]['city'] = pre_answer.strip()
            if eval_feature['task'] == 'region' and \
                    sys_predictions[example_id]['region'] in [None, 'None']:
                sys_predictions[example_id]['region'] = pre_answer.strip()
            if eval_feature['task'] == 'country' and \
                    sys_predictions[example_id]['country'] in [None, 'None']:
                sys_predictions[example_id]['country'] = pre_answer.strip()

            if eval_feature['task'] == 'date' and sys_predictions[example_id][
                'date'] in [None, 'None']:
                sys_predictions[example_id]['date'] = pre_answer.strip()
            if eval_feature['task'] == 'month' and sys_predictions[example_id][
                'month'] in [None, 'None']:
                sys_predictions[example_id]['month'] = pre_answer.strip()
            if eval_feature['task'] == 'year' and sys_predictions[example_id][
                'year'] in [None, 'None']:
                sys_predictions[example_id]['year'] = pre_answer.strip()

            if eval_feature['task'] == 'victim':
                pre_answer_list = pre_answer.strip().split(',')
                # sys_count += len(victims_label)
                # gold_count += len(pre_answer_list)
                for one_vic in pre_answer_list:
                    one_vic = one_vic.strip()
                    if one_vic.strip() == '':
                        continue
                    insert_flag = True
                    for x_vic in sys_predictions[example_id]['victims']:
                        if one_vic in x_vic[0]:
                            insert_flag = False
                    if insert_flag:
                        sys_predictions[example_id]['victims'].append(
                            [one_vic])

            if example_id not in sys_victims_dic:
                sys_victims_dic[example_id] = {}
            if eval_feature['task'] == 'victim_age':
                cur_victim_name = eval_feature['victim']
                if cur_victim_name not in sys_victims_dic[example_id]:
                    sys_victims_dic[example_id][cur_victim_name] = {}
                sys_victims_dic[example_id][cur_victim_name][
                    'age'] = pre_answer.strip()
            if eval_feature['task'] == 'victim_population_type':
                cur_victim_name = eval_feature['victim']
                if cur_victim_name not in sys_victims_dic[example_id]:
                    sys_victims_dic[example_id][cur_victim_name] = {}
                sys_victims_dic[example_id][cur_victim_name][
                    'population'] = pre_answer.strip()
            if eval_feature['task'] == 'victim_sex_type':
                cur_victim_name = eval_feature['victim']
                if cur_victim_name not in sys_victims_dic[example_id]:
                    sys_victims_dic[example_id][cur_victim_name] = {}
                sys_victims_dic[example_id][cur_victim_name][
                    'sex'] = pre_answer.strip()
            if eval_feature['task'] == 'victim_type':
                cur_victim_name = eval_feature['victim']
                cur_victim_type = eval_feature['victim_type']
                if cur_victim_name not in sys_victims_dic[example_id]:
                    sys_victims_dic[example_id][cur_victim_name] = {}
                if 'type' not in sys_victims_dic[example_id][cur_victim_name]:
                    sys_victims_dic[example_id][cur_victim_name][
                        'type'] = set()
                if pre_answer.strip() == 'yes':
                    sys_victims_dic[example_id][cur_victim_name]['type'].add(
                        cur_victim_type)

            if eval_feature['task'] == 'violation_types':
                cur_violation_type = eval_feature['violation type']
                if pre_answer.strip() == 'yes':
                    # and cur_violation_type not in flag_cache[example_id]:
                    sys_predictions[example_id]['violation_types'].add(
                        cur_violation_type)
                fusion_flag_checker[example_id][
                    cur_violation_type] = pre_answer.strip()
            all_num += 1

    all_victim_count = 0
    correct_age = 0
    correct_population = 0
    correct_sex = 0
    correct_type = 0

    f1_victim_sys = 0
    f1_victim_gold = 0
    f1_victim_overlap = 0

    correct_city = 0
    correct_region = 0
    correct_country = 0
    correct_date = 0
    correct_month = 0
    correct_year = 0

    correct_perpetrator_type = 0
    correct_violation_type = 0
    top_correct_violation_type = 0
    gold_correct_violation_type = 0
    sys_correct_violation_type = 0
    shared_correct_violation_type = 0

    outputs = []
    outputs_victims = []
    for example_id in gold_predictions:
        '''
        'id': id,
        'context': context,
        'perpetrator_mention': perpetrator_mention,
        'victims': victims
        '''
        text = examples[example_id]['context']
        perpetrator_mention = examples[example_id]['perpetrator_mention']
        assert perpetrator_mention == gold_predictions[example_id][
            'perpetrator_mention']

        gold_pre = gold_predictions[example_id]
        sys_pre = sys_predictions[example_id]

        if gold_pre['date'] == sys_pre['date']:
            correct_date += 1
        if gold_pre['month'] == sys_pre['month']:
            correct_month += 1
        if gold_pre['year'] == sys_pre['year']:
            correct_year += 1

        if gold_pre['city'] == sys_pre['city']:
            correct_city += 1
        if gold_pre['region'] == sys_pre['region']:
            correct_region += 1
        if gold_pre['country'] == sys_pre['country']:
            correct_country += 1

        if gold_pre['perpetrator_type'] == sys_pre['perpetrator_type']:
            correct_perpetrator_type += 1
        gold_pre['violation_types'] = set(gold_pre['violation_types'])
        if len(sys_pre['violation_types']) == 0:
            sys_pre['violation_types'] = set(['unknown'])
        sys_correct_violation_type += len(sys_pre['violation_types'])
        gold_correct_violation_type += len(gold_pre['violation_types'])
        violation_flag = False
        for x_vio in sys_pre['violation_types']:
            if x_vio in gold_pre['violation_types']:
                violation_flag = True
                shared_correct_violation_type += 1
        if gold_pre['violation_types'] == sys_pre['violation_types']:
            correct_violation_type += 1
        if violation_flag:
            top_correct_violation_type += 1
        if sys_pre['perpetrator_mention']:
            sys_perpetrator += 1
            if gold_pre['perpetrator_mention']:
                overlap_perpetrator += 1
        if gold_pre['perpetrator_mention']:
            gold_perpetrator += 1

        sys_count += len(sys_pre['victims'])
        gold_count += len(gold_pre['victims'])

        sys_pre['victims'].reverse()
        tmp_sys_pre_vic = []
        for one_vic in sys_pre['victims']:
            inser_f = True
            for one_vic_cur in tmp_sys_pre_vic:
                if one_vic[0] in one_vic_cur[0]:
                    inser_f = False
                    break
            if inser_f:
                tmp_sys_pre_vic.append(one_vic)
        tmp_sys_pre_vic.reverse()
        sys_pre['victims'] = tmp_sys_pre_vic

        added_fuzzy = 0
        gold_victim_name_list = [x[0] for x in gold_pre['victims']]
        for one_vic in sys_pre['victims']:
            if one_vic[0].strip() == '':
                continue
            if one_vic[0] in gold_victim_name_list:
                overlap_count += 1
            for two_vic in gold_pre['victims']:
                if two_vic[0].strip() == '':
                    continue
                if (one_vic[0] in two_vic[0]) or (two_vic[0] in one_vic[0]):
                    added_fuzzy += 1
                    break
        overlap_count_fuzzy += added_fuzzy
        print(gold_victim_name_list, sys_pre['victims'])
        if added_fuzzy > len(sys_pre['victims']) or added_fuzzy > len(
                gold_pre['victims']):
            print(sys_pre['victims'], gold_pre['victims'], added_fuzzy,
                  len(sys_pre['victims']),
                  len(gold_pre['victims']))

        pre_vic_str_list = ' '.join([x[0] for x in sys_pre['victims']])
        pre_vic_str_list = pre_vic_str_list.split()

        gold_vic_str_list = ' '.join(gold_victim_name_list)
        gold_vic_str_list = gold_vic_str_list.split()
        outputs.append(
            '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t'
            '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %
            (text.replace('\n', ' '), perpetrator_mention,
             sys_pre['perpetrator_mention'],
             gold_pre['perpetrator_type'], sys_pre['perpetrator_type'],
             gold_pre['city'], sys_pre['city'],
             gold_pre['region'], sys_pre['region'],
             gold_pre['country'], sys_pre['country'],
             gold_pre['date'], sys_pre['date'],
             gold_pre['month'], sys_pre['month'],
             gold_pre['year'], sys_pre['year'],
             ', '.join(gold_pre['violation_types']),
             ', '.join(sys_pre['violation_types']),
             ', '.join(gold_victim_name_list),
             ', '.join([x[0] for x in sys_pre['victims']])))

        for x in pre_vic_str_list:
            if x.strip() == '':
                continue
            f1_victim_sys += 1
            if x.strip() in gold_vic_str_list:
                f1_victim_overlap += 1
        for x in gold_vic_str_list:
            if x.strip() == '':
                continue
            f1_victim_gold += 1

        victims_gold = gold_victims_dic[example_id]
        victims_sys = sys_victims_dic[example_id]
        for victim in victims_gold:
            if victim not in victims_sys:
                continue
            all_victim_count += 1
            if victims_gold[victim]['age'] == victims_sys[victim]['age']:
                correct_age += 1
            if victims_gold[victim]['population'] == victims_sys[victim][
                'population']:
                correct_population += 1
            if victims_gold[victim]['sex'] == victims_sys[victim]['sex']:
                correct_sex += 1
            if len(victims_sys[victim]['type']) == 0:
                victims_sys[victim]['type'] = set(['insufficient information'])
            if victims_gold[victim]['type'] == victims_sys[victim]['type']:
                correct_type += 1

            outputs_victims.append(
                '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
                    text.replace('\n', ' '), victim,
                    victims_gold[victim]['age'],
                    victims_sys[victim]['age'],
                    victims_gold[victim]['population'],
                    victims_sys[victim]['population'],
                    victims_gold[victim]['sex'],
                    victims_sys[victim]['sex'],
                    ', '.join(
                        victims_gold[victim]['type']),
                    ', '.join(
                        victims_sys[victim]['type'])))

    print(gold_count, sys_count, overlap_count, overlap_count_fuzzy)

    if sys_count == 0:
        p = 0
        p_fuzzy = 0
    else:
        p = overlap_count / sys_count
        p_fuzzy = overlap_count_fuzzy / sys_count

    if gold_count == 0:
        r = 0
        r_fuzzy = 0
    else:
        r = overlap_count / gold_count
        r_fuzzy = overlap_count_fuzzy / gold_count

    if p == 0 or r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)

    if p_fuzzy == 0 or r_fuzzy == 0:
        f1_fuzzy = 0
    else:
        f1_fuzzy = 2 * p_fuzzy * r_fuzzy / (p_fuzzy + r_fuzzy)

    if sys_perpetrator == 0:
        p_perpetrator = 0
    else:
        p_perpetrator = overlap_perpetrator / sys_perpetrator

    if gold_perpetrator == 0:
        r_perpetrator = 0
    else:
        r_perpetrator = overlap_perpetrator / gold_perpetrator

    if p_perpetrator == 0 or r_perpetrator == 0:
        f1_perpetrator = 0
    else:
        f1_perpetrator = 2 * p_perpetrator * r_perpetrator / (
                p_perpetrator + r_perpetrator)

    age_acc = correct_age / all_victim_count
    population_acc = correct_population / all_victim_count
    sex_acc = correct_sex / all_victim_count
    type_acc = correct_type / all_victim_count
    city_acc = correct_city / len(gold_predictions)
    region_acc = correct_region / len(gold_predictions)
    country_acc = correct_country / len(gold_predictions)

    date_acc = correct_date / len(gold_predictions)
    month_acc = correct_month / len(gold_predictions)
    year_acc = correct_year / len(gold_predictions)

    perpetrator_type_acc = correct_perpetrator_type / len(gold_predictions)
    violation_type_acc = correct_violation_type / len(gold_predictions)
    top_violation_type_acc = top_correct_violation_type / len(gold_predictions)
    if sys_correct_violation_type == 0:
        p_violation_type = 0
    else:
        p_violation_type = shared_correct_violation_type / \
                           sys_correct_violation_type

    if gold_correct_violation_type == 0:
        r_violation_type = 0
    else:
        r_violation_type = shared_correct_violation_type / \
                           gold_correct_violation_type

    if p_violation_type == 0 or r_violation_type == 0:
        f1_violation_type = 0
    else:
        f1_violation_type = 2 * p_violation_type * r_violation_type / (
                p_violation_type + r_violation_type)
    result = {
        'perpetrator pre': 100 * p_perpetrator,
        'perpetrator rec': 100 * r_perpetrator,
        'perpetrator f1': 100 * f1_perpetrator,
        'victim pre': 100 * p,
        'victim rec': 100 * r,
        'victim f1': 100 * f1,
        'victim loose pre': 100 * p_fuzzy,
        'victim loose rec': 100 * r_fuzzy,
        'victim loose f1': 100 * f1_fuzzy,
        'age acc': 100 * age_acc,
        'population acc': 100 * population_acc,
        'sex acc': 100 * sex_acc,
        'type acc': 100 * type_acc,
        'city acc': 100 * city_acc,
        'region acc': 100 * region_acc,
        'country acc': 100 * country_acc,
        'date acc': 100 * date_acc,
        'month acc': 100 * month_acc,
        'year acc': 100 * year_acc,
        'perpetrator type acc': 100 * perpetrator_type_acc,
        'violation type acc': 100 * violation_type_acc,
        'violation type loose acc': 100 * top_violation_type_acc,
        'violation type pre': 100 * p_violation_type,
        'violation type rec': 100 * r_violation_type,
        'violation type f1': 100 * f1_violation_type
    }
    average_score_list = []
    for metric in result:
        if 'pre' in metric or 'rec' in metric:
            continue
        # if 'violation type acc' in metric or \
        #         'violation type loose acc' in metric:
        if 'violation' in metric and 'type acc' not in metric:
            continue
        average_score_list.append(result[metric])
    result['average'] = float(sum(average_score_list)) / float(
        len(average_score_list))
    return result, outputs, outputs_victims


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
    dev_examples = load_examples(args.dev_file, training=False)
    test_examples = load_examples(args.test_file, training=False)

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
                                         context_filter=args.context_filter)

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

    # with mlflow.start_run():
    optimizer.zero_grad()
    # model.zero_grad()
    # mlflow.set_tag("mlflow.runName", args.run_name)
    # mlflow.log_params(saved_params)
    for epoch in range(args.epoch):
        epoch_loss = 0
        # model.train()
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
                # model.zero_grad()
                global_step += 1

                # do eval on dev every {eval_step} optimizer updates
                if global_step % eval_step == 0:
                    save_model = False

                    result, preds, preds_victims = evaluate_test_squad(
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
                        # mlflow.log_metrics(logged_metrics,
                        #                    step=global_step)
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

                        # mlflow.log_artifact("dev_results.csv")
                        # mlflow.log_artifact("dev_results_victims.csv")

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
                        # mlflow.log_artifacts("./pretrained_model",
                        #                      artifact_path="model")

        model_name = './pretrained_model'
        model_class = T5ForConditionalGeneration
        tokenizer_mame = T5Tokenizer  # Fast
        config_name = T5Config
        config = config_name.from_pretrained(model_name,
                                             local_files_only=True)
        tokenizer = tokenizer_mame.from_pretrained(model_name,
                                                   local_files_only=True)
        model = model_class.from_pretrained(model_name, local_files_only=True)
        if args.gpu >= 0:
            model.cuda()
        result, test_preds, test_preds_victims = evaluate_test_squad(
            test_dataloader,
            test_examples, test_features,
            tokenizer, model)

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
        # mlflow.log_artifact("test_results.csv")
        # mlflow.log_artifact("test_results_victims.csv")
