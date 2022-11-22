import os
import json
import math

import spacy
import transformers
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pprint

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config


class T5Dataset(Dataset):
    def __init__(self, all_input_ids, all_mask_ids,
                 all_target_ids, all_target_mask_ids, all_feature_idex):
        self.all_input_ids = all_input_ids
        self.all_mask_ids = all_mask_ids
        self.all_target_ids = all_target_ids
        self.all_target_mask_ids = all_target_mask_ids
        self.all_feature_idex = all_feature_idex

    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.all_mask_ids[idx], \
               self.all_target_ids[idx], self.all_target_mask_ids[idx], \
               self.all_feature_idex[idx]

    def __len__(self):
        return len(self.all_input_ids)


def load_examples_e2e(input_doc, pred_vims):
    type_set = set()
    name_set = set()
    cache_set = set()
    examples = []
    count_filtered = 0
    month_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May',
                 6: 'June', 7: 'July', 8: 'August', 9: 'September',
                 10: 'October', 11: 'November', 12: 'December'}

    with open(input_doc) as f:
        data = json.load(f)

    for idx, sample in enumerate(data):
        date, month, year = None, None, None
        city, region, country = None, None, None
        perpetrator_type = None
        violation_types = None

        if 'city' not in sample['merged_sample']:

            # if doc == 'full_batch_2.json':
            annotation_type = 'part'
        else:
            annotation_type = 'full'
        assert annotation_type == 'full'
        id = sample['sample']['Input.GLOBALEVENTID']
        # title = sample['title']
        context = sample['sample']['Input.article_interface']

        doc = nlp(context)
        str_list = []
        tmp_str = ''
        for sent in doc.sents:
            tmp_str += ' ' + sent.text
            #                 if len(tmp_str.split()) > 200:
            #                     str_list.append(tmp_str.strip())
            #                     print(len(tmp_str))
            #                     tmp_str = ''
            if len(tmp_str) > 600:
                str_list.append(tmp_str.strip())
                tmp_str = ''
        if tmp_str.strip() != '':
            str_list.append(tmp_str.strip())

        perpetrator_mention = sample['merged_sample'][
            'perpetrator_mention']
        publish_date = sample['sample']['Input.publish_date']
        if annotation_type == 'full':
            city = sample['merged_sample']['city']
            if city and city.lower() in ['n/a', 'none']:
                city = None
            if city == 'City':
                city = 'Pune'
                # pp.pprint(sample)

            region = sample['merged_sample']['region']
            if region and region.lower() in ['n/a', 'none']:
                region = None
            country = sample['merged_sample']['country']
            if country and country.lower() in ['n/a', 'none']:
                country = None
            date = sample['merged_sample']['date']
            month = sample['merged_sample']['month']
            if month and month.lower() == 'decenmber':
                month = 'December'
            if month and month.lower() == 'ocober':
                month = 'October'
            if month and month.lower() in ['n//a', 'none']:
                month = None
            try:
                if month and 1 <= int(float(month.strip())) <= 12:
                    month = month_map[int(float(month))]
            except:
                month = month
            # if month == '2015':
            #     pp.pprint((sample))
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
            # if month and (int(float(month))<1 or int(float(month))>12):
            #     pp.pprint(sample)
            perpetrator_type = sample['merged_sample']['perpetrator_type']
            violation_types = sample['merged_sample']['violation_types']
            for x in violation_types:
                cache_set.add(x)
            if len(violation_types) > 1 and 'unknown' in violation_types:
                # print(violation_types, sample['sample']['Input.GLOBALEVENTID'])
                violation_types.remove('unknown')
                # count_filtered += 1

        victims = []
        pre_victims_list = pred_vims[idx]
        for victim in pre_victims_list:
            victim_name = victim[0]
            name_set.add(victim_name)
            victim_age = None
            victim_population_type = None
            victim_sex_type = None
            victim_type = None
            victims.append(
                [victim_name, victim_age, victim_population_type,
                 victim_sex_type, victim_type])

        example = {
            'id': id,
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
            'str_list': str_list

        }
        #             print(example)
        examples.append(example)
    # print(type_set, count_filtered)
    # print(name_set)
    # print(cache_set)
    return examples


def generate_features_t5(examples, tokenizer, max_len=512):
    len_m = 0
    violation_all_list = ['arbitrary detention', 'enforced disappearance',
                          'kidnapping', 'killing', 'unknown', 'other',
                          'torture']
    features = []
    for example_id, example in enumerate(examples):

        id = example['id']
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
        str_list = example['str_list']

        for context in str_list[:]:
            #         for _ in range(1):
            #             context = context
            #             input_encodings = tokenizer.encode_plus(one_context)
            #             len_m = max(len_m, len(input_encodings['input_ids']))

            prefix = 'detect perpetrator'

            question = 'Does it mention any perpetrator?'
            # input_text = '%s question: %s context: %s </s>' % (prefix, question, context)
            input_text = '%s question: %s context: %s </s>' % (
                prefix, question, context)

            input_encodings = tokenizer.encode_plus(input_text,
                                                    max_length=max_len)
            input_ids = input_encodings['input_ids']
            input_mask = input_encodings['attention_mask']

            norm_text = tokenizer.decode(input_ids)
            if perpetrator_mention:
                target_text = 'yes </s>'
            else:
                target_text = 'no </s>'
            # print([input_text, target_text])
            target_encodings = tokenizer.encode_plus(target_text)
            target_ids = target_encodings['input_ids']
            target_mask = target_encodings['attention_mask']
            one_feature = {
                'example_id': example_id,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'target_ids': target_ids,
                'target_mask': target_mask,
                'task': 'perpetrator',
                'victim': None
            }
            features.append(one_feature)
            # print(one_feature)

            norm_victims = []
            for victim in victims:
                if victim[0] and victim[0] in norm_text:
                    norm_victims.append([victim[0], norm_text.find(victim[0])])
            norm_victims.sort(key=lambda x: x[1])
            # print(norm_victims)
            prefix = 'extract victims'
            question = 'Who is the victims of the violation?'
            # input_text = '%s question: %s context: %s </s>' % (prefix, question, context)
            input_text = '%s question: %s context: %s </s>' % (
                prefix, question, context)

            input_encodings = tokenizer.encode_plus(input_text, max_length=512)
            input_ids = input_encodings['input_ids']
            input_mask = input_encodings['attention_mask']

            target_text = '%s </s>' % (', '.join([x[0] for x in norm_victims]))
            target_encodings = tokenizer.encode_plus(target_text)
            target_ids = target_encodings['input_ids']
            target_mask = target_encodings['attention_mask']

            # print([input_text, norm_text, target_text])
            one_feature = {
                'example_id': example_id,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'target_ids': target_ids,
                'target_mask': target_mask,
                'task': 'victim',
                'victim': None
            }
            features.append(one_feature)

            '''
            victim_age = victim['victim_age_group']
            victim_population_type = victim['victim_population_type']
            victim_sex_type = victim['victim_sex_type']
            victim_type = victim['victim_type']
            '''

            for victim in victims:
                if victim[0] and victim[0] in norm_text:
                    victim_name = victim[0]

                    # age
                    prefix = 'extract victim age'
                    question = 'What is the age group of %s?' % victim_name
                    input_text = '%s question: %s context: %s </s>' % (
                        prefix, question, context)

                    input_encodings = tokenizer.encode_plus(input_text,
                                                            max_length=512)
                    input_ids = input_encodings['input_ids']
                    input_mask = input_encodings['attention_mask']

                    target_text = '%s </s>' % (victim[1])
                    target_encodings = tokenizer.encode_plus(target_text)
                    target_ids = target_encodings['input_ids']
                    target_mask = target_encodings['attention_mask']

                    # print([input_text, norm_text, target_text])
                    one_feature = {
                        'example_id': example_id,
                        'input_ids': input_ids,
                        'input_mask': input_mask,
                        'target_ids': target_ids,
                        'target_mask': target_mask,
                        'task': 'victim_age',
                        'victim': victim_name
                    }
                    features.append(one_feature)

                    # population type
                    prefix = 'extract victim population type'
                    question = 'What is the population type of %s?' % victim_name
                    input_text = '%s question: %s context: %s </s>' % (
                        prefix, question, context)

                    input_encodings = tokenizer.encode_plus(input_text,
                                                            max_length=512)
                    input_ids = input_encodings['input_ids']
                    input_mask = input_encodings['attention_mask']

                    target_text = '%s </s>' % (victim[2])
                    target_encodings = tokenizer.encode_plus(target_text)
                    target_ids = target_encodings['input_ids']
                    target_mask = target_encodings['attention_mask']
                    # print([input_text, target_text])
                    one_feature = {
                        'example_id': example_id,
                        'input_ids': input_ids,
                        'input_mask': input_mask,
                        'target_ids': target_ids,
                        'target_mask': target_mask,
                        'task': 'victim_population_type',
                        'victim': victim_name
                    }

                    features.append(one_feature)
                    # sex
                    prefix = 'extract victim sex'
                    question = 'What is the sex of %s?' % victim_name
                    input_text = '%s question: %s context: %s </s>' % (
                        prefix, question, context)

                    input_encodings = tokenizer.encode_plus(input_text,
                                                            max_length=512)
                    input_ids = input_encodings['input_ids']
                    input_mask = input_encodings['attention_mask']

                    target_text = '%s </s>' % (victim[3])
                    target_encodings = tokenizer.encode_plus(target_text)
                    target_ids = target_encodings['input_ids']
                    target_mask = target_encodings['attention_mask']
                    # print([input_text, norm_text, target_text])
                    one_feature = {
                        'example_id': example_id,
                        'input_ids': input_ids,
                        'input_mask': input_mask,
                        'target_ids': target_ids,
                        'target_mask': target_mask,
                        'task': 'victim_sex_type',
                        'victim': victim_name
                    }
                    features.append(one_feature)

                    # type
                    # trade unionist', 'journalist', 'human rights defender', 'insufficient information'
                    prefix = 'extract victim type'
                    question = 'Is %s a trade unionist?' % victim_name
                    input_text = '%s question: %s context: %s </s>' % (
                        prefix, question, context)

                    input_encodings = tokenizer.encode_plus(input_text,
                                                            max_length=512)
                    input_ids = input_encodings['input_ids']
                    input_mask = input_encodings['attention_mask']
                    type_yes_flag = False
                    target_text = 'no </s>'
                    target_encodings = tokenizer.encode_plus(target_text)
                    target_ids = target_encodings['input_ids']
                    target_mask = target_encodings['attention_mask']
                    # print([input_text, norm_text, target_text])
                    one_feature = {
                        'example_id': example_id,
                        'input_ids': input_ids,
                        'input_mask': input_mask,
                        'target_ids': target_ids,
                        'target_mask': target_mask,
                        'task': 'victim_type',
                        'victim': victim_name,
                        'victim_type': 'trade unionist'
                    }
                    features.append(one_feature)

                    prefix = 'extract victim type'
                    question = 'Is %s a journalist?' % victim_name
                    input_text = '%s question: %s context: %s </s>' % (
                        prefix, question, context)

                    input_encodings = tokenizer.encode_plus(input_text,
                                                            max_length=512)
                    input_ids = input_encodings['input_ids']
                    input_mask = input_encodings['attention_mask']
                    target_text = 'no </s>'
                    target_encodings = tokenizer.encode_plus(target_text)
                    target_ids = target_encodings['input_ids']
                    target_mask = target_encodings['attention_mask']
                    # print([input_text, norm_text, target_text])
                    one_feature = {
                        'example_id': example_id,
                        'input_ids': input_ids,
                        'input_mask': input_mask,
                        'target_ids': target_ids,
                        'target_mask': target_mask,
                        'task': 'victim_type',
                        'victim': victim_name,
                        'victim_type': 'journalist'
                    }
                    features.append(one_feature)

                    prefix = 'extract victim type'
                    question = 'Is %s a human rights defender?' % victim_name
                    input_text = '%s question: %s context: %s </s>' % (
                        prefix, question, context)

                    input_encodings = tokenizer.encode_plus(input_text,
                                                            max_length=512)
                    input_ids = input_encodings['input_ids']
                    input_mask = input_encodings['attention_mask']
                    target_text = 'no </s>'
                    target_encodings = tokenizer.encode_plus(target_text)
                    target_ids = target_encodings['input_ids']
                    target_mask = target_encodings['attention_mask']
                    # print([input_text, target_text])
                    one_feature = {
                        'example_id': example_id,
                        'input_ids': input_ids,
                        'input_mask': input_mask,
                        'target_ids': target_ids,
                        'target_mask': target_mask,
                        'task': 'victim_type',
                        'victim': victim_name,
                        'victim_type': 'human rights defender'
                    }
                    features.append(one_feature)

            if annotation_type == 'full':
                # city
                prefix = 'extract violation city'
                question = 'In which city did the violation happen?'
                # input_text = '%s question: %s context: %s </s>' % (prefix, question, context)
                input_text = '%s question: %s context: %s </s>' % (
                    prefix, question, context)

                input_encodings = tokenizer.encode_plus(input_text,
                                                        max_length=max_len)
                input_ids = input_encodings['input_ids']
                input_mask = input_encodings['attention_mask']

                norm_text = tokenizer.decode(input_ids)
                target_text = '%s </s>' % city
                # print([input_text, target_text])
                target_encodings = tokenizer.encode_plus(target_text)
                target_ids = target_encodings['input_ids']
                target_mask = target_encodings['attention_mask']
                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'target_ids': target_ids,
                    'target_mask': target_mask,
                    'task': 'city',
                    'victim': None
                }
                features.append(one_feature)

                # region
                prefix = 'extract violation region'
                question = 'In which region did the violation happen?'
                # input_text = '%s question: %s context: %s </s>' % (prefix, question, context)
                input_text = '%s question: %s context: %s </s>' % (
                    prefix, question, context)

                input_encodings = tokenizer.encode_plus(input_text,
                                                        max_length=max_len)
                input_ids = input_encodings['input_ids']
                input_mask = input_encodings['attention_mask']

                norm_text = tokenizer.decode(input_ids)
                target_text = '%s </s>' % region
                # print([input_text, target_text])
                target_encodings = tokenizer.encode_plus(target_text)
                target_ids = target_encodings['input_ids']
                target_mask = target_encodings['attention_mask']
                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'target_ids': target_ids,
                    'target_mask': target_mask,
                    'task': 'region',
                    'victim': None
                }
                features.append(one_feature)

                # country
                prefix = 'extract violation country'
                question = 'In which country did the violation happen?'
                # input_text = '%s question: %s context: %s </s>' % (prefix, question, context)
                input_text = '%s question: %s context: %s </s>' % (
                    prefix, question, context)

                input_encodings = tokenizer.encode_plus(input_text,
                                                        max_length=max_len)
                input_ids = input_encodings['input_ids']
                input_mask = input_encodings['attention_mask']

                norm_text = tokenizer.decode(input_ids)
                target_text = '%s </s>' % country
                # print([input_text, target_text])
                target_encodings = tokenizer.encode_plus(target_text)
                target_ids = target_encodings['input_ids']
                target_mask = target_encodings['attention_mask']
                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'target_ids': target_ids,
                    'target_mask': target_mask,
                    'task': 'country',
                    'victim': None
                }
                features.append(one_feature)

                # date
                prefix = 'extract violation date'
                question = 'On which date did the violation happen?'
                # input_text = '%s question: %s context: %s </s>' % (prefix, question, context)
                input_text = '%s question: %s publish date: %s context: %s </s>' % (
                    prefix, question, publish_date, context)

                input_encodings = tokenizer.encode_plus(input_text,
                                                        max_length=max_len)
                input_ids = input_encodings['input_ids']
                input_mask = input_encodings['attention_mask']

                norm_text = tokenizer.decode(input_ids)
                target_text = '%s </s>' % date
                # print([input_text, target_text])
                target_encodings = tokenizer.encode_plus(target_text)
                target_ids = target_encodings['input_ids']
                target_mask = target_encodings['attention_mask']
                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'target_ids': target_ids,
                    'target_mask': target_mask,
                    'task': 'date',
                    'victim': None
                }
                features.append(one_feature)

                # month
                prefix = 'extract violation month'
                question = 'In which month did the violation happen?'
                # input_text = '%s question: %s context: %s </s>' % (prefix, question, context)
                input_text = '%s question: %s publish date: %s context: %s </s>' % (
                    prefix, question, publish_date, context)

                input_encodings = tokenizer.encode_plus(input_text,
                                                        max_length=max_len)
                input_ids = input_encodings['input_ids']
                input_mask = input_encodings['attention_mask']

                norm_text = tokenizer.decode(input_ids)
                target_text = '%s </s>' % month
                # print([input_text, target_text])
                target_encodings = tokenizer.encode_plus(target_text)
                target_ids = target_encodings['input_ids']
                target_mask = target_encodings['attention_mask']
                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'target_ids': target_ids,
                    'target_mask': target_mask,
                    'task': 'month',
                    'victim': None
                }
                features.append(one_feature)

                # year
                prefix = 'extract violation year'
                question = 'In which year did the violation happen?'
                # input_text = '%s question: %s context: %s </s>' % (prefix, question, context)
                input_text = '%s question: %s publish date: %s context: %s </s>' % (
                    prefix, question, publish_date, context)

                input_encodings = tokenizer.encode_plus(input_text,
                                                        max_length=max_len)
                input_ids = input_encodings['input_ids']
                input_mask = input_encodings['attention_mask']

                norm_text = tokenizer.decode(input_ids)
                target_text = '%s </s>' % year
                # print([input_text, target_text])
                target_encodings = tokenizer.encode_plus(target_text)
                target_ids = target_encodings['input_ids']
                target_mask = target_encodings['attention_mask']
                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'target_ids': target_ids,
                    'target_mask': target_mask,
                    'task': 'year',
                    'victim': None
                }
                features.append(one_feature)

                # perpetrator_type = example['perpetrator_type']
                # violation_types = example['violation_types']

                # perpetrator_type
                prefix = 'extract perpetrator type'
                question = 'What is the type of the perpetrator?'
                # input_text = '%s question: %s context: %s </s>' % (prefix, question, context)
                input_text = '%s question: %s context: %s </s>' % (
                    prefix, question, context)

                input_encodings = tokenizer.encode_plus(input_text,
                                                        max_length=max_len)
                input_ids = input_encodings['input_ids']
                input_mask = input_encodings['attention_mask']

                norm_text = tokenizer.decode(input_ids)
                target_text = '%s </s>' % perpetrator_type
                # print([input_text, target_text])
                target_encodings = tokenizer.encode_plus(target_text)
                target_ids = target_encodings['input_ids']
                target_mask = target_encodings['attention_mask']
                one_feature = {
                    'example_id': example_id,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'target_ids': target_ids,
                    'target_mask': target_mask,
                    'task': 'perpetrator_type',
                    'victim': None
                }
                features.append(one_feature)

                for one_violation_type in violation_all_list:
                    if one_violation_type == 'unknown':
                        continue
                    violation_type_answer = 'yes' if one_violation_type in violation_types else 'no'
                    prefix = 'extract violation type'
                    question = 'Is there any %s violation mentioned in the text?' % one_violation_type
                    # input_text = '%s question: %s context: %s </s>' % (prefix, question, context)
                    input_text = '%s question: %s context: %s </s>' % (
                        prefix, question, context)

                    input_encodings = tokenizer.encode_plus(input_text,
                                                            max_length=max_len)
                    input_ids = input_encodings['input_ids']
                    input_mask = input_encodings['attention_mask']

                    norm_text = tokenizer.decode(input_ids)
                    target_text = '%s </s>' % violation_type_answer
                    # print([input_text, target_text])
                    target_encodings = tokenizer.encode_plus(target_text)
                    target_ids = target_encodings['input_ids']
                    target_mask = target_encodings['attention_mask']
                    one_feature = {
                        'example_id': example_id,
                        'input_ids': input_ids,
                        'input_mask': input_mask,
                        'target_ids': target_ids,
                        'target_mask': target_mask,
                        'task': 'violation_types',
                        'violation type': one_violation_type
                    }
                    features.append(one_feature)

        # print(one_feature)
    print('len max', len_m)
    return features


def generate_dataset_t5(features):
    all_input_ids = []
    all_target_ids = []
    all_mask_ids = []
    all_target_mask_ids = []
    all_feature_idex = []
    for feature_idex, feature in enumerate(features):
        all_input_ids.append(feature['input_ids'])
        all_mask_ids.append(feature['input_mask'])
        all_target_ids.append(feature['target_ids'])
        all_target_mask_ids.append(feature['target_mask'])
        all_feature_idex.append(feature_idex)

    bucket_dataset = T5Dataset(all_input_ids, all_mask_ids, all_target_ids,
                               all_target_mask_ids, all_feature_idex)
    return bucket_dataset


def my_collate_threat_t5(batch):
    list_word = []
    list_atten_mak = []
    list_target_word = []
    list_target_atten_mak = []
    list_feature_idx = []
    max_len = 0
    max_len_tgt = 0
    for idx, sample in enumerate(batch):
        input_ids = sample[0]
        target_ids = sample[2]
        max_len = max(max_len, len(input_ids))
        max_len_tgt = max(max_len_tgt, len(target_ids))

    for idx, sample in enumerate(batch):
        cur_len = len(sample[0])
        cur_len_tgt = len(sample[2])

        input_ids = sample[0] + [0] * (max_len - cur_len)
        list_word.append(input_ids)

        attention_mask = sample[1] + [0] * (max_len - cur_len)
        list_atten_mak.append(attention_mask)

        target_ids = sample[2] + [0] * (max_len_tgt - cur_len_tgt)
        list_target_word.append(target_ids)

        target_attention_mask = sample[3] + [0] * (max_len_tgt - cur_len_tgt)
        list_target_atten_mak.append(target_attention_mask)

        # list_target_word.append(sample[2])
        list_feature_idx.append(sample[4])

    word_tensor = torch.LongTensor(list_word)
    list_atten_mak_tensor = torch.LongTensor(list_atten_mak)
    target_word_tensor = torch.LongTensor(list_target_word)
    target_word_tensor[target_word_tensor == 0] = -100
    list_target_atten_mak_tensor = torch.LongTensor(list_target_atten_mak)
    list_feature_idx_tensor = torch.LongTensor(list_feature_idx)

    return word_tensor, list_atten_mak_tensor, \
           target_word_tensor, list_target_atten_mak_tensor, list_feature_idx_tensor


model_name = './pretrained_model'
model_class = T5ForConditionalGeneration
tokenizer_mame = T5Tokenizer
config_name = T5Config
config = config_name.from_pretrained(model_name,
                                     local_files_only=True)
tokenizer = tokenizer_mame.from_pretrained(model_name,
                                           local_files_only=True)
model = model_class.from_pretrained(model_name, local_files_only=True)
nlp = spacy.load("en_core_web_sm")
