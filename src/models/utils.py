import json
import spacy
from spacy.cli import download
from tqdm import tqdm

import logging
import torch

from dataset import T5Dataset

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S %Z')
logger = logging.getLogger(__name__)


def load_examples(input_doc, filter_name=True, split_doc=False,
                  max_len=100, top_token=False):
    """
    load data from files
    :param top_token: split the doc by number of tokens if True,
    otherwise by number of characters
    :param max_len: maximum number of characters/tokens for
    each splitted paragraph
    :param input_doc: path of the data file
    :param filter_name: True if filter out the Unknown victims
    :param split_doc: true for inference with knowledge fusion
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

    if split_doc:
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

    with open(input_doc) as fin:
        data = json.load(fin)

    for sample in data:
        # initialize class values as None
        date, month, year = None, None, None
        city, region, country = None, None, None
        perpetrator_type = None
        violation_types = None
        str_list = []

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

        if split_doc:
            doc = nlp(context)
            tmp_str = ''
            for sent in doc.sents:
                tmp_str += ' ' + sent.text
                if top_token:
                    if len(tmp_str.split()) > max_len:
                        str_list.append(tmp_str.strip())
                        tmp_str = ''
                else:
                    if len(tmp_str) > max_len:
                        str_list.append(tmp_str.strip())
                        tmp_str = ''
            if tmp_str.strip() != '':
                str_list.append(tmp_str.strip())

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
            'str_list': str_list

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
                         max_len=512, context_filter=False,
                         split_doc=False, top_sentence=None,
                         replicate=False):
    """
    generate features from examples
    :param examples: input examples
    :param tokenizer: tokenizer of PLM backbone
    :param add_prefix: True if add task prefix to T5 input
    :param max_len: maximum length for T5 input
    :param context_filter: check if the input paraphrase contains answer
                           before generate question-answer pairs for T5
    :param split_doc: true for inference with knowledge fusion
    :param top_sentence: the number of the first paraphrases for
    knowledge fusion
    :return: list of features
    """
    violation_all_list = ['arbitrary detention', 'enforced disappearance',
                          'kidnapping', 'killing', 'unknown', 'other',
                          'torture']
    features = []
    for example_id, example in enumerate(tqdm(examples)):
        raw_context = example['context']
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
        if not split_doc:
            context_list = [raw_context]
        else:
            context_list = str_list[:] if not top_sentence else str_list[
                                                                :top_sentence]

        for context in context_list:
            # generate question answer pairs for 'Perpetrator Mention'
            task_prefix = 'detect perpetrator'
            question = 'Does it mention any perpetrator?'
            input_ids, input_masks, truncated_text_1 = \
                input_ids_generate(context, question, task_prefix, tokenizer,
                                   max_len=max_len, add_prefix=add_prefix)
            if replicate:
                _, _, truncated_text_raw = input_ids_generate(
                    raw_context,
                    question,
                    task_prefix,
                    tokenizer,
                    max_len=max_len,
                    add_prefix=add_prefix)

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

            if replicate:
                tmp_context = raw_context
                tmp_truncated_text = truncated_text_raw
            else:
                tmp_context = context
                tmp_truncated_text = truncated_text_1

            for victim in victims:
                if victim[0] and victim[0] in tmp_truncated_text:
                    victim_name = victim[0]

                    # age
                    task_prefix = 'extract victim age'
                    question = 'What is the age group of %s?' % victim_name
                    input_ids, input_masks, truncated_text = \
                        input_ids_generate(tmp_context, question, task_prefix,
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
                        input_ids_generate(tmp_context, question, task_prefix,
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
                        input_ids_generate(tmp_context, question, task_prefix,
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
                        input_ids_generate(tmp_context, question, task_prefix,
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
                        input_ids_generate(tmp_context, question, task_prefix,
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
                        input_ids_generate(tmp_context, question, task_prefix,
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
                    input_ids_generate(tmp_context, question, task_prefix,
                                       tokenizer,
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
                    input_ids_generate(tmp_context, question, task_prefix,
                                       tokenizer,
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
                    input_ids_generate(tmp_context, question, task_prefix,
                                       tokenizer,
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
                    input_ids_generate(tmp_context, question, task_prefix,
                                       tokenizer,
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
                    input_ids_generate(tmp_context, question, task_prefix,
                                       tokenizer,
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
                    input_ids_generate(tmp_context, question, task_prefix,
                                       tokenizer,
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
                    input_ids_generate(tmp_context, question, task_prefix,
                                       tokenizer,
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
    all_feature_idx = []
    for feature_idx, feature in enumerate(data_features):
        all_input_ids.append(feature['input_ids'])
        all_input_masks.append(feature['input_mask'])
        all_output_ids.append(feature['target_ids'])
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

        list_feature_idx.append(sample[3])

    batch_input_ids = torch.LongTensor(list_input_ids)
    batch_input_masks = torch.LongTensor(list_input_masks)
    batch_output_ids = torch.LongTensor(list_output_ids)
    batch_output_ids[batch_output_ids == 0] = -100
    batch_feature_idx = torch.LongTensor(list_feature_idx)

    return batch_input_ids, batch_input_masks, \
           batch_output_ids, batch_feature_idx


def evaluate_dev(dataloader, examples, features, tokenizer, model,
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

                # filter out victim names that are empty
                # or part of a name already predicted
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

            # sys_victims_dic store predictions on victim-dependent classes
            # TODO implement confidence score based knowledge fusion
            #  for victim attributes
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

        # add 'unknown' to violation types if no other label is applicable
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

        # make sure all the predicted victim names are unique,
        # none of them partially match with other predicted names
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
        # print(gold_victim_name_list, sys_pre['victims'])
        # if added_fuzzy > len(sys_pre['victims']) or added_fuzzy > len(
        #         gold_pre['victims']):
        #     print(sys_pre['victims'], gold_pre['victims'], added_fuzzy,
        #           len(sys_pre['victims']),
        #           len(gold_pre['victims']))

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

    # print(gold_count, sys_count, overlap_count, overlap_count_fuzzy)

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


def evaluate_all(dataloader, examples, features, tokenizer, model,
                 gpu=True, num_beams=2):
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
            if num_beams > 1:
                model_outputs = model.generate(input_ids=batch_input_ids,
                                               attention_mask=batch_input_masks,
                                               return_dict_in_generate=True,
                                               output_scores=True,
                                               num_beams=num_beams,
                                               length_penalty=0.0)
            else:
                model_outputs = model.generate(input_ids=batch_input_ids,
                                               attention_mask=batch_input_masks,
                                               return_dict_in_generate=True,
                                               output_scores=True,
                                               num_beams=num_beams)
        preds = model_outputs['sequences']
        if num_beams > 1:
            sequences_scores = model_outputs['sequences_scores']
        for i, example_index in enumerate(batch_feature_idx):
            pre_answer = tokenizer.decode(preds[i], skip_special_tokens=True)
            if num_beams > 1:
                pre_score = sequences_scores[i].cpu().item()

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
                if 'perpetrator_mention' not in fusion_flag_checker[
                    example_id] or (
                        num_beams > 1 and pre_score >
                        fusion_flag_checker[example_id][
                            'perpetrator_mention']):
                    if pre_answer.strip().lower() == 'yes':
                        sys_predictions[example_id][
                            'perpetrator_mention'] = True
                    else:
                        sys_predictions[example_id][
                            'perpetrator_mention'] = False
                    if num_beams > 1:
                        fusion_flag_checker[example_id][
                            'perpetrator_mention'] = pre_score

            if eval_feature[
                'task'] == 'perpetrator_type':
                if 'perpetrator_type' not in fusion_flag_checker[
                    example_id]:

                    sys_predictions[example_id][
                        'perpetrator_type'] = pre_answer.strip()
                    if num_beams > 1:
                        fusion_flag_checker[example_id][
                            'perpetrator_type'] = pre_score
            if eval_feature['task'] == 'city':
                if 'city' not in fusion_flag_checker[
                    example_id]:

                    sys_predictions[example_id]['city'] = pre_answer.strip()
                    if num_beams > 1:
                        fusion_flag_checker[example_id]['city'] = pre_score

            if eval_feature['task'] == 'region':
                if 'region' not in fusion_flag_checker[
                    example_id]:

                    sys_predictions[example_id]['region'] = pre_answer.strip()
                    if num_beams > 1:
                        fusion_flag_checker[example_id]['region'] = pre_score

            if eval_feature['task'] == 'country':
                if 'country' not in fusion_flag_checker[
                    example_id]:

                    sys_predictions[example_id]['country'] = pre_answer.strip()
                    if num_beams > 1:
                        fusion_flag_checker[example_id]['country'] = pre_score

            if eval_feature['task'] == 'date':
                if 'date' not in fusion_flag_checker[example_id]:
                    sys_predictions[example_id]['date'] = pre_answer.strip()
                    if num_beams > 1:
                        fusion_flag_checker[example_id]['date'] = pre_score

            if eval_feature['task'] == 'month':
                if 'month' not in fusion_flag_checker[example_id]:
                    sys_predictions[example_id]['month'] = pre_answer.strip()
                    if num_beams > 1:
                        fusion_flag_checker[example_id]['month'] = pre_score

            if eval_feature['task'] == 'year':
                if 'year' not in fusion_flag_checker[example_id]:
                    sys_predictions[example_id]['year'] = pre_answer.strip()
                    if num_beams > 1:
                        fusion_flag_checker[example_id]['year'] = pre_score

            if eval_feature['task'] == 'victim':
                pre_answer_list = pre_answer.strip().split(',')
                # sys_count += len(victims_label)
                # gold_count += len(pre_answer_list)

                # filter out victim names that are empty
                # or part of a name already predicted
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

            # sys_victims_dic store predictions on victim-dependent classes
            # TODO implement confidence score based knowledge fusion
            #  for victim attributes
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

        # add 'unknown' to violation types if no other label is applicable
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

        # make sure all the predicted victim names are unique,
        # none of them partially match with other predicted names
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
        # print(gold_victim_name_list, sys_pre['victims'])
        # if added_fuzzy > len(sys_pre['victims']) or added_fuzzy > len(
        #         gold_pre['victims']):
        #     print(sys_pre['victims'], gold_pre['victims'], added_fuzzy,
        #           len(sys_pre['victims']),
        #           len(gold_pre['victims']))

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

    # print(gold_count, sys_count, overlap_count, overlap_count_fuzzy)

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
