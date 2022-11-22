# UN-OHCHR-Event-Extraction
The code to train and evaluate the model on Human Rights Defenders dataset.

# Requirements
- `python3.7+`
- `pip3 install -r requirements.txt`

# data
first create a data sub-directory under the root directory
```
mkdir data
```
put the downloaded data files (train.json, dev.json) into the data sub-directory

# Training
```
python src/models/few_shot_t5.py --train_file ./data/train.json --dev_file ./data/dev.json
   --lr 4e-5 --lr_decay 1e-5 --epoch 20 --batch_size 4
   --eval_per_epoch 3 --gpu 0 --gradient_accumulation_steps 16
   --model_name t5-large-prefix-average-metric --dataset_name v1.0 --eval_metric average --add_prefix
   --use_metric
```

# Evaluate
