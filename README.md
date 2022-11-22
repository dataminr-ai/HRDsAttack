# UN-OHCHR-Event-Extraction
The code to train and evaluate the model on Human Rights Defenders dataset.

# Requirements
Install the dependencies:

- `python3.8+`
- `pip3 install -r requirements.txt`

Download the Spacy English model:
```
python -m spacy download en_core_web_sm
```

# Training
All of the experiments were conducted on a single AWS g5.xlarge	machine, which is based on a NVIDIA A10G GPU. To train the model on different machines, the batch size should be modified to fit the GPU memory size.
```
python src/models/few_shot_t5.py \
       --train_file src/data/train.json \
       --dev_file src/data/dev.json \
       --test_file src/data/dev.json \
       --lr 4e-5 \
       --lr_decay 1e-5 \
       --epoch 20 \
       --batch_size 4 \
       --eval_per_epoch 3 \
       --gpu 0 \
       --gradient_accumulation_steps 16 \
       --model_name t5-large-prefix-average-metric \
       --dataset_name v1.0 \
       --eval_metric average \
       --add_prefix \
       --use_metric
```
