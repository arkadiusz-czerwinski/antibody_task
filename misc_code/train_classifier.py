from ablstm import ModelLSTM
import torch
import datasets
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from functools import partial
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
# initialize model
# change device to 'cpu' if CUDA is not working properly
def compute_text_length(example):
    example['text_length'] = len(example['text'])
    return example

def remove_newlines(texts):
    return [text.replace("\n", "") for text in texts]

def add_special_tokenns(example):
    example['text'] = ["[SOS]" + e+ "[EOS]" for e in example['text']]
    return example

def tokenize_example(example, tokenizer, max_length):
    # Tokenize the text and store token ids under a new key 'input_ids'
    encodings = tokenizer.encode_batch(example['text'])
    example['input_ids'] = [x.ids for x in encodings]
    return example

def build_dataset():
    # Load the three pre-split datasets for label 'b'
    train_b = Dataset.from_text('data/sample/human_train_vlen.txt')
    val_b = Dataset.from_text('data/sample/human_val_vlen.txt')
    test_b = Dataset.from_text('data/sample/human_test_vlen.txt')

    # Load the single dataset with label 'a'
    a_texts = []
    with open('data/sample/mouse_test_vlen.txt', 'r') as f:
        a_texts = f.readlines()

    # Calculate proportions from the label 'b' datasets
    total_b = len(train_b) + len(val_b) + len(test_b)
    _ = len(train_b) / total_b
    val_prop = len(val_b) / total_b
    test_prop = len(test_b) / total_b

    # Split the 'a' dataset into train/val/test based on the proportions
    train_a, temp_a = train_test_split(a_texts, test_size=(val_prop + test_prop))
    val_a, test_a = train_test_split(temp_a, test_size=test_prop / (val_prop + test_prop))

    # Create Dataset objects for label 'a'
    train_a_dataset = Dataset.from_dict({"text": train_a, "label": [0]*len(train_a)})
    val_a_dataset = Dataset.from_dict({"text": val_a, "label": [0]*len(val_a)})
    test_a_dataset = Dataset.from_dict({"text": test_a, "label": [0]*len(test_a)})

    # Append the label 'a' datasets to label 'b' datasets
    train_b = train_b.add_column('label', [1]*len(train_b))  # Assuming label 'b' = 1
    val_b = val_b.add_column('label', [1]*len(val_b))
    test_b = test_b.add_column('label', [1]*len(test_b))

    # Merge train, val, and test datasets for labels 'a' and 'b'
    train_dataset = Dataset.from_dict({
        'text': remove_newlines(train_a + train_b['text']),
        'label': train_a_dataset['label'] + train_b['label']
    })

    val_dataset = Dataset.from_dict({
        'text': remove_newlines(val_a + val_b['text']),
        'label': val_a_dataset['label'] + val_b['label']
    })

    test_dataset = Dataset.from_dict({
        'text': remove_newlines(test_a + test_b['text']),
        'label': test_a_dataset['label'] + test_b['label']
    })

    # Finally, create a DatasetDict to store train/val/test datasets
    dataset = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    })

    return dataset

def train_tokenizer(dataset):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    # Step 1: Prepare data for tokenizer training
    train_texts = dataset['train']['text']
    val_texts = dataset['val']['text']
    test_texts = dataset['test']['text']

    # Combine all texts
    all_texts = train_texts + val_texts + test_texts

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    
    trainer = tokenizer.model.get_trainer()
    trainer.vocab_size = 10000

    trainer.special_tokens = [ "[UNK]", "[PAD]", '[SOS]', '[EOS]' ]
    # Step 5: Train the tokenizer on character level
    tokenizer.train_from_iterator(all_texts, trainer)

    

    # Step 8: Save the tokenizer
    tokenizer.save("character_level_tokenizer.json")
    return tokenizer


if not os.path.isfile("./dataset.hf/dataset_dict.json"):
    dataset = build_dataset()
    tokenizer = train_tokenizer(dataset)

    dataset = dataset.map(compute_text_length)
    # Step 2: Apply the function to each split using map()

    # Step 3: Find the maximum length from all splits
    max_train_length = max(dataset['train']['text_length'])
    max_val_length = max(dataset['val']['text_length'])
    max_test_length = max(dataset['test']['text_length'])

    # Step 4: Find the overall maximum length
    max_text_length = max(max_train_length, max_val_length, max_test_length) + 2

    tokenizer.enable_padding(length=max_text_length)

    tokenize_with_tokenizer = partial(tokenize_example, tokenizer=tokenizer, max_length = max_text_length)
    dataset = dataset.map(add_special_tokenns, batched=True)
    dataset = dataset.map(tokenize_with_tokenizer, batched=True)

    dataset = dataset.shuffle(seed=42)

    dataset.save_to_disk("./dataset.hf")
    tokenizer.save("character_level_tokenizer.json")

dataset = DatasetDict.load_from_disk("./dataset.hf")
tokenizer = Tokenizer.from_file("character_level_tokenizer.json")

max_train_length = max(dataset['train']['text_length'])
max_val_length = max(dataset['val']['text_length'])
max_test_length = max(dataset['test']['text_length'])

# Step 4: Find the overall maximum length
max_text_length = max(max_train_length, max_val_length, max_test_length) + 2

dataset.set_format("torch")
print(dataset)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device="cpu"
in_dim = len(tokenizer.get_vocab())+5
model = ModelLSTM(in_dim = in_dim, embedding_dim=64, hidden_dim=64, device=device, gapped=True, fixed_len=False, out_dim=1, max_len=max_text_length)
print('Model initialized.')
# fit model w/o save
model.fit(dataset=dataset, n_epoch=1, trn_batch_size=128, vld_batch_size=512, lr=.002, save_fp=None)
model.eval(dataset=dataset)
# # fit model w/ save
# model.fit(trn_fn=trn_fn, vld_fn=vld_fn, n_epoch=1, trn_batch_size=128, vld_batch_size=512, lr=.002, save_fp='./saved_models/tmp')
print('Done.')