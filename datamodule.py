import os
from functools import partial
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer

class TextDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./data/sample/", tokenizer_path: str = "character_level_tokenizer.json", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.max_len = None

    def compute_text_length(self, example):
        example['text_length'] = len(example['text'])
        return example

    def remove_newlines(self, texts):
        return [text.replace("\n", "") for text in texts]

    def add_special_tokenns(self, example):
        example['text'] = ["[SOS]" + e+ "[EOS]" for e in example['text']]
        return example

    def tokenize_example(self, example, tokenizer, max_length):
        encodings = tokenizer.encode_batch(example['text'])
        example['input_ids'] = [x.ids for x in encodings]
        return example
    
    def find_duplicates_between_sets(self, dataset_dict, set1_key, set2_key):
        # Get the text lines from both datasets
        set1_lines = set(dataset_dict[set1_key]['text'])
        set2_lines = set(dataset_dict[set2_key]['text'])
        
        # Find the common lines (duplicates) between the two sets
        duplicates = set1_lines.intersection(set2_lines)
        
        if duplicates:
            print(f"Duplicates between {set1_key} and {set2_key}:")
            for line in duplicates:
                print(line.strip())
        else:
            print(f"No duplicates found between {set1_key} and {set2_key}.")
        
        return duplicates
    
    def build_dataset(self):
        # Load datasets for label 'b'
        train_b = Dataset.from_text(os.path.join(self.data_dir, 'human_train_vlen.txt'))
        val_b = Dataset.from_text(os.path.join(self.data_dir, 'human_val_vlen.txt'))
        test_b = Dataset.from_text(os.path.join(self.data_dir, 'human_test_vlen.txt'))

        # Load dataset for label 'a'
        with open(os.path.join(self.data_dir, 'mouse_test_vlen.txt'), 'r') as f:
            a_texts = f.readlines()
            a_texts = list(set(a_texts))

        # Split label 'a' into train/val/test
        total_b = len(train_b) + len(val_b) + len(test_b)
        val_prop = len(val_b) / total_b
        test_prop = len(test_b) / total_b

        train_a, temp_a = train_test_split(a_texts, test_size=(val_prop + test_prop))
        val_a, test_a = train_test_split(temp_a, test_size=test_prop / (val_prop + test_prop))

        # Create Dataset objects for label 'a'
        train_a_dataset = Dataset.from_dict({"text": train_a, "label": [0]*len(train_a)})
        val_a_dataset = Dataset.from_dict({"text": val_a, "label": [0]*len(val_a)})
        test_a_dataset = Dataset.from_dict({"text": test_a, "label": [0]*len(test_a)})

        # Add label column for 'b' datasets
        train_b = train_b.add_column('label', [1]*len(train_b))
        val_b = val_b.add_column('label', [1]*len(val_b))
        test_b = test_b.add_column('label', [1]*len(test_b))

        # Merge datasets for label 'a' and 'b'
        train_dataset = Dataset.from_dict({
            'text': self.remove_newlines(train_a + train_b['text']),
            'label': train_a_dataset['label'] + train_b['label']
        })

        val_dataset = Dataset.from_dict({
            'text': self.remove_newlines(val_a + val_b['text']),
            'label': val_a_dataset['label'] + val_b['label']
        })

        test_dataset = Dataset.from_dict({
            'text': self.remove_newlines(test_a + test_b['text']),
            'label': test_a_dataset['label'] + test_b['label']
        })

        return DatasetDict({
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        })

    def train_tokenizer(self, dataset):
        # Tokenizer training code (same as before)
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        # Prepare data for tokenizer training
        train_texts = dataset['train']['text']
        val_texts = dataset['val']['text']
        test_texts = dataset['test']['text']

        # Combine all texts
        all_texts = train_texts + val_texts + test_texts

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")

        trainer = tokenizer.model.get_trainer()
        trainer.vocab_size = 10000
        trainer.special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]

        # Train the tokenizer
        tokenizer.train_from_iterator(all_texts, trainer)
        tokenizer.save(self.tokenizer_path)
        return tokenizer

    def prepare_data(self):
        if not os.path.isfile("./dataset.hf/dataset_dict.json"):
            dataset = self.build_dataset()
            duplicates_train_test = self.find_duplicates_between_sets(dataset, 'train', 'test')
            duplicates_train_val = self.find_duplicates_between_sets(dataset, 'train', 'val')
            duplicates_val_test = self.find_duplicates_between_sets(dataset, 'val', 'test')

            dataset['val'] = dataset['val'].filter(lambda example: example['text'] not in duplicates_train_val)
            dataset['test'] = dataset['test'].filter(lambda example: example['text'] not in duplicates_train_test)
            dataset['test'] = dataset['test'].filter(lambda example: example['text'] not in duplicates_val_test)
            
            tokenizer = self.train_tokenizer(dataset)

            dataset = dataset.map(self.compute_text_length)

            max_train_length = max(dataset['train']['text_length'])
            max_val_length = max(dataset['val']['text_length'])
            max_test_length = max(dataset['test']['text_length'])
            max_text_length = max(max_train_length, max_val_length, max_test_length) + 2

            tokenizer.enable_padding(length=max_text_length)

            tokenize_with_tokenizer = partial(self.tokenize_example, tokenizer=tokenizer, max_length=max_text_length)
            dataset = dataset.map(self.add_special_tokenns, batched=True)
            dataset = dataset.map(tokenize_with_tokenizer, batched=True)

            dataset = dataset.shuffle(seed=42)
            dataset['metadata'] =  Dataset.from_dict({
                "max_len": [max_text_length]
            })
            dataset.save_to_disk("./dataset.hf")
            tokenizer.save(self.tokenizer_path)

    def setup(self, stage=None):
        # Load the dataset from disk
        self.prepare_data()
        self.dataset = DatasetDict.load_from_disk("./dataset.hf")

        self.max_len = self.dataset['metadata']['max_len']

        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)

        # Set data format
        self.dataset.set_format("torch")

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset['val'], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.batch_size)
