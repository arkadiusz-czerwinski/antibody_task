import os
import json
from functools import partial
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from typing import List
from datasets.formatting.formatting import LazyRow


class TextDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer_path: str = "character_level_tokenizer.json",
        config_path: str = "./cofigs/dataset_config.json",
        batch_size: int = 32,
    ):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.max_len = None
        with open(config_path, 'r') as file:
            self.config = json.load(file)

    def compute_text_length(self, example: LazyRow) -> LazyRow:
        """Calculating number of characters in the text"""
        example["text_length"] = len(example["text"])
        return example

    def remove_newlines(self, texts: List[str]) -> List[str]:
        """Remove newlines in the split text file"""
        return [text.replace("\n", "") for text in texts]

    def add_special_tokenns(self, example: LazyRow) -> LazyRow:
        """Add start of the sentence and end of the sentence tokens to the sequence"""
        example["text"] = ["[SOS]" + e + "[EOS]" for e in example["text"]]
        return example

    def tokenize_example(self, example: LazyRow, tokenizer: Tokenizer) -> LazyRow:
        """Add tokenization of the text to each of the texts from the dataset"""
        encodings = tokenizer.encode_batch(example["text"])
        example["input_ids"] = [x.ids for x in encodings]
        return example

    def find_duplicates_between_sets(
        self, dataset_dict: DatasetDict, set1_key: str, set2_key: str
    ) -> set:
        """Find duplicates between 2 different dataset splits"""
        # Get the text lines from both datasets
        set1_lines = set(dataset_dict[set1_key]["text"])
        set2_lines = set(dataset_dict[set2_key]["text"])

        # Find the common lines (duplicates) between the two sets
        duplicates = set1_lines.intersection(set2_lines)

        if duplicates:
            print(f"Duplicates between {set1_key} and {set2_key}:")
            for line in duplicates:
                print(line.strip())
        else:
            print(f"No duplicates found between {set1_key} and {set2_key}.")

        return duplicates

    def build_dataset(self) -> DatasetDict:
        """Build dataset for the antibody classification
        TODO: Add config for the dynamic pathing"""
        # Load datasets for label 'b'
        train_human = Dataset.from_text(os.path.join(self.config['data_dir'], self.config['human_train_filename']))
        val_human = Dataset.from_text(os.path.join(self.config['data_dir'], self.config['human_val_filename']))
        test_human = Dataset.from_text(os.path.join(self.config['data_dir'], self.config['human_test_filename']))

        # Load dataset for label 'a'
        with open(os.path.join(self.config['data_dir'], self.config['mouse_filename']), "r") as f:
            mouse_texts = f.readlines()
            mouse_texts = list(set(mouse_texts))

        # Split label 'a' into train/val/test
        total_human = len(train_human) + len(val_human) + len(test_human)
        val_prop = len(val_human) / total_human
        test_prop = len(test_human) / total_human

        train_mouse, temp_mouse = train_test_split(mouse_texts, test_size=(val_prop + test_prop))
        val_mouse, test_mouse = train_test_split(
            temp_mouse, test_size=test_prop / (val_prop + test_prop)
        )

        # Create Dataset objects for label 'a'
        train_a_dataset = Dataset.from_dict(
            {"text": train_mouse, "label": [0] * len(train_mouse)}
        )
        val_a_dataset = Dataset.from_dict({"text": val_mouse, "label": [0] * len(val_mouse)})
        test_a_dataset = Dataset.from_dict({"text": test_mouse, "label": [0] * len(test_mouse)})

        # Add label column for 'b' datasets
        train_human = train_human.add_column("label", [1] * len(train_human))
        val_human = val_human.add_column("label", [1] * len(val_human))
        test_human = test_human.add_column("label", [1] * len(test_human))

        # Merge datasets for label 'a' and 'b'
        train_dataset = Dataset.from_dict(
            {
                "text": self.remove_newlines(train_mouse + train_human["text"]),
                "label": train_a_dataset["label"] + train_human["label"],
            }
        )

        val_dataset = Dataset.from_dict(
            {
                "text": self.remove_newlines(val_mouse + val_human["text"]),
                "label": val_a_dataset["label"] + val_human["label"],
            }
        )

        test_dataset = Dataset.from_dict(
            {
                "text": self.remove_newlines(test_mouse + test_human["text"]),
                "label": test_a_dataset["label"] + test_human["label"],
            }
        )

        return DatasetDict(
            {"train": train_dataset, "val": val_dataset, "test": test_dataset}
        )

    def train_tokenizer(self, dataset: DatasetDict) -> Tokenizer:
        """Train tokenizer on character level"""
        # Tokenizer training code (same as before)
        from tokenizers import Tokenizer, models, pre_tokenizers

        # Prepare data for tokenizer training
        train_texts = dataset["train"]["text"]
        val_texts = dataset["val"]["text"]
        test_texts = dataset["test"]["text"]

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
        """Build dataset for the classification if it does not exists"""
        if not os.path.isfile("./dataset.hf/dataset_dict.json"):
            dataset = self.build_dataset()
            duplicates_train_test = self.find_duplicates_between_sets(
                dataset, "train", "test"
            )
            duplicates_train_val = self.find_duplicates_between_sets(
                dataset, "train", "val"
            )
            duplicates_val_test = self.find_duplicates_between_sets(
                dataset, "val", "test"
            )

            dataset["val"] = dataset["val"].filter(
                lambda example: example["text"] not in duplicates_train_val
            )
            dataset["test"] = dataset["test"].filter(
                lambda example: example["text"] not in duplicates_train_test
            )
            dataset["test"] = dataset["test"].filter(
                lambda example: example["text"] not in duplicates_val_test
            )

            tokenizer = self.train_tokenizer(dataset)

            dataset = dataset.map(self.compute_text_length)

            max_train_length = max(dataset["train"]["text_length"])
            max_val_length = max(dataset["val"]["text_length"])
            max_test_length = max(dataset["test"]["text_length"])
            max_text_length = max(max_train_length, max_val_length, max_test_length) + 2

            tokenizer.enable_padding(length=max_text_length)

            tokenize_with_tokenizer = partial(
                self.tokenize_example, tokenizer=tokenizer
            )
            dataset = dataset.map(self.add_special_tokenns, batched=True)
            dataset = dataset.map(tokenize_with_tokenizer, batched=True)

            dataset = dataset.shuffle(seed=42)
            dataset["metadata"] = Dataset.from_dict({"max_len": [max_text_length]})
            dataset.save_to_disk("./dataset.hf")
            tokenizer.save(self.tokenizer_path)

    def setup(self, stage=None):
        # Load the dataset from disk
        self.prepare_data()
        self.dataset = DatasetDict.load_from_disk("./dataset.hf")

        self.max_len = self.dataset["metadata"]["max_len"]

        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)

        # Set data format
        self.dataset.set_format("torch")

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"], batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.batch_size)
