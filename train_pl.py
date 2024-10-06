import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from datamodule import TextDataModule
from pl_model import ModelLSTM

# Assuming your custom DataModule is named CustomDataModule and ModelLSTM is the model class
# from your previous DataModule and ModelLSTM implementations.

# Step 1: Initialize the DataModule
data_module = TextDataModule(
    data_dir='data/sample/',  # Path to your dataset
    tokenizer_path='character_level_tokenizer.json',  # Path to tokenizer
    batch_size=512  # Customize batch size if needed
)
data_module.setup()

# Step 2: Initialize the LightningModule (the model)
model = ModelLSTM(
    in_dim=40,  # Input dimension (as per your needs)
    embedding_dim=64,  # Embedding dimension
    hidden_dim=64,  # Hidden dimension for LSTM
    out_dim=1,  # Output dimension (automatically set)
    gapped=True,  # If you want to use the gapped setting
    fixed_len=True,  # Use fixed-length sequences
    max_len=data_module.max_len[0],  # Pass max_len from data module
    lr=0.002  # Learning rate
)

# Step 3: Initialize the Trainer
trainer = Trainer(
    max_epochs=10,  # Number of epochs
)

# Step 4: Train the model
trainer.fit(model, data_module)

# Step 5: (Optional) Test the model after training
trainer.test(model, data_module)
