from pytorch_lightning import Trainer
from datamodule import TextDataModule
from pl_model import ModelLSTM
from pytorch_lightning.callbacks import ModelCheckpoint

# Assuming your custom DataModule is named CustomDataModule and ModelLSTM is the model class
# from your previous DataModule and ModelLSTM implementations.

# Step 1: Initialize the DataModule
data_module = TextDataModule(
    config_path="./configs/dataset_config.json",  # Path to your dataset
    tokenizer_path="character_level_tokenizer.json",  # Path to tokenizer
    batch_size=512,  # Customize batch size if needed
)
data_module.setup()
input_dim = data_module.tokenizer.get_vocab_size()
# Step 2: Initialize the LightningModule (the model)
model = ModelLSTM(
    in_dim=input_dim,
    embedding_dim=64,  # Embedding dimension
    hidden_dim=64,  # Hidden dimension for LSTM
    out_dim=1,  # Output dimension - Number of target
    max_len=data_module.max_len[0],
    lr=0.002,
)

checkpoint_callback = ModelCheckpoint(dirpath="./models", save_top_k=1, monitor="val_acc", filename="model")

# Step 3: Initialize the Trainer
trainer = Trainer(
    max_epochs=5,  # Number of epochs
    callbacks=[checkpoint_callback],
)
# Step 4: Train the model
trainer.fit(model, data_module)

# Step 5: (Optional) Test the model after training
trainer.test(model, data_module)
