import numpy as np
import torch
from pytorch_lightning import LightningModule
from bi_lstm import Bi_LSTM
from tqdm import tqdm
from torchmetrics.classification import Accuracy, F1Score, ConfusionMatrix

class ModelLSTM(LightningModule):
    def __init__(self, in_dim=40, embedding_dim=64, hidden_dim=64, out_dim=1, gapped=True, fixed_len=True, max_len=131, lr=0.002):
        super().__init__()
        self.gapped = gapped
        self.lr = lr
        self.max_len = max_len

        # Set output dimension based on `gapped` if not provided
        self.out_dim = out_dim

        # Define the LSTM model
        self.nn = Bi_LSTM(in_dim, embedding_dim, hidden_dim, out_dim=out_dim, fixed_len=fixed_len, max_len=max_len)
        self.loss_fn = torch.nn.BCELoss()  # Binary Cross Entropy loss
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.confusion_matrix = ConfusionMatrix(task="binary", num_classes=2)

    def forward(self, X):
        """Forward pass."""
        return self.nn(X).squeeze()

    def training_step(self, batch, batch_idx):
        """Training step."""
        X = batch['input_ids']
        y = batch['label'].float()  # Convert to float for BCE Loss

        # Forward pass
        preds = self(X)

        # Calculate loss
        loss = self.loss_fn(preds, y)

        # Calculate accuracy
        # Log accuracy and F1 score for training step
        acc = self.train_acc(preds, y.int())
        f1 = self.train_f1(preds, y.int())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_f1', f1, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X = batch['input_ids']
        y = batch['label'].float()

        # Forward pass
        preds = self(X)

        # Calculate loss
        loss = self.loss_fn(preds, y)
        # Log accuracy and F1 score for validation step
        acc = self.val_acc(preds, y.int())
        f1 = self.val_f1(preds, y.int())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)

        # Confusion matrix (optional)
        cm = self.confusion_matrix(preds, y.int())

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        X = batch['input_ids']
        y = batch['label'].float()

        # Forward pass
        scores = self(X)

        # Calculate accuracy
        predicted = (scores > 0.5).long().flatten()
        acc = (predicted == y.flatten()).float().mean()

        # Log test accuracy
        self.log('test_acc', acc)

        return acc
    
    def on_validation_epoch_end(self):
        # Print the confusion matrix at the end of each validation epoch
        cm = self.confusion_matrix.compute()
        print(f"Confusion Matrix:\n{cm}")
        self.confusion_matrix.reset()

    def configure_optimizers(self):
        """Set up the optimizer."""
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.lr)
        return optimizer

    def save(self, fn):
        """Save the model parameters."""
        param_dict = self.nn.get_param()
        param_dict['gapped'] = self.gapped
        np.save(fn, param_dict)

    def load(self, fn):
        """Load the model parameters."""
        param_dict = np.load(fn, allow_pickle=True).item()
        self.gapped = param_dict['gapped']
        self.nn.set_param(param_dict)

    def summary(self):
        """Print a summary of the model."""
        for n, w in self.nn.named_parameters():
            print('{}:\t{}'.format(n, w.shape))
        print('Fixed Length:\t{}'.format(self.nn.fixed_len))
        print('Gapped:\t{}'.format(self.gapped))
        print('Device:\t{}'.format(self.device))
