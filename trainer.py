
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from model import EarlyStopping, set_seed
from tqdm import tqdm

# hyperparams = [CONFIG['MODEL'], CONFIG['NUM_CLASSES'], CONFIG['BATCH_SIZE'], CONFIG['NUM_EPOCHS'], CONFIG['LEARNING_RATE']]
SEED = 2021
set_seed(seed=SEED)

class ModelTrainer(object):

    def __init__(self, model, config, train_dataset, valid_dataset, early_stopping=EarlyStopping(), test_dataset=None,
                 savepath=None, num_seed=42):     
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        
        self.learning_rate = config['LEARNING_RATE']
        self.n_epochs = config['NUM_EPOCHS']
        
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
        self._num_seed = num_seed
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.early_stopping = EarlyStopping(patience=5, delta=0.01, verbose=True)

        self.savepath = savepath

    def train_eval(self, get_pred=False): #np.random.randint(1000)):
        print(f"seed used for training: {self._num_seed}")
        # out_dir = self.modelpath.joinpath(f"{self.model_name}-{self._lang_to_train}-finetuned")
        # print(f"Model to be saved in {out_dir}")
        
        for epoch in range(self.n_epochs):
            # training loop
            with tqdm(self._train_dataset, unit="batch", bar_format='\n{l_bar}{bar:20}{r_bar}') as tepoch:
                train_loss = 0.0

                self.model.train()
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    print(f"Class distribution in training: {labels.unique(return_counts=True)} ")
                    
                    inputs, labels = inputs.type(torch.FloatTensor).to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                    tepoch.set_postfix(loss=loss.item())
                epoch_loss = train_loss / len(self._train_dataset)
                print(f'Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.4f}', end='\n ', flush=True)
                
                # Validation loop
                valid_loss = 0.0

                self.model.eval()
                correct = total = 0
                y = torch.tensor([], dtype=torch.long, device=self.device)
                y_hat = torch.tensor([], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    for inputs_val, labels_val in self._valid_dataset:
                        print(f"Class distribution in validation: {labels_val.unique(return_counts=True)} ")
                        
                        inputs_val, labels_val = inputs_val.type(torch.FloatTensor).to(self.device), labels_val.to(self.device)
                        outputs_val = self.model(inputs_val)
                        loss_val = self.criterion(outputs_val, labels_val)
                        valid_loss += loss_val.item()
                        
                        _, predicted = torch.max(outputs_val, 1)
                        total += labels_val.size(0)
                        correct += (predicted == labels_val).sum().item()
                        conf_matrix = confusion_matrix(labels_val, predicted)
                        print(conf_matrix)
                        y_hat = torch.cat((y_hat, predicted), 0)
                        y = torch.cat((y, labels_val), 0)

                # Average validation loss
                valid_loss /= len(self._valid_dataset)
                print(f'Validation Loss: {valid_loss:.4f}')
                val_accuracy = correct / total
                print(f'Validation Accuracy: {val_accuracy:.4f}')

                # Check early stopping condition
                self.early_stopping.check_early_stop(valid_loss)
                if self.early_stopping.stop_training:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # add metrics    
        y = y.cpu().numpy()  
        y_hat = y_hat.cpu().numpy()
        print("\nConfusion Matrix (final):")
        conf_matrix = confusion_matrix(y, y_hat)
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y, y_hat))
        # print(f"Macro F1-score: {round(f1_score(y, y_hat, average='macro'),2)}")
        # print(f"Weighted F1-score: {round(f1_score(y, y_hat, average='weighted'),2)}")
        print("\n")

        torch.save(self.model.state_dict(), self.savepath)
        # print(f"free space by deleting: {self.savepath}")
        # shutil.rmtree(self.savepath, ignore_errors=True)
        
        return self.model

