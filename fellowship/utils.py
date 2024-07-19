import numpy as np
import torch
import torch.nn as nn
from fastai.vision.all import *
import gc
import cv2
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()
    
class FlowerDataset_v2(Dataset):
    ''' To increase the length of dataset with augmentations'''
    def __init__(self, im_pths, targets, num_augmentations, transform = None, target_transform = None):
        self.im_pths = im_pths
        self.targets = targets
        self.num_augmentations = num_augmentations
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.im_pths)*self.num_augmentations
        
    def __getitem__(self, idx):
        actual_idx = idx // self.num_augmentations
        augmentation_idx = idx % self.num_augmentations
        
        im_pth = str(self.im_pths[actual_idx])
        image = Image.fromarray(cv2.cvtColor(cv2.imread(im_pth), cv2.COLOR_BGR2RGB))
        label = self.targets[actual_idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

  
   
def class_distribution_dataloader(dataloader):
    class_counts = Counter()
    for _, labels in dataloader:
        class_counts.update(labels.numpy())
    return class_counts


def plot_train_val_batches(dls, invTrans=None, num_batches = 3):
    
    for dl in dls:
        for b, (ims, labels)  in enumerate(dl):
            if invTrans:
                ims_t = [invTrans(o) for o in ims]
                titles = [o.item() for o in labels]
            else:
                ims_t = ims
                titles = [o.item() for o in labels]

            show_image_batch([ims_t, titles], items=16, cols=16, figsize=(10,1))
            if b==num_batches-1:
                break


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): Function to trace print statements.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), self.path)
        torch.save(model, self.path)
        self.val_loss_min = val_loss

def plot_lr_schedule(base_lr, max_lr, step_size_up, num_epochs, num_minibatches_per_epoch, mode = 'exp_range'):
    
    # Dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    # CyclicLR scheduler
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=base_lr, 
        max_lr=max_lr, 
        step_size_up=step_size_up, 
        mode=mode, 
        gamma = 0.9994,
        cycle_momentum=True, 
        base_momentum=0.8, 
        max_momentum=0.9
    )
    
    # Simulation parameters
    num_epochs = num_epochs
    num_minibatches_per_epoch = num_minibatches_per_epoch
    
    # Store LR values
    lrs = []
    
    # Simulate the training process
    for epoch in range(num_epochs):
        for batch in range(num_minibatches_per_epoch):
            # Simulate a training step
            optimizer.step()
            # Record the current LR
            lrs.append(optimizer.param_groups[0]['lr'])
            # Update the scheduler
            scheduler.step()
    
    # Plot the LR values
    plt.plot(lrs)
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('CyclicLR Learning Rate Schedule')
    plt.show()

def train_supervised(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, tag, device):
    
    early_stopping = EarlyStopping(patience=patience, path = f'{tag}.pth', verbose=True)
    best_val_accuracy = 0.0

    for epoch in range(1, epochs+1):
    
        model.train()
        train_loss = 0.0
        train_corrects = 0
        start = time.time()
    
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
            train_loss += loss.item() * inputs.size(0)
            train_corrects += (logits.argmax(1) == labels).type(torch.float).sum().item()
    
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_corrects / len(train_loader.dataset)
    
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
    
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += (logits.argmax(1) == labels).type(torch.float).sum().item()
    
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_corrects / len(val_loader.dataset)
        end = time.time()
        
        early_stopping(val_loss, model)
        
        print(f"Epoch {epoch}/{epochs}  Time: {end-start} sec")
        print(f"Training Loss: {train_loss:.4f},   Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print()
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
        # Save the best model - saving all models
        ##if val_accuracy > best_val_accuracy:
            #best_val_accuracy = val_accuracy
            ##torch.save(model.state_dict(), f"{tag}_{epoch}.pth")
            
def test_model(model, test_dl, criterion):
    model.eval()
    test_loss = 0.0
    test_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += (logits.argmax(1) == labels).type(torch.float).sum().item()

    test_loss = test_loss / len(test_dl.dataset)
    test_accuracy = test_corrects / len(test_dl.dataset)
    end = time.time()

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")