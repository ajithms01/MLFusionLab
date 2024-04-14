import torch
import os
import gc
from torch import nn


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

# Global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ImageFolder.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def create_train_data():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225],)
    ])
    # Create training data
    train_data = ImageFolder('data/train',transform=train_transform)
    
    train_loader = DataLoader(train_data, batch_size= 32)

    return train_loader

def create_test_data():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225],)
    ])
    # Creating Test data
    test_data = ImageFolder('data/test', transform=test_transform)

    test_loader = DataLoader(test_data, batch_size = 32)

    return test_loader


def train_step(model : nn.Module, train_data : DataLoader, loss_fn:torch.nn.Module, optimizer: torch.optim.Optimizer):

    train_loss, train_acc = 0.0, 0.0

    # Setting model to GPU
    model.to(device)

    # Converting model to training mode
    model.train()

    for batch_idx, (data, target) in enumerate(train_data):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        output_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        train_acc += (output_class == target).sum().item() / len(output)

    train_loss /= len(train_data)
    train_acc /= len(train_data)

    gc.collect()
    torch.cuda.empty_cache()

    return train_loss, train_acc

def test_step(model, test_data, loss_fn):

    test_acc, test_loss = 0.0, 0.0

    model.eval()

    with torch.inference_mode():
        for batch_idx, (data, target) in enumerate(test_data):

            data, target = data.to(device), target.to(device)

            output_logits = model(data)
            loss = loss_fn(output_logits, target)
            test_loss += loss.item()

            output_labels = output_logits.argmax(dim=1)
            test_acc += ((output_labels == target).sum().item()/len(output_labels))
    
    test_loss = test_loss / len(test_data)
    test_acc = test_acc / len(test_data)

    gc.collect()
    torch.cuda.empty_cache()

    return test_loss, test_acc

def train_models(model : nn.Module , train_data : DataLoader, test_data: DataLoader, epochs: int, save_name: str):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    metrics = { 'train loss': 0.0, 'train accuracy': 0.0, 'test loss': 0.0, 'test accuracy': 0.0 }

    # Training Model on each epoch
    for epoch in range(epochs):

        # Training Step
        metrics['train loss'], metrics['train accuracy'] = train_step(model=model, train_data=train_data, loss_fn=loss_fn, optimizer=optimizer)
        
        # Testing Step
        metrics['test loss'], metrics['test accuracy'] = test_step(model=model, test_data=test_data, loss_fn=loss_fn)

        model_path = os.path.join('models/', save_name +".pth")
        torch.save(model.state_dict(), model_path)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return metrics