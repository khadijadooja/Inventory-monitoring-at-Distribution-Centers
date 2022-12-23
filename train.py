#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from smdebug.profiler.utils import str2bool
import smdebug.pytorch as smd
import argparse
import logging
import sys
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader,criterion,device,hook):
    
    #hook = get_hook(create_if_not_exists=True)
    print("Testing Model on all Testing Dataset")
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    running_loss=0
    running_corrects=0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    logger.info(f"Test set: Average loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    
    pass

def train(model, train_loader,validation_loader, criterion,epochs, optimizer,device,hook):
    hook.set_mode(smd.modes.TRAIN)
    print("training Model on Dataset")
    logger.info("Start Model Training")
    
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                
                model.train()
                hook.set_mode(smd.modes.TRAIN)
                
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
                
            running_loss = 0.0
            running_corrects = 0
            
            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
              
            epoch_loss = running_loss / len(image_dataset[phase])
            epoch_acc = running_corrects / len(image_dataset[phase])
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase, epoch_loss, epoch_acc, best_loss))

        if loss_counter==1:
            break
    return model

    
def net():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    num_features=model.fc.in_features
    
    model.fc = nn.Sequential(nn.Linear(num_features, 128),
                             nn.ReLU(inplace=True),
                             nn.Linear(128, 5))
    
    return model

def create_data_loaders(data, batch_size):
    
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset = datasets.ImageFolder(data, transform=data_transform)

    total = len(dataset)
    train_length = int(np.ceil(0.7 * total))
    test_length = int(np.floor(0.15 * total))
    valid_length = int(np.floor(0.15 * total))
    
    lengths = [train_length, test_length, valid_length]

    trainset, testset, validset = torch.utils.data.random_split(dataset, lengths)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size) 
        
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size) 
    
    return trainloader,testloader,validloader


def main(args):
    
    logger.info(f'Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.train}')
    
    model=net()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    
    model=model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    hook.register_loss(loss_criterion)
    
    train_loader,test_loader,validation_loader=create_data_loaders(args.train, args.batch_size)
    
    logger.info("start training")  
    
    model=train(model, train_loader, validation_loader,loss_criterion,args.epochs, optimizer,device,hook)
    
    logger.info("start testing")
    test(model, test_loader, loss_criterion,device,hook)
    
    logger.info("Starting to Save the Model")
    torch.save(model.state_dict(),os.path.join(args.model_dir, "model.pth"))
    logger.info("Completed Saving the Model")

if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--model_dir", type=str, default=os.environ['SM_MODEL_DIR']
    )
    
    parser.add_argument(
        "--epochs", type=int, default=6
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=32
    )
    parser.add_argument(
        "--train", type=str, default=os.environ['SM_CHANNEL_TRAIN']
    )
        # Container environment
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    
    args=parser.parse_args()
    
    main(args)
