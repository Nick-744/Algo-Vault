import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')



def get_train_valid_loader(data_dir:   str,
                           batch_size: int,
                           augment:    bool,
                           valid_size: float = 0.1,
                           shuffle:    bool  = True):
    normalize = transforms.Normalize(
        mean = [0.4914, 0.4822, 0.4465],
        std  = [0.2023, 0.1994, 0.2010]
    )
    # Πως προέκυψαν αυτοί οι αριθμοί; Διαφοροποιούνται από dataset σε dataset; Τι σημαίνουν;
    # https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data

    valid_transform = transforms.Compose([
            transforms.Resize((227, 227)), # Γιατί όχι 224x224;
            # Το σχήμα του paper δεν σημφωνεί με τα μαθηματικά του...
            # Παρουσιάζεται πρόβλημα με τις διαστάσεις στους πολλαπλασιασμούς πινάκων!
            transforms.ToTensor(),
            normalize
    ])

    # Data augmentation
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root      = data_dir,
        train     = True,
        download  = True,
        transform = train_transform
    )

    valid_dataset = datasets.CIFAR10(
        root      = data_dir,
        train     = True,
        download  = True,
        transform = valid_transform
    )

    num_train = len(train_dataset)
    indices   = list(range(num_train))
    split     = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.shuffle(indices)

    (train_idx, valid_idx) = (indices[split:], indices[:split])
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        sampler    = train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = batch_size,
        sampler    = valid_sampler
    )

    return (train_loader, valid_loader);

def get_test_loader(data_dir:   str,
                    batch_size: int,
                    shuffle:    bool = True):
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.CIFAR10(
        root      = data_dir,
        train     = False,
        download  = True,
        transform = transform
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle    = shuffle
    )

    return data_loader;

# CIFAR10 dataset 
(train_loader, valid_loader) = get_train_valid_loader(data_dir = './data', batch_size = 64, augment = False)

test_loader = get_test_loader(data_dir = './data', batch_size = 64)



class AlexNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(AlexNet, self).__init__() # Επειδή κληρονομεί από nn.Module

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 0), # 96 φίλτρα βάση αρχιτεκτονικής
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        ) # W_output = floor((W_input + 2 * padding - kernel_size) / stride) + 1
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(nn.Linear(4096, num_classes))

        return;

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.flatten(1)

        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out;

num_classes   = 10
num_epochs    = 20
batch_size    = 64
learning_rate = 0.005

model = AlexNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.005, momentum = 0.9)  



# Train!!! ~30 λεπτά κάνει στο Google Colab με GPU
total_step = len(train_loader)
for epoch in range(num_epochs):
    for (i, (images, labels)) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss    = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

    # Validation
    with torch.no_grad():
        correct = 0
        total   = 0
        for (images, labels) in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs        = model(images)
            (_, predicted) = torch.max(outputs.data, 1) # Το μοντέλο επιστρέφει πιθανότητες για κάθε κλάση!
            # Έτσι, παίρνουμε την κλάση με τη μέγιστη πιθανότητα.

            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

            del images, labels, outputs;

        print(f'Accuracy of the network on the {5000} validation images: {100 * correct / total} %')



with torch.no_grad(): # Λέμε στο Torch να μην κρατάει γράφους για τον υπολογισμό των παραγώγων (backpropagation)!
    correct = 0
    total   = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs        = model(images)
        (_, predicted) = torch.max(outputs.data, 1)

        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

        del images, labels, outputs;

    print(f'Accuracy of the network on the {10000} test images: {100 * correct / total} %')
    # Βγαίνει ~80% το αποτέλεσμα.
