import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import pudb
from PIL import Image

def split_data(self, data_set, split_ratios=(0.8, 0.1, 0.1)):
    #pudb.set_trace()
    train_split_index = int(len(data_set) * split_ratios[0])
    validation_split_index = train_split_index + int(len(data_set) * split_ratios[1])
    test_split_index = validation_split_index + int(len(data_set) * split_ratios[2])
    return data_set[:train_split_index], data_set[train_split_index:validation_split_index], data_set[test_split_index:]

def subset_data(self, data_set, subset_size=0):
    if subset_size == 0:
        subset_size = len(data_set)
        
    return data_set[:subset_size]

class FFHQDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_vector = self.dataframe.iloc[idx]['image vector']
        image = Image.fromarray(image_vector)
        
        # display image
        image.show()
        
        age_label = self.dataframe.iloc[idx]['age']
        age_label = torch.tensor(age_label, dtype=torch.float32)
        # Assuming 'image vector' column contains NumPy arrays representing images
        
        # Convert image vector to tensor
        #image_tensor = torch.tensor(image_vector, dtype=torch.float32)
        
        # Apply transformation if specified
        if self.transform:
            image_tensor = self.transform(image)

        # drop image vector and age columns from the dataframe
        label_data = self.dataframe.drop(columns=['image vector', 'age'])
        
        # pudb.set_trace()
        label_values = label_data.iloc[idx].values
        #print(binary_label_values)
        label_set = torch.tensor(list(label_values), dtype=torch.float32)

        return image_tensor, age_label, label_set

# Training loop
def train_model(self, model, criterion, optimizer, scheduler, num_epochs=1, save=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    sigmoid = nn.Sigmoid()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels1, labels2 in train_dataloader:
            inputs, labels1, labels2 = inputs.to(device), labels1.to(device), labels2.to(device)

            optimizer.zero_grad()
            
            #outputs1 = model(inputs)
            outputs2 = sigmoid(model(inputs))
            #print(labels2.shape)
            #print(outputs2.shape)
            
            #loss1 = criterion(outputs1, labels1)
            loss2 = criterion(outputs2, labels2)
            
            loss = loss2 
            loss.backward()
            optimizer.step()

            #running_loss1 += loss1.item() * inputs.size(0)
            running_loss += loss2.item() * inputs.size(0)
            
            # Calculate accuracy
            #preds1 = outputs1.data
            preds2 = outputs2.data
        
        #print(preds1)
        print(preds2.round(decimals=1))
        print(labels2)
        
        epoch_loss = running_loss / len(training_input)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    print('Training complete')

    if save:
        torch.save(model.state_dict(), 'model.pth')
        
    return model

### execution starts here
input_data = pickle.load(open('processed_image_data_set_128x128.pkl', 'rb'))
print(input_data.shape)

# take only a subset of the data
input_data = subset_data(input_data)
# split the data into training, validation, and test sets
training_input, validation_input, testing_input = split_data(input_data, (0.2, 0.1, 0.1))
# confirm the shape of the training data
print(training_input.shape)

model = torchvision.models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 9)  # Assuming 9 classes
#model.fc2 = nn.Linear(num_features, 9)  # Emotion prediction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object

transform = transforms.Compose([                      # Convert tensor to PIL image
    transforms.ToTensor(),                          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

train_dataset = FFHQDataset(training_input, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)

# Define loss function and optimizer
numeric_criterion = nn.MSELoss()  # Mean Squared Error
binary_criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(model, numeric_criterion, optimizer, exp_lr_scheduler, num_epochs=3, save=True)

print(model)
