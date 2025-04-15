
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = self.conv_block(1, 16)    
        self.conv2 = self.conv_block(16, 32)   
        self.conv3 = self.conv_block(32, 64)   
        self.conv4 = self.conv_block(64, 128)  
        self.conv5 = self.conv_block(128, 256) 
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 16 * 16, 512)  # Adjust size according to the final feature map size
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)  # Output: binary classification (0 or 1)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        
        return x
