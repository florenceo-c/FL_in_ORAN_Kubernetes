import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNN(nn.Module):
    def __init__(self, slice_len=4, num_feats=18, classes=3, numChannels=1):
        super(ConvNN, self).__init__()
        
        # Calculate input size: (Batch, 1, Slice_Len, Num_Feats)
        # We process the "image" of (Slice_Len x Num_Feats)
        
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=32, kernel_size=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Calculate the size after convolutions to determine Fully Connected input
        # We use a dummy forward pass to figure this out automatically
        self._to_linear = None
        dummy_input = torch.zeros(1, numChannels, slice_len, num_feats)
        self._forward_conv(dummy_input)
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, classes)
        
        self.dropout = nn.Dropout(0.5)

    def _forward_conv(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (1, 2)) # Pool over features
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (1, 2))
        
        if self._to_linear is None:
            self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
        return x

    def forward(self, x):
        # x shape comes in as [Batch, Slice_Len, Num_Feats]
        # We need to add the Channel dimension: [Batch, 1, Slice_Len, Num_Feats]
        x = x.unsqueeze(1) 
        
        x = self._forward_conv(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) # Log Softmax for NLLLoss
