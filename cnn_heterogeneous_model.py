import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNN(nn.Module):
    def __init__(
        self,
        slice_len=4,
        num_feats=18,
        classes=3,
        numChannels=1,
        conv1_out=32,
        conv2_out=64,
        fc1_out=512
    ):
        super(ConvNN, self).__init__()

        self.slice_len = slice_len
        self.num_feats = num_feats
        self.numChannels = numChannels
        self.conv1_out = conv1_out
        self.conv2_out = conv2_out
        self.fc1_out = fc1_out
        self.classes = classes

        self.conv1 = nn.Conv2d(
            in_channels=numChannels,
            out_channels=conv1_out,
            kernel_size=(2, 2),
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(conv1_out)

        self.conv2 = nn.Conv2d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=(2, 2),
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(conv2_out)

        self._to_linear = None
        dummy_input = torch.zeros(1, numChannels, slice_len, num_feats)
        self._forward_conv(dummy_input)

        self.fc1 = nn.Linear(self._to_linear, fc1_out)
        self.fc2 = nn.Linear(fc1_out, classes)

        self.dropout = nn.Dropout(0.5)

    def _forward_conv(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (1, 2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (1, 2))

        if self._to_linear is None:
            self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self._forward_conv(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def build_small_model(slice_len=4, num_feats=18, classes=3):
    return ConvNN(
        slice_len=slice_len,
        num_feats=num_feats,
        classes=classes,
        conv1_out=16,
        conv2_out=32,
        fc1_out=128
    )


def build_medium_model(slice_len=4, num_feats=18, classes=3):
    return ConvNN(
        slice_len=slice_len,
        num_feats=num_feats,
        classes=classes,
        conv1_out=32,
        conv2_out=64,
        fc1_out=256
    )


def build_large_model(slice_len=4, num_feats=18, classes=3):
    return ConvNN(
        slice_len=slice_len,
        num_feats=num_feats,
        classes=classes,
        conv1_out=64,
        conv2_out=128,
        fc1_out=512
    )
