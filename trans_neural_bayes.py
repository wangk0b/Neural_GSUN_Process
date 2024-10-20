import torch
import torch.nn as nn
#graphical neural bayes CNN + Transformer
import torch.nn.functional as F
from GAT_layer import GraphAttentionLayer

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),nn.BatchNorm2d(outchannel),nn.ReLU(inplace=True),nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(outchannel))
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)                                                                                                       
        return out


class TC_Neural_Bayes(nn.Module):
    def __init__(self, ResidualBlock):
        super(TC_Neural_Bayes, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.inchannel),
                nn.ReLU()
                )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        encoder = nn.TransformerEncoderLayer(d_model = 512,nhead = 8)
        self.layer5 = nn.TransformerEncoder(encoder, num_layers=12,)
        #self.layer6 = nn.linear()
        self.fc = nn.Linear(512*4, 5)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
          layers.append(block(self.inchannel, channels, stride))
          self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.mean(out,dim = 0)
        #out = out.view(-1,512)
        #print(out.shape)
        #out = self.layer5(out)
        #print("One success!")
        out = out.flatten()
        out = self.fc(out)
        return out

def neural_bayes():
    return TC_Neural_Bayes(ResidualBlock)


class CNN_Bayes(nn.Module):
        def __init__(self,num_classes):
            super(LeNet5,self).__init__()
            #first convolution layer
            self.layer1 = nn.Sequential(
                    nn.Conv2d(3,6,kernel_size=3,stride=1,padding=0),
                    nn.BatchNorm2d(6),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2))
            #second convolution layer
            self.layer2 = nn.Sequential(
                    nn.Conv2d(6,16,kernel_size=3,stride=1,padding=0),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2,stride=2))
            #fully connected
            self.layer3 = nn.Sequential(
                    nn.Linear(16,120),
                    nn.ReLU(),
                    nn.Linear(120,84),
                    nn.ReLU(),
                    nn.Linear(84,num_classes))
            self.shortcut =  nn.Sequential(
                              nn.Conv2d(3,16,kernel_size=9,stride=1,padding=0),
                              nn.BatchNorm2d(16),
                              nn.ReLU(),
                              nn.MaxPool2d(kernel_size=2, stride=2))

        def forward(self,input):
            output=self.layer1(input)
            #print(output.shape)
            output=self.layer2(output)
            #print(output.shape)
            s = self.shortcut(input)
            output += s
            output=output.reshape(output.size(0),-1)
            output = torch.mean(output, dim = 0)
            #print(output.shape)
            output=self.layer3(output)
            return output


class GAT_Encoder(nn.Module):
    def __init__(self, in_features, out_features, n_heads, concat = True, sample_size):
        super(GAT_Encoder, self).__init__()
        self.GAT_1 = GraphAttentionLayer(in_features[0], out_features[0], n_heads[0], concat)
        self.GAT_2 = GraphAttentionLayer(in_features[1], out_features[1], n_heads[1], concat)
        self.GAT_3 = GraphAttentionLayer(in_features[2], out_features[2], n_heads[2], concat)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.linear_1 = nn.Linear(sample_size*512,2048)
        self.linear_2 = nn.Linear(2048,256)
        self.linear_3 = nn.Linear(256,7)
        

    def forward(self, x):
        out = self.GAT_1(x)
        out = self.GAT_2(out)
        out = self.GAT_3(out)
        out = self.encoder(out)
        out = self.linear_1(out)
        out = self.linear_2(out)
        out = self.linear_3(out)                                                                                
        return out














