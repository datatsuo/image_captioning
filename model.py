import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # embedding layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True) # LSTM 
        self.fc = nn.Linear(hidden_size, vocab_size) # fully connected layer 
    
    def forward(self, features, captions):
        embed_cap = self.embed(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1), embed_cap), 1) # input features from encoder and caption
        output, hidden = self.lstm(inputs)
        result = self.fc(output) # softmax is not needed here (softmax is applied in nn.CrossEntropyLoss() in the training step)
        return result
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
       
        list_inds = [] # for storing the indices 
        for _ in range(max_len):
            output, states = self.lstm(inputs, states)
            result = self.fc(output.view(len(output), -1))
            
            _, ind = result.max(dim = 1) # find the index of the largest value in result
            inputs = self.embed(ind).unsqueeze(1) # for the next turn, process the index with embedding layer
            
            list_inds.append(ind.item()) 
            if ind == 1: # if <end> shows up, stop appending afterward.
                return list_inds 
        
        return list_inds 
            
        
        
        
        
                         