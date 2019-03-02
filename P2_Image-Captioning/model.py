import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        #self.bn = nn.BatchNorm1d(embed_size, momentum=0.005)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        #features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size,
                            num_layers = num_layers, bias = True, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeds = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs
    

        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sample_output = []
        

        for i in range(max_len):
            lstm_hidden, states = self.lstm(inputs, states)
            outputs = self.linear(lstm_hidden).squeeze(1)
            predicted = outputs.argmax(1)
            sample_output.append(predicted.item())
            inputs = self.embed(predicted.unsqueeze(0))
        return sample_output
   