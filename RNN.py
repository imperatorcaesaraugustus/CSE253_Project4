################################################################################
# CSE 253: Programming Assignment 4
# Feb 2019
################################################################################

import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as func
import torch.nn.init as torch_init
import torchvision
from torchvision import transforms, utils
from DataLoader import create_datasets
import numpy as np
import matplotlib.pyplot as plt
import random
import os


def make_chunks(data, batch, seq_len, index):
    inputs, targets, targets_C = [], [], []   # targets_C is none one-hot version of targets, 25*100
    for i in range(batch):
        seq0, seq1, seq2, tmp = [], [], [], [0]*93
        for j in range(seq_len):
            tmp[data[index + i*seq_len + j][1]] = 1
            seq0.append(list(tmp))
            tmp[data[index + i*seq_len + j][1]] = 0
            tmp[data[index + i*seq_len + j + 1][1]] = 1
            seq1.append(list(tmp))
            seq2.append(data[index + i*seq_len + j + 1][1])
            tmp[data[index + i*seq_len + j + 1][1]] = 0
        inputs.append(list(seq0))
        targets.append(list(seq1))
        targets_C.append(list(seq2))
    #print(np.array(inputs).shape)
    return torch.Tensor(inputs), torch.Tensor(targets), torch.Tensor(targets_C).long()
    

class RNN_composer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN_composer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = True)
        self.log_softmax = nn.LogSoftmax(dim = 2)
        self.out2tag = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(p = 0.5)

    def forward(self, inputs, hidden, Temperature):
        batch = inputs.shape[0]
        hidden = self.dropout(hidden)
        outputs, hidden = self.rnn(inputs, hidden)
        outputs = self.log_softmax(self.out2tag(torch.mul(outputs, 1.0/Temperature).view(inputs.shape[0], inputs.shape[1], self.hidden_size)))
        return outputs, hidden

    def init_hidden(self, num_layers, batch, hidden_size):
        return autograd.Variable(torch.zeros(num_layers, batch, hidden_size)).type(torch.FloatTensor).cuda()


LR = 0.0001
epochs = 100
seq_len = 100
batch = 1
input_size = 93
hidden_size = 100
output_size = 93
num_layers = 1
softmax_temperature = 0.01

train_data, val_data, test_data, dict1, dict2 = create_datasets()
#print(train_data[:20], '\n')
train_losses, val_losses = [], []

use_cuda = torch.cuda.is_available()
# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 0, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

model = RNN_composer(input_size, hidden_size, output_size, num_layers)
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = LR)
h = model.init_hidden(num_layers, batch, hidden_size)

for epoch_cnt in range(epochs):
    train_loss, train_leng = 0.0, int(len(train_data)/(batch*seq_len))
    model.train()
    for batch_cnt in range(train_leng):
        model.zero_grad()
        h = h.detach()
        inputs, targets, targets_C = make_chunks(train_data, batch, seq_len, seq_len*batch*batch_cnt)
        inputs, targets_C = inputs.to(computing_device), targets_C.to(computing_device)
        outputs, h = model(inputs, h, softmax_temperature)
        #print(outputs.shape, targets_C.shape)
        loss = loss_function(outputs.reshape(batch*seq_len, 93), targets_C.reshape(batch*seq_len,))
        loss.backward(retain_graph = False)
        optimizer.step()
        train_loss += float(loss.item())
        del loss, inputs, targets, targets_C, outputs
    print("Epoch:", epoch_cnt + 1, "\nTrain loss:", train_loss/train_leng)
    #print(train_loss, train_leng)
    train_losses.append(train_loss/train_leng)

    val_loss, val_leng = 0.0, int(len(val_data)/(batch*seq_len))
    model.eval()
    with torch.no_grad():
        if epoch_cnt % 1 == 0:
            h2 = model.init_hidden(num_layers, batch, hidden_size)
            for batch_cnt2 in range(val_leng):
                inputs, targets, targets_C = make_chunks(val_data, batch, seq_len, seq_len*batch*batch_cnt2)
                inputs, targets_C = inputs.to(computing_device), targets_C.to(computing_device)
                outputs, h2 = model(inputs, h2, softmax_temperature)
                loss = loss_function(outputs.reshape(batch*seq_len, 93), targets_C.reshape(batch*seq_len,))
                val_loss += float(loss.item())
                h2.detach()
                del loss, inputs, targets, targets_C, outputs
            print("Val loss:", val_loss/val_leng)
            val_losses.append(val_loss/val_leng)
    
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0.0, 3.0)
plt.grid()
x_label = [i + 1 for i in range(len(train_losses))]
plt.plot(x_label, train_losses, label = "Training Loss")
plt.plot(x_label, val_losses, label = "Validation Loss")
plt.legend()
plt.show()


### generate music
def decode(outputs, ctrl, dict2):   # outputs: 1*7*93 
    res = ""
    if ctrl == 0:   # pick up index from weighted probability vector
        for liste in outputs[0]:
            rade = random.random()
            cur_total, index = 0.0, 0
            for i in range(len(liste)):
                cur_total += liste[i] 
                if rade <= cur_total:
                    index = i
                    break
            res += str(dict2[index])
    elif ctrl == 1:   # pick up argmax index
        for liste in outputs[0]:
            maxi, max_index = -100, 0
            for i in range(len(liste)):
                if maxi < liste[i]:
                    maxi = liste[i]
                    max_index = i
            res += str(dict2[max_index])
    return res

def generate_input(inputs, dict1):
    res, tmp = [[]], [0]*93
    for i in range(len(inputs)):
        tmp[dict1[inputs[i]]] = 1
        res[0].append(tmp)
        tmp[dict1[inputs[i]]] = 0
    return torch.Tensor(res)

#for i in dict2: print(i, dict2[i])
begin_txt = "<start>X:19\nT:Dusty Miller, The\nR:hop jig\nD:Tommy Keane: The Piper's Apron\nZ:id:hn-slipjig-19\nM:9/8\nK:D\n~A3"
music_txt = "<start>X:19\nT:Dusty Miller, The\nR:hop jig\nD:Tommy Keane: The Piper's Apron\nZ:id:hn-slipjig-19\nM:9/8\nK:D\n~A3"
model.eval()
input_txt = str(begin_txt)
target_len = 200
with torch.no_grad():
    h3 = model.init_hidden(num_layers, batch, hidden_size)
    while len(music_txt) < target_len:
        inputs = generate_input(input_txt, dict1)    # generate len(input_txt) chars each time
        inputs = inputs.to(computing_device)
        outputs, h3 = model(inputs, h3, softmax_temperature)
        output_txt = decode(torch.exp(outputs), 0, dict2)
        input_txt = output_txt
        music_txt += output_txt
        h3.detach()
        del inputs, outputs
print("\nStarts w heading:\n",music_txt,"\nEnds.")
f1 = open('music_generated.txt', 'w')
f1.write(music_txt)
f1.close()

begin_txt = "<start>"
music_txt = "<start>"
model.eval()
input_txt = str(begin_txt)
target_len = 200
with torch.no_grad():
    (h3, c3) = model.init_hidden(num_layers, batch, hidden_size)
    while len(music_txt) < target_len:
        inputs = generate_input(input_txt, dict1)    # generate len(input_txt) chars each time
        inputs = inputs.to(computing_device)
        outputs, (h3, c3) = model(inputs, (h3, c3), softmax_temperature)
        output_txt = decode(torch.exp(outputs), 0, dict2)
        input_txt = output_txt
        music_txt += output_txt
        h3.detach()
        c3.detach()
        del inputs, outputs
print("\nStarts w/o heading:\n",music_txt,"\nEnds.")





