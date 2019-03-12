import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

config={}
config['start_token'] = '<start>\n'
config['start_id'] = 0
config['end_token'] = '<end>\n'
config['end_id'] = 1
config['data_dir'] = 'data/'
config['num_epochs'] = 30
config['chunk_size'] = 100
config['batch_size'] = 1
config['train_path'] = config['data_dir']+'train.txt'
config['val_path'] = config['data_dir']+'val.txt'
config['test_path'] = config['data_dir']+'test.txt'
config['T'] = 1                             #tempreture parameter
config['num_hidden_units'] = 100            #number of hidden units
config['num_layers'] = 1                    #number of hidden layer
config['modelname'] = 'lstm'                # lstm or rnn
config['drop rate'] = 0.5
config['bidirectional'] = False
config['learning rate'] = 0.001
config['model_path'] = 'save/model.ckpt'
config['plot_dir'] = 'plot/'

# UNK_TOKEN = '<UNK>'
# UNK_ID = 1

def prep_data(datafile):
    data = []
    with open(datafile,"r") as f: 
        for i,line in enumerate(f):
            if line == config['start_token'] or line == config['end_token']:
                data.append(line)
            else:
                for char in line:
                    data.append(char)
    return data

def generate_data_dic(datafile):
    text2idx = {}
    idx2text = {}
    idx_count= 2
    text2idx[config['start_token']]= config['start_id']
    idx2text[config['start_id']] =config['start_token'] 
    text2idx[config['end_token']]= config['end_id']
    idx2text[config['end_id']] = config['end_token']
    data = prep_data(datafile)
    for char in list(set(data)):
        if char not in text2idx:
            text2idx[char] = idx_count
            idx2text[idx_count] = char
            idx_count+=1
    return text2idx,idx2text

def text2embedding(data_source,text2idx):
    if not isinstance(data_source,list):
        data_idx =  prep_data(data_source)
    else:
        data_idx = data_source
    data_idx = [text2idx[char] for char in data_idx]
    data_emb = np.zeros((len(data_idx),len(text2idx)))
    data_emb[np.arange(len(data_idx)), data_idx] = 1
    return torch.tensor(data_emb).float()

def embedding2text(data_emb,idx2text):
    data_idx  = np.argmax(data_emb,axis = 1)
    text_list = [idx2text[idx] for idx in data_idx]
    text = ''.join(text_list)
    return text

def batch_generator(data):
    base = 0
    offset = config['chunk_size']
    maximum = len(data)
    length = int(maximum/offset)+1
    for batch_id in range(length):
        if base+offset < maximum:
            #yild data and target(left shift by 1)
            features = data[base:base+offset]
            labels =data[base+1:base+offset+1]
            yield features,labels
            base += offset
    
class MusicRNN(nn.Module):
    def __init__(self):
        super(MusicRNN, self).__init__()
        self.input_size =config['output size']
        self.hidden_size = config['num_hidden_units']
        self.num_layers = config['num_layers']
        self.output_size = config['output size']
        self.modelname = config['modelname']
        self.batch_size = config['batch_size']
        print('Here we use:', self.modelname)
        if self.modelname == 'lstm':
#             print('here')
            self.rnn = nn.LSTM(self.input_size, self.hidden_size,self.num_layers,bidirectional=config['bidirectional'])
        elif self.modelname == 'rnn':
            self.rnn = nn.RNN(self.input_size, self.hidden_size,self.num_layers)
        if config['bidirectional']:
            self.fc =  nn.Linear(self.hidden_size*2, self.output_size)
        else:
            self.fc =  nn.Linear(self.hidden_size, self.output_size)
        self.drop = nn.Dropout(p=config['drop rate'])
            
    def init_hidden(self):
        if self.modelname == 'lstm':
            if config['bidirectional']:
                self.hidden = (Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size)).to(computing_device),  #h_0
                           Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size)).to(computing_device)) 
            else:
                self.hidden = (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).to(computing_device),  #h_0
                               Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).to(computing_device))  #c_0
        elif self.modelname == 'rnn':
            self.hidden = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).to(computing_device)
        return self.hidden
    
    def forward(self,x):
        x = x.view(x.shape[0],1,-1)
#         print(x.shape)
#         print(x[:10])
        out, self.hidden = self.rnn(x, self.hidden)
       
        if self.modelname == 'lstm': 
            self.hidden =(self.hidden[0].detach(),self.hidden[1].detach() )
        else:self.hidden = self.hidden.detach()
        out = self.drop(out)
        out = self.fc(out)
        out = out.view(x.shape[0],x.shape[2])
        return out
        
def run_epoch(model,data,training = False,optimizer=None):
    model.init_hidden()
    N_minibatch_loss =0.0
    data_loader = batch_generator(data)
    for minibatch_count, (data, labels) in enumerate(data_loader):
        data,labels = data.to(computing_device),labels.to(computing_device)
        if training:
            model.train()
        else:
            model.eval()
        if training:
            optimizer.zero_grad()
        outputs = model(data)
        labels = torch.argmax(labels,dim=1)
        loss = criterion(outputs, labels)
        N_minibatch_loss+=loss
        if training:
            loss.backward()
            optimizer.step()
    return  outputs, N_minibatch_loss/minibatch_count
        
def generate_music(model,prime ='data/prime.txt',maxlen=500):
    model.hidden = model.init_hidden()
    prime_emd = text2embedding(prime,text2idx).to(computing_device)
    song =prep_data(prime)
    out = prime_emd
    for i in range(maxlen):
        out=model(out)
        out_dis = np.exp(out[-1].detach().cpu().numpy()/0.7)
        out_dis = out_dis / np.sum(out_dis)
        out_id = np.random.choice(len(out_dis), p=out_dis)
        song.append(idx2text[out_id])
        if out_id == config['end_id']:
            break
    print(''.join(song))
    
    
use_cuda = torch.cuda.is_available()
# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 4, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")
text2idx,idx2text=generate_data_dic(config['train_path'])
train_data = text2embedding(config['train_path'],text2idx)
val_data = text2embedding(config['val_path'],text2idx)
test_data = text2embedding(config['test_path'],text2idx)
config['output size'] = train_data.shape[1]
criterion  = nn.CrossEntropyLoss()

def train():
    model = MusicRNN()
    model = model.to(computing_device)
    model.init_hidden()
    optimizer = torch.optim.Adam(model.parameters(),lr = config['learning rate'])

    train_loss_list = []
    val_loss_list = []
    for epoch in range(config['num_epochs']):
        min_val_loss = 100
        _,train_loss = run_epoch(model,train_data,training=True,optimizer=optimizer)
        _,val_loss = run_epoch(model,val_data,training = False)
        if val_loss<min_val_loss:
            min_val_loss=val_loss
            #torch.save(model.state_dict(), config['model_path'])
        print('Epoch %d,training loss: %.3f, validation loss:%.3f'%(epoch + 1, train_loss,val_loss ))
        train_loss_list.append(train_loss.detach())
        val_loss_list.append(val_loss.detach())
    print(min_val_loss)
    _,test_loss = run_epoch(model,test_data)
#     plot(train_loss_list,val_loss_list)
    print('Training completed after %d epochs, test loss is %.3f'%(epoch+1,test_loss))
    return train_loss_list,val_loss_list, model

def plot(train_loss,val_loss):
    plot_dir = config['plot_dir']
    plot_range = np.arange(len(train_loss))
    lines = plt.plot(plot_range,train_loss,plot_range,val_loss)
    plt.legend(('train loss', 'valid loss'),
                   loc='upper right',prop={'size': 8})
    plt.ylabel('loss')
    plt.xlabel('Number of iterations')
    foo_fig = plt.gcf()
    foo_fig.savefig(plot_dir+'h'+str(config['num_hidden_units'])+'loss.jpg',dpi=500)
    plt.show()

train_loss,val_loss,model = train()
generate_music(model)

# config['bidirectional'] = False
# model = MusicRNN()
# model.zero_grad()
# model = model.to(computing_device)
# optimizer = torch.optim.Adam(model.parameters(),lr = config['learning rate'])
# train()
    
    

# BIDIRE = False
# model = MusicRNN()
# model.zero_grad()
# model = model.to(computing_device)
# optimizer = torch.optim.Adam(model.parameters(),lr = LR)
# train()
    
    
