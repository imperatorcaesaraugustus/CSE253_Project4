from train_model import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='lstm')
parser.add_argument('--T',type=int,default =1)
parser.add_argument('--num_hidden_units',type=int,default =100)
parser.add_argument('--learning rate',type=float,default =0.001)
parser.add_argument('--drop rate',type=float,default =0.5)
args = parser.parse_args()
opt = vars(args)
if __name__ == "__main__":
#     print(opt['modelname'])
    config['modelname'] =opt['modelname']
    config['T'] =opt['T']
    config['num_hidden_units'] =opt['num_hidden_units']
    config['learning rate'] =opt['learning rate']
    config['drop rate'] =opt['drop rate']
    train_loss,val_loss = train()
