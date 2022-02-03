import random
import torch
from io import open
import string
import matplotlib.pyplot as plt

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, ip_s, h_s, o_s):
        super(RNN, self).__init__()
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #CUDA_LAUNCH_BLOCKING=1
        self.h_s = h_s

        self.h1 = nn.Linear(ip_s + h_s, h_s)
        self.l1 = nn.Linear(ip_s + h_s, o_s)
        self.l2 = nn.Linear(ip_s + h_s, o_s)
        self.l3 = nn.Linear(h_s + o_s, o_s)
        self.l4 = nn.Linear(o_s, o_s)
        self.d1 = nn.Dropout(0.1)
        self.l5 = nn.Linear(o_s, o_s)
        self.d2 = nn.Dropout(0.1)
        self.l6 = nn.Linear(o_s, o_s)
        self.d3 = nn.Dropout(0.1)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, ip, h):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #CUDA_LAUNCH_BLOCKING=1
        ip_c = torch.cat((ip, h), 1)
        #ab = torch.cat([a.cuda(),b.cuda()], out=c)
        h = self.h1(ip_c)
        output = self.l1(ip_c)
        o1=self.l2(ip_c)
        out_c = torch.cat((h, output), 1)
        out_c1 = torch.cat((h,o1), 1)
        oc = out_c+out_c1
        oc1=torch.div(oc,2.0)
        output = self.l3(oc1)
        output=self.l4(output)
        output = self.d1(output)
        output=self.l5(output)
        output = self.d2(output)
        output=self.l6(output)
        output = self.d3(output)
        output = self.sm(output)
        return output, h

    def initHidden(self):
        return torch.zeros(1, self.h_s)




def main():

  
  #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  #CUDA_LAUNCH_BLOCKING=1

  print("Please run this only on CPU, the weights are not set for GPU")
  zaz= input("Enter the first character ") 
  print("Please enter the location of the file of the weights with its name as : abc/def/ghi.jkl")
  zz= input("Enter path to load the weights file from ")
  print("Please wait, I will be back in a jiffy")


  

  learning_rate = 0.0005

  los_l=[]
  zq=0

  
  l_c = string.ascii_lowercase+"!"
  l_s = len(l_c)

  rnn = RNN(l_s, 110, l_s)
  checkpoint=torch.load(zz)
  rnn.load_state_dict(checkpoint)

  criterion = nn.NLLLoss()
  ll=1000
  n_iters = 100000
  #print_every = 1
  #plot_every = 1
  all_losses = []
  total_loss = 0 

  r2=n_iters + 1



  w=zaz.lower()
  w=w*20




  sample(w,zz,l_c,l_s, rnn)


def f1(start_letter, l_s, l_c, rnn):
  t = torch.zeros(len(start_letter), 1, l_s)
  for li in range(len(start_letter)):
    letter = start_letter[li]
    t[li][0][l_c.find(letter)] = 1
  input = t
  hidden = rnn.initHidden()

  output_name = start_letter

  return(start_letter, letter, hidden, input)

def f2(rnn, input, hidden, output_name, l_c, l_s):

  for i in range(11):
    output, hidden = rnn(input[0], hidden)
    topv, topi = output.topk(1)
    topi = topi[0][0]
    if topi == l_s - 1:
        break
    else:
        letter = l_c[topi]
        output_name += letter
    t = torch.zeros(len(letter), 1, l_s)
    for li in range(len(letter)):
        letters = letter[li]
        t[li][0][l_c.find(letters)] = 1
    input = t
  return (output_name)


def sample( start_letters, zz, l_c, l_s, rnn):


  c=0
  s=set()
  while len(s)<20:
    for start_letter in start_letters:
      with torch.no_grad():  

        start_letter, letter, hidden, input = f1(start_letter, l_s, l_c, rnn)
        output_name= start_letter

 
        output_name=f2(rnn, input, hidden, output_name, l_c, l_s)
        s.add(output_name)


  
  print("Thank you for waiting")
  s9s_lst = list(s)
  print("The predicted names are : ")
  for i in range(20):
    print(s9s_lst[i])

if __name__ == '__main__':
  main()