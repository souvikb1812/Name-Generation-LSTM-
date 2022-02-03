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
  

  learning_rate = 0.0005

  los_l=[]
  zq=0
  print("Please enter the location of the file with its name as : abc/def/ghi.txt")
  path = input("Enter the path where the file is located ")
  loc=input("Enter location to store the model weights at ")
  looc=loc+'/lstm_weights_best.model'
  
  print("\nIf the program doesn't run properly, then please check the accuracy of the paths provided above\n")
  #path='/content/sample_data/names.txt'

  l_c = string.ascii_lowercase+"!"
  l_s = len(l_c)
  l9 = open(path).read().strip().split('\n')
  for i in range(len(l9)):
    t9=11-len(l9[i])
    for j in range(t9):
      l9[i]+='!'



  rnn = RNN(l_s, 110, l_s)
  criterion = nn.NLLLoss()
  ll=1000
  n_iters = 10000
  all_losses = []
  total_loss = 0 

  r2=n_iters + 1

  for e in range(1, r2):


      line = l9[0][random.randint(0, len(l9[0]) - 1)]
      l1=line
      l2=line
      t = torch.zeros(len(l1), 1, l_s)
      r=len(l1)
      for li in range(r):
          letter = l1[li]
          t[li][0][l_c.find(letter)] = 1
      input_line_tensor = t


      letter_indexes = [l_c.find(l2[lj]) for lj in range(1, len(l2))]
      letter_indexes.append(l_s - 1) 

      target_line_tensor = torch.LongTensor(letter_indexes)

      f = input_line_tensor
      g = target_line_tensor



      g.unsqueeze_(-1)
      hidden = rnn.initHidden()

      rnn.zero_grad()

      loss = 0
      r1=f.size(0)
      for i in range(r1):
          output, hidden = rnn(f[i], hidden)
          l = criterion(output, g[i])
          loss += l

      loss.backward()

      for p in rnn.parameters():
          p.data.add_(p.grad.data, alpha=-learning_rate)

      los= (loss.item() / f.size(0))
      los_l.append(los)

      zm=0
      if los<ll:

        torch.save(rnn.state_dict(),'lstm_weights_best.model')
        torch.save(rnn.state_dict(), (loc+'/lstm_weights_best.model'))
        ll=los
        zm=1


      total_loss += los


      print("Epoch: {}  Completed: {}%  Loss: {}".format(e, e / n_iters * 100, los))


      all_losses.append(total_loss)
      total_loss = 0

      if (e>2 and e>2000):
        n1= float(los_l[-2])
        n2= float(los_l[-1])
        if (abs(n1-n2)<0.0001):
          zq=1
      if zq==1 and zm==1:
        print("\n The model has converged \n")
        break

  plt.figure(figsize=(10, 5))
  plt.plot(all_losses)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend(['Loss'])
  plt.title('Loss vs. No. of epochs')
  plt.show()

  print("\nThe weight are stored at {}".format(looc))



if __name__ == '__main__':
    main()
