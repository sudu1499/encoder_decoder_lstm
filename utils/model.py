from torch import nn

class Encoder(nn.Module):

    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.l=nn.LSTM(input_size,hidden_size)

    def forward(self,x):
        return self.l(x)
    
class Decoder(nn.Module):

    def __init__(self,input_size,hidden_size,number_clsses):

        super().__init__()
        self.l=nn.LSTM(input_size,hidden_size)
        self.d=nn.Linear(hidden_size,number_clsses)
        self.sm=nn.Softmax(dim=1)

    def forward(self,x,h,c):
        return (self.sm(self.d(self.l(x,(h,c))[0])),self.l(x,(h,c))[1][0],self.l(x,(h,c))[1][1])
    
