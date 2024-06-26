from utils.model import Encoder,Decoder
from utils.data_from_ohe import data,data_ohe
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np

number_of_items=103
path=r'E:\ENCODER_DECODER_LSTM\data\Dataset_English_Hindi.csv'

(o,_),(eng_ohe,hindi_ohe)=data_ohe(path,number_of_items)
eng_x,hindi_x,input_length_eng,input_length_hindi=data(path,number_of_items-1,eng_ohe,hindi_ohe)

class my_dataset(Dataset):

    def __init__(self,e,h):

        self.eng=e
        self.hindi=h
        self.n=number_of_items
    
    def __getitem__(self,index):
        return self.eng[index],self.hindi[index]
    
    def __len__(self):
        return self.n
    
dataset=my_dataset(eng_x,hindi_x)

enc=Encoder(input_length_eng[0],32).to("cuda")
dec=Decoder(input_length_hindi[0],32,input_length_hindi[0]).to("cuda")

opt=torch.optim.Adam(params=dec.parameters())
loss=nn.CrossEntropyLoss().to("cuda")

for e in range(100):
    total_diff=0
    for i,j in dataset:
        ypred=[]
        y=[]
        for k in  i:

            r,(h,c)=enc(torch.tensor(k,device="cuda",dtype=torch.float32))
        
        yp=torch.tensor(hindi_ohe.transform([['start']]).toarray(),device="cuda",dtype=torch.float32)
        end=torch.tensor(hindi_ohe.transform([['end']]).toarray(),device="cuda",dtype=torch.float32)

        for l in range(len(j)+1):
            yp,h,c=dec(yp,h,c)
            yp=yp.detach().cpu().numpy()
            ypred.append(yp)
            if l==len(j):
                y.append(end)
                break
            else:
                y.append(j[l])
            yp=torch.tensor(j[l],device="cuda",dtype=torch.float32)
        ypred=torch.tensor(np.array(ypred),requires_grad=True).squeeze(1).to("cuda")
        y=torch.tensor(y,dtype=torch.float32,device="cuda",requires_grad=True).squeeze(1)
        diff=loss(ypred,y)
        total_diff+=diff
        opt.zero_grad()
        diff.backward()
        opt.step()

    print(f"in the epoc  {e}  the loss is {total_diff/103}")


def test_the_seq2seq(str):
    global enc
    global dec
    global eng_ohe
    global hindi_ohe
    global input_length_hindi
    result=[]
    enc_result=enc(torch.tensor(eng_ohe.transform([[str]]).toarray(),dtype=torch.float32,device='cuda'))

    start=torch.tensor(hindi_ohe.transform([['start']]).toarray(),device="cuda",dtype=torch.float32)
    end=torch.tensor(hindi_ohe.transform([['end']]).toarray(),device="cuda",dtype=torch.float32)

    dec_result,h,c=dec(start,enc_result[1][0],enc_result[1][1])
    z_op=torch.zeros((1,input_length_hindi[0]))
    z_op[0,torch.argmax(dec_result)]=1
    result.append(hindi_ohe.inverse_transform(z_op))

    while(torch.eq(dec_result,end).all()):
        dec_result,h,c=dec(dec_result,h,c)
        z_op=torch.zeros((1,input_length_hindi[0]))
        z_op[0,torch.argmax(dec_result)]=1
        result.append(hindi_ohe.inverse_transform(z_op))
    return result

test_the_seq2seq("Cheers!")

# def lossss(y1,y2):

#     s=0
#     for i,j in zip(y1,y2):
#         s+=-np.multiply(j,np.log(i)).sum()

#     return s
