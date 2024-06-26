from utils.data_prep import data_ohe
from tqdm import tqdm
import pandas as pd
import numpy as np

def data(path,number_items,eng_ohe,hindi_ohe):

    dataset=pd.read_csv(path)

    eng_x=[]
    hindi_x=[]

    input_length_eng=eng_ohe._n_features_outs
    input_length_hindi=hindi_ohe._n_features_outs

    for k,i in enumerate(dataset.values):
        if k<=number_items:
            t=[]
            t=[eng_ohe.transform([[j]]).toarray() for j in i[0].split()]
            eng_x.append(t)
            t=[]
            t=[hindi_ohe.transform([[j]]).toarray() for j in i[1].split()]
            hindi_x.append(t)

        else:
            break
    hindi_x.append(hindi_ohe.transform([['start']]).toarray())
    hindi_x.append(hindi_ohe.transform([['end']]).toarray())
    return eng_x,hindi_x,input_length_eng,input_length_hindi
