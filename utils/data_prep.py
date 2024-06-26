from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

def data_ohe(path,number_items):
    data=pd.read_csv(path)

    ohe_hindi=OneHotEncoder()
    ohe_eng=OneHotEncoder()
    hindi_vocab=[]
    eng_vocab=[]

    for i in data.values[:number_items]:
        for j in i[1].split():
            hindi_vocab.append(j)
        for k in i[0].split():
            eng_vocab.append(k)
    
    hindi_vocab.append("start")
    hindi_vocab.append("end")
    hindi_vocab=np.array(hindi_vocab).reshape(-1,1)
    hindi_vocab_ohe=ohe_hindi.fit_transform(hindi_vocab).toarray()
    hindi_vocab_ohe=np.unique(hindi_vocab_ohe,axis=0)

    eng_vocab=np.array(eng_vocab).reshape(-1,1)
    eng_vocab_ohe=ohe_eng.fit_transform(eng_vocab).toarray()
    eng_vocab_ohe=np.unique(eng_vocab_ohe,axis=0)

    return (eng_vocab_ohe,hindi_vocab_ohe),(ohe_eng,ohe_hindi)

