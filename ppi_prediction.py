from keras.models import Sequential,load_model
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.layers.core import Dense, Dropout, Merge
import utils.tools as utils
from keras.regularizers import l2
import pandas as pd
from gensim.models.word2vec import Word2Vec
import copy
import h5py
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import SGD
from keras.regularizers import activity_l2
from keras import regularizers
import psutil
import os
from time import time
from urllib.request import urlopen
import sys

def token(dataset):
    token_dataset = []
    for i in range(len(dataset)):
        seq = []
        for j in range(len(dataset[i])):
            seq.append(dataset[i][j])
        token_dataset.append(seq)  
        
    return  token_dataset

def connect(protein_A,protein_B):
    # contect protein A and B
    protein_AB = []
    for i in range(len(protein_A)):
        con = protein_A[i] + protein_B[i] 
        protein_AB.append(con)
        
    return np.array(protein_AB)
    
def read_human_and_hpylori_seq(file_name_human,posnum,negnum):
    protein = pd.read_csv(file_name_human)  
   
    #pos_protein_A=protein.iloc[0:posnum,:] 
    #pos_protein_B=protein.iloc[posnum:posnum*2,:]  
    seq_pos_protein_A = []
    seq_pos_protein_B = []
    #neg_protein_A=protein.iloc[posnum*2:posnum*2+negnum,:]
    #neg_protein_B=protein.iloc[posnum*2+negnum:posnum*2+negnum*2,:]
    seq_neg_protein_A = []
    seq_neg_protein_B = []

    data = open( 'full_data.txt', 'r' )
    

    e3=  open( 'ligase_list.txt', 'r' )  

    e3l=[]

    for i in e3.readlines():
        e3l.append(i.rstrip('\n'))

    for line in data.readlines():
        one_line = line.rstrip('\n').split(' ')
        if int(one_line[4])==1:
            #if one_line[0] in ['P10275','P41159'] and one_line[1] in ['P10275','P41159']:
            #    break
            if one_line[0] in e3l or one_line[1] in e3l:
        
                if len(one_line[2])<=1166 and len(one_line[3])<=1166:
                   seq_pos_protein_A.append(one_line[2])
                   seq_pos_protein_B.append(one_line[3])
               
        else:
            #if one_line[0] in ['P10275','P41159'] and one_line[1] in ['P10275','P41159']:
             #   break
            if one_line[0] in e3l or one_line[1] in e3l:            
                if len(one_line[2])<=1166 and len(one_line[3])<=1166:
                   seq_neg_protein_A.append(one_line[2])
                   seq_neg_protein_B.append(one_line[3])
             
    return   seq_pos_protein_A, seq_pos_protein_B, seq_neg_protein_A, seq_neg_protein_B    

#%% 
def merged_DBN(sequence_len):
    # left model
    model_left = Sequential()
    model_left.add(Dense(2048, input_dim=sequence_len ,activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    model_left.add(Dropout(0.5))
    model_left.add(Dense(1024, activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    model_left.add(Dropout(0.5))
    model_left.add(Dense(512, activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    model_left.add(Dropout(0.5))
    model_left.add(Dense(128, activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
   
    # right model
    model_right = Sequential()
    model_right.add(Dense(2048,input_dim=sequence_len,activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    model_right.add(Dropout(0.5))   
    model_right.add(Dense(1024, activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    model_right.add(Dropout(0.5))
    model_right.add(Dense(512, activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    model_right.add(Dropout(0.5))
    model_right.add(Dense(128, activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    # together
    merged = Merge([model_left, model_right])
      
    model = Sequential()
    model.add(merged)
    model.add(Dense(8, activation='relu',W_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    #model.summary()
    
    return model
#%%

def pandding_J(protein,maxlen):           
    padded_protein = copy.deepcopy(protein)   
    for i in range(len(padded_protein)):
        if len(padded_protein[i])<maxlen:
            for j in range(len(padded_protein[i]),maxlen):
                padded_protein[i]=padded_protein[i]+'J'
    return padded_protein  
       

def residue_representation(wv,tokened_seq_protein,maxlen,size):  
    represented_protein  = []
    for i in range(len(tokened_seq_protein)):
        temp_sentence = []
        for j in range(maxlen):
            if tokened_seq_protein[i][j]=='J':
                temp_sentence.extend(np.zeros(size))
            else:
                temp_sentence.extend(wv[tokened_seq_protein[i][j]])
        represented_protein.append(np.array(temp_sentence))    
   
    return np.array(represented_protein)
    

        
def protein_reprsentation(wv,pos_protein_A,pos_protein_B,neg_protein_A,neg_protein_B,maxlen,size):
    # put positive and negative samples together
    pos_neg_protein_A = copy.deepcopy(pos_protein_A)   
    pos_neg_protein_A.extend(neg_protein_A)
    pos_neg_protein_B = copy.deepcopy(pos_protein_B)   
    pos_neg_protein_B.extend(neg_protein_B)
    
    # padding
    padded_pos_neg_protein_A = pandding_J(pos_neg_protein_A,maxlen)   
    padded_pos_neg_protein_B = pandding_J(pos_neg_protein_B,maxlen)   
                
    # token 
    token_padded_pos_neg_protein_A = token(padded_pos_neg_protein_A) 
    token_padded_pos_neg_protein_B = token(padded_pos_neg_protein_B)
                   
    # generate feature of pair A
    feature_protein_A = residue_representation(wv,token_padded_pos_neg_protein_A,maxlen,size )
    feature_protein_B = residue_representation(wv,token_padded_pos_neg_protein_B,maxlen,size )
    
    feature_protein_AB = np.hstack((np.array(feature_protein_A),np.array(feature_protein_B)))
    
    return feature_protein_AB


    
def human_data_processing(p1,p2,wv,maxlen,size):
    # get hpylori sequences
    file_name_human = 'dataset/human/human_protein.csv'
    pos_human_pair_A,pos_human_pair_B,neg_human_pair_A,neg_human_pair_B = read_human_and_hpylori_seq(file_name_human,3899,4262)
    seq_p1=get_sequence(p1)
    seq_p2=get_sequence(p2)
    neg_human_pair_A.append(seq_p1)
    neg_human_pair_B.append(seq_p2)
    print(len(pos_human_pair_A))
    print(len(neg_human_pair_A))
    feature_protein_AB = protein_reprsentation(wv, pos_human_pair_A,pos_human_pair_B,neg_human_pair_A,neg_human_pair_B,maxlen,size)
  
    # creat label
    label = np.ones(len(pos_human_pair_A)+len(neg_human_pair_A))
    
    label[len(pos_human_pair_A):] = 0

    
    return feature_protein_AB,label

def query_data_processing(p1,p2,wv,maxlen,size):
    # get  sequences
    seq_p1=[get_sequence(p1)]
    seq_p2=[get_sequence(p2)]
    
    feature_protein_12 = query_protein_reprsentation(wv,seq_p1,seq_p2,maxlen,size)
  
    return feature_protein_12

def get_sequence(proteinid):
    
    link = 'https://www.uniprot.org/uniprot/'+proteinid+'.fasta'          
    myfile = urlopen(link).read().decode('utf-8')
    return ''.join(myfile.split('\n')[1:])

def query_protein_reprsentation(wv,seq_p1,seq_p2,maxlen,size):
    
    # padding
    padded_seq_p1 = pandding_J(seq_p1,maxlen)   
    padded_seq_p2 = pandding_J(seq_p2,maxlen)   
                
    # token 
    token_padded_seq_p1 = token(padded_seq_p1) 
    token_padded_seq_p2 = token(padded_seq_p2)
                   
    # generate feature of pair A
    feature_protein_1 = residue_representation(wv,token_padded_seq_p1,maxlen,size )
    feature_protein_2 = residue_representation(wv,token_padded_seq_p2,maxlen,size )
    
    feature_protein_12 = np.hstack((np.array(feature_protein_1),np.array(feature_protein_2)))
    
    return feature_protein_12
"""
Main functions
"""

if __name__ == "__main__":  
    # load dictionary
    model_wv = Word2Vec.load('model/word2vec/wv_swissProt_size_20_window_4.model')
    size=20
    window=4
    maxlen=850
    batch_size=32
    nb_epoch=35
    sequence_len=size*maxlen
    #get training data
    p1=sys.argv[1]
    p2=sys.argv[2]
    train_fea_protein_ABs,train_label=human_data_processing(p1,p2,model_wv.wv,maxlen,size)

    #get testing data
    
    # standardscalar
    scaler = StandardScaler().fit(train_fea_protein_ABs)
    train_fea_protein_ABs = scaler.transform(train_fea_protein_ABs)
    #train test split
    
    fea_protein_AB = train_fea_protein_ABs[:-1,:]

    Y = utils.to_categorical(train_label[:-1])
    
    fea_protein_A = fea_protein_AB[:,0:sequence_len]
    fea_protein_B = fea_protein_AB[:,sequence_len:sequence_len*2]

    X_test_left = train_fea_protein_ABs[-1:,0:sequence_len]
    X_test_right = train_fea_protein_ABs[-1:,sequence_len:sequence_len*2]
                    
                        
    X_test_left  = np.array(X_test_left)
    X_test_right  = np.array(X_test_right)
                          
    skf = StratifiedKFold(n_splits = 5,random_state= 20181106,shuffle= True)
    predictions_tests=[]
    label_predict_tests=[]
    for (train_index, test_index) in skf.split(fea_protein_AB,train_label[:-1]):
        X_train_left = fea_protein_A[train_index]
        X_train_right = fea_protein_B[train_index]
        X_val_left = fea_protein_A[test_index]
        X_val_right = fea_protein_B[test_index]
                                      
        X_train_left  = np.array(X_train_left)
        X_train_right  = np.array(X_train_right)
                            
        X_val_left  = np.array(X_val_left)
        X_val_right  = np.array(X_val_right)
                          
        y_train = Y[train_index]
        y_val = Y[test_index]



        model =  merged_DBN(sequence_len)
        sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['precision'])
        hist = model.fit([X_train_left, X_train_right], y_train,
                                      batch_size = batch_size,
                                      nb_epoch = nb_epoch,
                                      verbose = 1,validation_data=([X_val_left, X_val_right],y_val)) 
    
        predictions_test = model.predict([X_test_left, X_test_right])
        predictions_tests.append(predictions_test)
        label_predict_test = utils.categorical_probas_to_classes(predictions_test)
        label_predict_tests.append(label_predict_test)
    
        print(predictions_tests)
        print(label_predict_tests)

        K.clear_session()


