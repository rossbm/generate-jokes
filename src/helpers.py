import os
from collections import deque
from math import exp
from keras.layers import Dense, Input, Masking, BatchNormalization, Layer, Embedding, Dropout
from keras.layers import LSTM, Reshape, TimeDistributed, Concatenate, Multiply, RepeatVector
from keras import Model
from keras import backend as K
import numpy as np
import h5py
import requests
import string
import pickle
import pandas as pd

def input_processer(seqs_path, dict_path):
    jokes_url = "https://onedrive.live.com/download?cid=ABD51044F5341265&resid=ABD51044F5341265%21112436&authkey=AFUOwOJbFyY6ZFM"
    jokes_path = "data/raw/reddit_jokes.csv"
    
    if not os.path.exists(jokes_path):
        print("Jokes file not found, so downloading it.")
        response = requests.get(jokes_url, allow_redirects=True)
        with open(jokes_path, 'wb') as f:
            f.write(response.content)
        print("Jokes downloaded.")

    jokes = pd.read_csv(jokes_path)
    #only funy ones
    jokes = jokes[jokes["ups"]>=10]

    #get rid of joke with very little text
    jokes = jokes[jokes["text"].str.len() >=4]
    jokes = jokes["title"].str.cat(jokes["text"], sep="\n\n")
    jokes = jokes.tolist()
    jokes = np.array(jokes)

    #pre make dict
    char_dict = {"<BOUND>":0}
    for ix, char in enumerate(string.printable):
        if char not in ('\x0c', '\x0b', "\r"):
            char_dict[char] = ix+1

    #now save the character dict
    pickle_out = open(dict_path,"wb")
    pickle.dump(char_dict, pickle_out)
    pickle_out.close()

    f = h5py.File(seqs_path, 'w')
    dt = h5py.special_dtype(vlen=np.dtype('int32'))
    dset = f.create_dataset('seqs', (len(jokes),), dtype=dt)
    for i in range(len(jokes)):
        text=jokes[i]
        #make certain all strings
        seq = [char_dict["<BOUND>"]]
        for char in text:
            if char in string.printable:
                try:
                    seq.append(char_dict[char])
                except KeyError:
                    pass
        dset[i]=np.array(seq, dtype="int32")



def sparse_softmax_cross_entropy_with_logits (y_true, y_pred):
    return K.sparse_categorical_crossentropy(target=y_true, output=y_pred, from_logits=True)
    #check if the jokes file is available, and download it if it is not


#class for iterating over jokes
class CharGenSequence(object):
    def __init__(self, seqs, char_dict, batch_size=500, seq_length=100):
        self.seqs = seqs
        self.char_dict = char_dict
        #need to know how big input to neural net (length of sequnece)
        self.seq_length = seq_length
        self.batch_size = batch_size
        #the  random permtuion returns a randomly sorted rangs
        self.seq_idxs = deque(np.random.permutation(len(seqs)).tolist())
        #now intitilize first batch
        #these are the indexes of seq_list that are used for the batch.
        #queu will help, can pop from left
        self.batch_idxs = [self.seq_idxs.pop() for _ in range(self.batch_size)]
        #walyas start at gegininb of joke
        self.seq_pos = [0 for seq in self.batch_idxs]
        
    def __next__(self):
        #make masks
        #used to determine if will use reset state or not...
        #will rely on brtaod casting to gie ii th eproer shape
        state_mask = np.ones((self.batch_size, 1), dtype=np.float32)
        #make x, the input a numpy array of zeros (initilaly)
        #nowing providing just indexes, since
        x = np.zeros((self.batch_size, self.seq_length, len(self.char_dict)), dtype=np.float)
        
        #will use sparse categorical
        y = np.zeros((self.batch_size, self.seq_length, 1), dtype=np.int32)
        
        #LOOP OVER BATCH
        #seq_idx is the index of the sequnce within seq_list
        #while batch_idx is the index of the sequence within the batch        
        for batch_idx, seq_idx in enumerate(self.batch_idxs):
            #work fowards...
            #GO UP TO LENGTH OF OUTPUTS...
            #check if this will be last batch for this input seq

            for pos_idx in range(self.seq_length):
                #self.seq_pos is start of sequence
                input_pos_idx =  self.seq_pos[batch_idx]+pos_idx
                #OUTPUTS ALWAYS ONE AHEAD OF INPUTS
                output_pos_idx = input_pos_idx + 1
                #if desired index does not exit, leave blank....
                try:
                    x[batch_idx, pos_idx, self.seqs[seq_idx][input_pos_idx]] = 1.
                except IndexError:
                    #leave at default of 0 (padding value)
                    pass
                try:
                    y[batch_idx, pos_idx, 0] = self.seqs[seq_idx][output_pos_idx]
                except IndexError:
                    #will be masked anyways?
                    y[batch_idx, pos_idx, 0] = self.char_dict["<BOUND>"]
        
            #DO SPECIAL STUFF IF length of sequence is less than than what was desired...
            if len(self.seqs[seq_idx]) <= (self.seq_length + self.seq_pos[batch_idx]):
                #first add back to avaialbe
                self.seq_idxs.append(self.batch_idxs[batch_idx])
                
                #pop left ensures first in, first out
                self.batch_idxs[batch_idx] = self.seq_idxs.popleft()
                    
                #set start pos back to 0
                self.seq_pos[batch_idx] = 0
                
                #make masks = 0 to reset state
                #could probaly do this outside of the generator...
                state_mask[batch_idx,0] = 0.0
            else:
                #increment position by seq_length
                #want last output character to be the first input character...
                self.seq_pos[batch_idx]=self.seq_pos[batch_idx]+ self.seq_length
        return(x, y, state_mask)
    def __iter__(self):
        return self

def create_model(batch_size, input_length, num_chars, batch_momentum=0.99, rnn_depth=1, dropout=0.0):
    #character sequences
    character_input = Input(batch_shape=(batch_size,input_length, num_chars), dtype='float', name='char_indx_input')
    sequences = Masking(name="mask")(character_input)

    for i in range(rnn_depth):
        units = 128 * (2 ** i)
        rnn =  LSTM(units=units,return_sequences=True, stateful=True,
                    dropout=dropout, recurrent_dropout=dropout, name="rnn"+str(i))(sequences)
        #don't use batch normaliztion on last layer
        if i == (rnn_depth-1):
            sequences = rnn
        else:
            sequences = BatchNormalization(momentum=batch_momentum, name="normalize"+str(i))(rnn)
    drop = TimeDistributed(Dropout(dropout), name="dropout")(rnn)
    preds = TimeDistributed(Dense(num_chars), name="logits")(drop)
    model = Model(inputs=[character_input], outputs=preds)
    
    return model

def reset_states(model, layer, mask):
    states = model.get_layer(layer)._states
    states = [np.multiply(K.eval(state), mask) for state in states]
    model.get_layer(layer).reset_states(states)

class GenerateJoke(object):
    def __init__(self, char_dict, max_len=1000, base_temp=1.0, temp_decay=0.001):
        #reverse the character dictionary, since will need to go from integer index to character
        self.char_dict_reverse = {value:key for key, value in char_dict.items()}
        self.char_dict = char_dict
        #keep track of last character
        self.end_ix = char_dict["<BOUND>"]
        self.max_len = max_len
        self.base_temp = base_temp
        self.temp_decay = temp_decay

    def __sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        #is logged proability
        #so exp(log(prob) / temperature) is smaller when temperature is higer
        #however derivative respect to temp: -log(prob) /temperature^2
        #-log(prob) is bigger when prob is smaller
        #so result is that lower temp makes smaller probs go to 0 faster
        preds = preds / temperature 
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)   

    def __call__(self, model):
        #need to make certain hiddent states are 0
        model.reset_states()

        x_input = np.zeros((1,1, len(self.char_dict)), dtype =np.float32)
        #first input to the model will be the <BOUND> token
        x_input[0,0, self.end_ix] = 1.0
        #x_indxs is used to store output, ant thus to generate jokes
        x_indxs = []

        for i in range(self.max_len):
            #want a decreasing temperature
            temperature = max(0.1, self.base_temp*exp(-i*self.temp_decay))
            #want only first...
            preds = model.predict_on_batch(x={"char_indx_input":x_input})[0,0]
            next_index = self.__sample(preds, temperature=temperature)
            #only need to update first index, since stateful...
            #make x_input again
            x_input = np.zeros((1,1, len(self.char_dict)), dtype =np.float32)
            x_input[0,0,next_index] = 1.0
            #now append to list that is used for text genration...
            if next_index == self.end_ix:
                break
            else:
                x_indxs.append(next_index)
        x_tokens = [self.char_dict_reverse[indx] for indx in x_indxs]
        x_string = "".join(x_tokens)
        return(x_string)

