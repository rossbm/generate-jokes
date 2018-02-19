import os
import sys
import pickle
import h5py
from keras.optimizers import Nadam
import numpy as np
from tqdm import tqdm

#add parent directory of script to python path so that it is always possible to import helpers
sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir)))
import helpers

CHARS_SEQS_PATH = "data/processed/char_sequences.h5"
CHAR_DICT_PATH = "models/char_dict.pkl"
BATCH_SIZE=512
RNN_DEPTH = 3
SEQ_LENGTH = 300
TRAINING_MODEL_PATH = "models/joke_gen_rnn.h5"
PREDICT_MODEL_PATH = "models/joke_gen_rnn_predict.h5"
BATCHES_PER_EPOCH = 250
DROPOUT_R = 0.1
NUM_EPOCHS = 150

if __name__ == "__main__":
    #change directory to grandparent of the script so that paths work
    os.chdir(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
    print("Changed directory to {}".format(os.getcwd()))

    if not os.path.exists(CHARS_SEQS_PATH):
        print("Processed input not found. Jokes will be downloaded and processed")
        helpers.input_processer(seqs_path=CHARS_SEQS_PATH, dict_path=CHAR_DICT_PATH)

    h5f = h5py.File(CHARS_SEQS_PATH, "r")
    seqs = h5f["seqs"][:]
    h5f.close()
    print("Loaded sequences")
    #load char dict
    pickle_in = open(CHAR_DICT_PATH,"rb")
    char_dict = pickle.load(pickle_in)
    pickle_in.close()
    num_seqs = seqs.shape[0]
    print("There are {} jokes in total.".format(num_seqs))
    batch_momentum = 1 - BATCH_SIZE/num_seqs
    num_chars=len(char_dict)
    gen_seq =  helpers.CharGenSequence(seqs, char_dict=char_dict, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)

    if os.path.exists(TRAINING_MODEL_PATH):
        print("Model already exists. If you continue this model will be ovewritten. Type 'yes' to continue. "
              "Type 'no' to exit and keep existing model.")
        resume = ""
        while resume not in ("yes", "no"):
            resume = input("yes/no: ")
        if resume == "no":
            sys.exit()
        else:
            print("Starting training from scratch. Existing model will be overwritten.")
    training_model = helpers.create_model(batch_size=BATCH_SIZE, input_length=SEQ_LENGTH, num_chars=num_chars,
                                              batch_momentum=batch_momentum, rnn_depth=RNN_DEPTH, dropout=DROPOUT_R)
    training_model.compile(loss=helpers.sparse_softmax_cross_entropy_with_logits, optimizer=Nadam())
    print("Model compiled and ready for training. Here is its summary:")
    print(training_model.summary())
    predict_model = helpers.create_model(batch_size=1, input_length=1, num_chars=num_chars,
                             rnn_depth=RNN_DEPTH)

    generator = helpers.GenerateJoke(char_dict)
    #now a loop
    epoch = 0
    while epoch < NUM_EPOCHS:
        print("***** Epoch {} *****".format(epoch)) 
        loss = np.zeros(BATCHES_PER_EPOCH, dtype=np.float32)
        for i in tqdm(range(BATCHES_PER_EPOCH)):
            char_input, y, state_mask = next(gen_seq)
            loss[i] = training_model.train_on_batch(x={"char_indx_input":char_input}, y=y)
            #now masking bit...
            for i in range(RNN_DEPTH):
                helpers.reset_states(training_model, "rnn"+str(i), state_mask)
        #save model, in case training is interupted
        training_model.save(PREDICT_MODEL_PATH)
        #lower learning rate after every "epoch"
        print("Average loss of {:.4f}".format(np.mean(loss)))
        print("***** EXAMPLE OUTPUT*****")
        predict_model.set_weights(training_model.get_weights())
        joke = generator(predict_model)
        print(joke)
        epoch+=1
    #save final predict model
    predict_model.save(PREDICT_MODEL_PATH)