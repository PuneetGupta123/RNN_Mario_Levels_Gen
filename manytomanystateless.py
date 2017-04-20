from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM,SimpleRNN
from keras.utils.data_utils import get_file
from keras.layers.wrappers import TimeDistributed
import numpy as np
from time import sleep
import random
import sys
from keras import optimizers
from keras.callbacks import ModelCheckpoint

##uncomment below if you want to use nietzches writings as the corpus

#path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
#text = open(path).read().lower()
text = open('transpose.txt').read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
print (char_indices)
indices_char = dict((i, c) for i, c in enumerate(chars))

# split the corpus into sequences of length=maxlen
#input is a sequence of 40 chars and target is also a sequence of 40 chars shifted by one position
#for eg: if you maxlen=3 and the text corpus is abcdefghi, your input ---> target pairs will be
# [a,b,c] --> [b,c,d], [b,c,d]--->[c,d,e]....and so on
maxlen = 40
step = 1
sentences = []
next_chars = []
X = []
y = [] 
totalsentences = 0
for j in range(13):
    txtname = "train"+str(j+1)+".txt"
    txt =  open(txtname).read()
    sents = []
    nextchars = []
    for i in range(0, len(txt) - maxlen+1, step):
        sents.append(txt[i: i + maxlen])
        nextchars.append(txt[i + 1: i + 1 + maxlen])
        sentences.append(sents)
        next_chars.append(nextchars)
    print('nb sequences:', len(sents))
    totalsentences = totalsentences + len(sents)
    XX = np.zeros(( len(sents), maxlen, len(chars)), dtype=np.bool)
    yy = np.zeros(( len(sents), maxlen, len(chars)), dtype=np.bool)  
    
    print('Vectorization...')   
    for ii, sentence in enumerate(sents):
        for t, char in enumerate(sentence):
            XX[ii, t, char_indices[char]] = 1

    for ii, sentence in enumerate(nextchars):
        for t, char in enumerate(sentence):
            yy[ii, t, char_indices[char]] = 1

    X.append(XX)
    y.append(yy)        
    
    print ('vetorization completed')   

print (totalsentences)    

# print (X[0][0][0])
# print (y[0][0][0])
# In[2]:

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
#model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))  # original one
#model.add(LSTM(32, input_shape=(maxlen,len(chars)), return_sequences = True)) #minesh witout specifying the input_length
#model.add(LSTM(512, return_sequences=True)) #- original
model.add(LSTM(32, batch_input_shape=(1, None, len(chars)), return_sequences = True, stateful=True)) 
model.add(Dropout(0.2))
#model.add(TimeDistributedDense(len(chars)))
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))
filename = "manytomanystatefulrmsprop1-00-0.6318.hdf5"
model.load_weights(filename)
rmsprop = optimizers.RMSprop(lr=0.0001)

#adam2 = optimizers.Adam(lr=0.01, decay=0.9)
# adam3 = optimizers.Adam(lr=0.01,decay=0.1)
# adam4 = optimizers.Adam(lr=0.01,decay=0.01)

model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

print ('model is made')

# train the model, output generated text after each iteration


# In[9]:

print (model.summary())

# filepath="manytomanystatefulrmsprop1-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='min')
# callbacks_list = [checkpoint]
# for i in range(500):
#     #print (adam3.lr)
#     for j in range(13):
#         #print (X[j][0])
#         #print (y[j][0])
#         if j==12:
#             hist = model.fit(X[j], y[j], epochs=1, batch_size=1, verbose=1, shuffle=False, callbacks=callbacks_list)
#             model.reset_states()
#             print (hist.history)
#         else:
#             hist = model.fit(X[j], y[j], epochs=1, batch_size=1, verbose=1, shuffle=False)
#             model.reset_states()
#             print (hist.history)   
#     print ("Epoch "+ str(i+1)+" completed")    
#fit the model
#model.fit(X, y, epochs=10, batch_size=64, callbacks=callbacks_list)

# start = np.random.randint(0, len(sentences)-1)
# #print start
# seed_string = sentences[start]


seed_string="s"
print ("seed string -->", seed_string)
print ('The generated text is')
sys.stdout.write(seed_string)
# x=np.zeros((1, len(seed_string), len(chars)))
for i in range(320):
    x=np.zeros((1, len(seed_string), len(chars)))
    for t, char in enumerate(seed_string):
        x[0, t, char_indices[char]] = 1
    preds = model.predict(x, verbose=0)[0]

    #############Comment start###############33

    preds = np.asarray(preds[len(seed_string)-1]).astype('float64')
    preds = np.log(preds) / 2.0
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    #print (preds)
    probas = np.random.multinomial(1, preds, 1)
    #print (probas)
    next_index=np.argmax(probas)


    ########33333333###commenr end##################33333


    #sys.stdout.write(str(preds.shape))
    #print (np.argmax(preds[7]))
    
    #next_index=np.argmax(preds[len(seed_string)-1])
    
    
    #next_index=np.argmax(preds[len(seed_string)-11])
    #print (preds.shape)
    #print (preds)
    #next_index = sample(preds, 1) #diversity is 1
    next_char = indices_char[next_index]
    seed_string = next_char
    #seed_string = seed_string[1:len(seed_string)]
    #print (seed_string)
    #print ('##############')
    #if i==40:
    #    print ('####')
    sys.stdout.write(next_char)

sys.stdout.flush()    
