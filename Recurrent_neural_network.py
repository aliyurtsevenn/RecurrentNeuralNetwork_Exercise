import  pandas as pd
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path_="../spam.csv"

data_=pd.read_csv(path_,encoding="latin-1")
data_=data_.drop(labels=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
data_.columns= ["labels","text"]




data_["labels"] = np.where(data_["labels"]=="spam",1,0)

X_train, X_test, Y_train, Y_test= train_test_split(data_["text"], data_["labels"],
                                                   test_size=0.2)




from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Use the tokenizer to transform the text messages in the training and test sets
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

print(X_test_seq[0])

# We need to pad the sequences, so that each sequence will have the same length

X_train_seq_padded = pad_sequences(X_train_seq,50)
X_test_seq_padded = pad_sequences(X_test_seq,50)

# We prepared our model, we can now fit our model!

import keras.backend as K
from keras.layers import Dense,Embedding,LSTM
from keras.models import Sequential


def recall(y_true,y_pred):
    true_positives = K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    possible_positives = K.sum(K.round(K.clip(y_true,0,1)))
    recall = true_positives/(possible_positives+K.epsilon())
    return recall
def precision(y_true,y_pred):
    true_positives = K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives/(predicted_positives+K.epsilon())
    return precision

# Construct model

model= Sequential()

# Let me add first embedding layer!

model.add(Embedding(len(tokenizer.index_word)+1,32)) # 32 length of embeddings!
model.add(LSTM(32,dropout=0,recurrent_dropout=0)) # Tell the dimensionality of the output space. Generally, you use the same dimension of input space
model.add(Dense(32,activation="relu")) # Tell the dimensionality of the output space. Generally, you use the same dimension of input space
model.add(Dense(1,activation="sigmoid")) # Tell the dimensionality of the output space. Generally, you use the same dimension of input space
summary_= model.summary()
print(summary_)

'''

LSTM layer - long short term memory, they are type of RNN.

Stacking LSTM hidden layers makes the model deeper, more accurately earning the description as a deep
learning technique

Generally, 2 layers have shown to be enough to detect more complex features. More layers can be better
but also harder to train. As a general rule of thumb — 1 hidden layer work with simple problems, like this,
and two are enough to find reasonably complex features.

A dense layer also referred to as a fully connected layer is a layer that is used in the final stages of the
neural network

dropout - this  hyper-parameter control the regularization of your model. One issue in neural networks, they are
prone to overfit your training data. Regularization is to one way to prevent the overfitting.The most common
type of it is called as dropout.  It basically drops certain percentage of the nodes per each pass, to force
the other nodes to pick out the slack and learn how to generalize better.

Dense layer - it is just a standard, fully connected neural network layer, that includes some kind of
transformation. Fully-connected means that every node in this layer is connected to the every layer before it
and layer after it. It also includes some kind of transformation. We just need to tell what transformation
we want to do, which is called as activation function. We will define the dimensionality output space,
we also need to specify the type of activation function.

'''

# We need to compile the model

# calculates precision for 1:100 dataset with 90 tp and 30 fp
model.compile(optimizer="adam",
              loss="binary_crossentropy", # This is the loss function
              metrics=["accuracy",precision,recall])

# Fit the model

history_= model.fit(X_train_seq_padded, Y_train,
                    batch_size=32, epochs=10,
                    validation_data=(X_test_seq_padded,Y_test))

import matplotlib.pyplot as plt

for each in ["accuracy","precision","recall"]:
    acc = history_.history[each]
    val_acc = history_.history["val_{}".format(each)]
    epochs = range(1,len(acc)+1)
    plt.figure()
    plt.plot(epochs,acc,label = "Training Accuracy")
    plt.plot(epochs,val_acc,label = "Validation Accuracy")
    plt.title("Results for {}".format(each))
    plt.legend()
    plt.show()

