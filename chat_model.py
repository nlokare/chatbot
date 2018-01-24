import json
# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')
stemmer = LancasterStemmer()
# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle

class ModelBuilder(object):
    def __init__(self):
        with open('intents.json') as json_data:
            self.intents = json.load(json_data)
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = [
            'what', 'are', 'is', 'the', 'why', 
            'does', 'how', 'in', 'on', '?', 'my',
            'I'
        ]

    def parse_intents_doc(self):     
        # loop through each sentence in our intents patterns
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # tokenize each word in the sentence
                w = nltk.word_tokenize(pattern)
                # add to our words list
                self.words.extend(w)
                # add to documents in our corpus
                self.documents.append((w, intent['tag']))
                # add to our classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # stem and lower each word and remove duplicates
        self.words = [stemmer.stem(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        # remove duplicates
        self.classes = sorted(list(set(self.classes)))

    def build_training_data(self):
        # create our training data
        training = []
        output = []
        # create an empty array for our output
        output_empty = [0] * len(self.classes)

        # training set, bag of words for each sentence
        for doc in self.documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in self.words:
                if w in pattern_words:
                    bag.append(1)
                else:
                    bag.append(0)

            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1

            training.append([bag, output_row])

        # shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training)

        # create train and test lists
        train_x = list(training[:,0])
        train_y = list(training[:,1])
        return train_x, train_y

    def train_neural_network(self, train_x, train_y):
        # reset underlying graph data
        tf.reset_default_graph()
        # Build neural network
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)

        # Define model and setup tensorboard
        model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
        # Start training (apply gradient descent algorithm)
        model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
        model.save('model.tflearn')

        # save all of our data structures
        pickle.dump({
            'words': self.words,
            'classes': self.classes,
            'train_x': train_x,
            'train_y': train_y
          }, 
          open('training_data', 'wb')
        )

if __name__ == '__main__':
    model_builder = ModelBuilder()
    model_builder.parse_intents_doc()
    train_x, train_y = model_builder.build_training_data()
    model_builder.train_neural_network(train_x, train_y)
