import json
import logging
from collections import namedtuple
import pickle
import random
# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf

Logger = logging.getLogger()
Logger.setLevel(logging.INFO)

data = pickle.load(open('training_data', 'rb'))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
with open('responses.json') as json_data:
    responses = json.load(json_data)

# load our saved model
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load('./model.tflearn')

ERROR_THRESHOLD = 0.75

Response = namedtuple('Response', 'top_match, match_rate, answer')

class ChatBot(object):

    def clean_up_sentence(self, sentence):
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, sentence, words, show_details=False):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words
        bag = [0]*len(words)  
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)

        return(np.array(bag))

    def classify(self, sentence):
        # generate probabilities from the model
        results = model.predict([self.bow(sentence, words)])[0]
        # filter out predictions below a threshold
        results = [[i,r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((classes[r[0]], r[1]))
        # return tuple of intent and probability
        logging.info('Classification Results')
        logging.info(return_list)
        return return_list

    def response(self, sentence, show_details=False):
        results = self.classify(sentence)
        # if we have a classification then find the matching intent tag
        if results:
            top_match = results[0][0]
            match_rate = results[0][1]
            answer = responses[top_match][random.randrange(0,2)]
            return Response(
                top_match=top_match,
                match_rate=match_rate.item(),
                answer=answer
            )
        else:
            return Response(
                top_match=u'no_match',
                match_rate=0.0,
                answer=responses['Unknown']
            )
