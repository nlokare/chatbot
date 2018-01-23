import json
import logging
import os

from flask import current_app, Flask, jsonify, render_template, request
import config
from chat_model import ModelBuilder
from responder import ChatBot
from chat_storage import ChatStorage
import datastore_interface

chat_bot = ChatBot()

app = Flask(__name__)
app.config.from_object(config)
Logger = logging.getLogger()
Logger.setLevel(logging.INFO)

# Setup the data model.
with app.app_context():
  Question = datastore_interface

def get_help_text():
  return "C.R.I.T.E.L.L.I is a chat bot that is designed to help answer your questions about Doubleclick Bid Manager.\n" + "The service is still in beta, so it's a bit rough around the edges. However, we are keeping track of the questions you ask, and will be working on making the chat bot smarter.\n"

@app.route('/', methods=['GET'])
def home():
  return 'Welcome to C.R.I.T.E.L.L.I'

@app.route('/train', methods=['GET'])
def train():
  model_builder = ModelBuilder()
  model_builder.parse_intents_doc()
  train_x, train_y = model_builder.build_training_data()
  model_builder.train_neural_network(train_x, train_y)
  return 'Chat model trained'

@app.route('/chat', methods=['POST'])
def chat():
  question = request.form['text']
  if question == 'help':
    return jsonify(
      response_type='in_channel',
      text=get_help_text()
    )
  else:
    tag, match_rate, chat_response = chat_bot.response(question)
    Question.create({
        'match_rate': unicode(match_rate),
        'tag': tag,
        'question': question,
        'response': chat_response
      })
    return jsonify(
      response_type='in_channel',
      text=chat_response
    )

@app.route('/chat-tester', methods=['GET'])
def chat_test():
  question = request.args.get('input')
  if question == 'help':
    return render_template('help.html', text=get_help_text())
  else: 
    tag, match_rate, chat_response = chat_bot.response(question)
    Question.create({
        'match_rate': unicode(match_rate),
        'tag': tag,
        'question': question,
        'response': chat_response
      })
    return jsonify(
      text=chat_response
    )

if __name__ == '__main__':
    app.run(debug=True)
