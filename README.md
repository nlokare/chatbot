## Chatbot Prototype

Download Jupyter: https://jupyter.readthedocs.io/en/latest/install.html
A field engineer can assist with setting up your python environment, including pip, and getting you up and running.

There are some packages that you need to install in order for your notebook to work.

`pip install -r requirements.txt` should do the trick

Once you have Jupyter installed and running and your dependencies installed, you are ready to start testing the chat bot.

When running the notebook, you can use `shift + enter` to run each code block. The `intents.json` file contains the seed data
for training the neural network. As you add to this, you will have to re-run the `model_builder` code block in the notebook.
The seed doc should be self explanatory. When seeding pattern questions, be sure to add both acronyms, official platform titles,
and other jargon that is commonly used so that the neural network is trained extensively.

Once you have trained the model and initialized an instance of `ChatBot`, you can now QA the quality of responses by running,
in separate code blocks, `chat_bot.response('YOUR QUESTION HERE')`

---

As you can see, the initial `intents.json` doc is barren. The key to this prototype relies on an extensive, exhaustive set of training data. A variety of patterns and thorough responses are ideal for what we are trying to achieve.