import sys
import os
import pickle
from keras.models import load_model
from flask import Flask
app = Flask(__name__)
#change directory to parent of the script (so helpers can be found and paths work)
sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir)))
import helpers

#change directory to grandparent of the script so that paths work
os.chdir(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
print("Changed directory to {}".format(os.getcwd()))

model = load_model("models/predict_model.h5")
pickle_in = open("models/char_dict.pkl","rb")
char_dict = pickle.load(pickle_in)
pickle_in.close()

joke_generator = helpers.GenerateJoke(char_dict)

@app.route('/generate-joke')
def generate_joke():
    joke = joke_generator(model)
    return joke

if __name__ == "__main__":

    app.run(host='0.0.0.0')