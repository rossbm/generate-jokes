# generate-jokes
This repository will host my work on generating jokes using the r/jokes corpus. It is a follow up to my capstone project for Udacity's Machine Learning Nanodegree.

## How to use
There are two main components to this repository: training and prediction

### Train
To train a model call the __train.py__ script in the __src__ folder. It will train a model, downloading the raw data if necessary. It will output example outputs at regular intervals. Under the default settings it will take about 66 hours on 1080 TI. Setting are currently set as global variables at the top of the script file. Feel free to play around with them.

### Predict
To generate outputs from a trained mode, call the __predict.py__ script. It will launch a Flask app on port 5000. Navigate to __http://localhost:5000/generate-joke__ to generate a joke. Refresh the page in your browser to generate a new joke.

## Model Explanation
The model is a reccurent neural network. It operate at the character level. It consists of several stacked hidden LSTM layers.

## Corpus
It used the same raw data as my capstone project.

## Example Outputs
Here are some example outputs from a fully trained model. They have been screened so as to avoid vulgarity, but are otherwise not cherry picked.

```Blonde Shadow works in a branch of driving...

Amover owned a message and told her about her tuxedo. The blonde of the bartender told her how to take the loan officer to her car and walked over to the guy for a different design.  When she came home the men said, "I'm a slight apple."

"No," she said, "I seven years old, and I called up to a nun hanging out."


The blonde said, "Yes, I don't want to wake up any more my mother." 

The guy said, "Yes, I'm a blonde."

 The man replied, "Well, that's about twenty years old.  You still look like you're not a man."

The bartender replied, "Well, then I was still alone."
```
```
Guy Walks Into a bar

And watches a driver set them up to him. He goes back into the bar and returns. He picks up the rib from his tub of tequila, and says, "what are you going to do with a blind man? I was with directions on my box and it sucks that big man playing like that that happened!"
```

```
Whats the difference between a tree and a magic dinosaur?

One is polite.
```