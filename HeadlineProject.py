import numpy as np 
import pandas as pd 
import re
import nltk 
import pprint
import yellowbrick
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('nps_chat')
import matplotlib.pyplot as plt
import matplotlib._animation_data
import matplotlib as animation
from matplotlib import style
from matplotlib.pyplot import text
from yellowbrick.text import DispersionPlot
#%matplotlib inline
f = open('headlines.txt')
raw = f.read()


#raw.dispersion_plot(["politics", "sports", "democracy", "health"])

posts = nltk.corpus.nps_chat.xml_posts()[:10000]
def dialouge_act_features(post):
    dialouge_act_features = {}
    for word in nltk.word_tokenize(post):
        dialouge_act_features['contains({})' .format(word.lower())] = True
    return dialouge_act_features

featuresets = [(dialouge_act_features(post.text), post.get('class'))
                for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
print(dialouge_act_features)
print(text)
style.use('fivethirtyeight') 
fig = plt.figure()
axl = fig.add_subplot(1,1,1) 
def animate (i):
    graph_data = ('headlines.txt', 'r').read()
    headlines = graph_data.split('\n')
xs = []
ys = []
for line in headlines:
        if len(line) > 1:
            x, y = line.split(',')
        xs.append(float(x))
        ys.append(float(y))
axl.clear()
axl.plot(xs, ys)
