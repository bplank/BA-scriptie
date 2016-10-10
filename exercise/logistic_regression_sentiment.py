__author__ = "bplank"
"""
Exercise: sentiment classification with logistic regression

1) Examine the code. What are the features used?
2) What is the distribution of labels in the data?
3) Add code to train and evaluate the classifier. What accuracy do you get? What is weird?
4) How could you improve the representation of the data?
5) Add confusion matrix output.
"""
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import random

def load_sentiment_sentences_and_labels():
    """
    loads the movie review data
    """
    ## Q1: What are the features used? Q4: How could you improve the representation of the data?
    positive_sentences = open("rt-polaritydata/rt-polarity.pos").readlines()
    negative_sentences = open("rt-polaritydata/rt-polarity.neg").readlines()

    ## Q2: What is the label distribution?
    positive_labels = [1 for sentence in positive_sentences]
    negative_labels = [0 for sentence in negative_sentences]

    sentences = np.concatenate([positive_sentences,negative_sentences], axis=0)
    labels = np.concatenate([positive_labels,negative_labels],axis=0)

    ## make sure we have a label for every data instance
    assert(len(sentences)==len(labels))
    data = list(zip(sentences,labels))

    ## return the data (instances + labels)
    return data

## read input data
print("load data..")
data = load_sentiment_sentences_and_labels()
print(len(data))
# Q: What accuracy do you get when you run the code? What is weird?
print("split data..")
split_point = int(0.75*len(data))

sentences = [sentence for sentence, label in data]
labels = [label for sentence, label in data]
X_train, X_test = sentences[:split_point], sentences[split_point:]
y_train, y_test = labels[:split_point], labels[split_point:]

print("#train instances: {} #test instances: {}".format(len(X_train),len(X_test)))
assert(len(X_train)==len(y_train))
assert(len(X_test)==len(y_test))

## Explain to your neighbor, what happens here?
majority_label = Counter(labels).most_common()[0][0]
majority_prediction = [majority_label for label in y_test]

print("vectorize data..")
vectorizer = CountVectorizer()

classifier = Pipeline( [('vec', vectorizer),
                        ('clf', LogisticRegression())] )

### Q2: add code to train and evaluate your classifier
print("train model..")
## your code here:

##
print("predict..")
## your code here: (instantiate y_predicted)


###
print("Accuracy:", accuracy_score(y_test, y_predicted))

print("Majority baseline:", accuracy_score(y_test, majority_prediction))

# you can access the vectorizer (feature to feature number mapping) this way
# print(vectorizer.vocabulary_.get('the'))
