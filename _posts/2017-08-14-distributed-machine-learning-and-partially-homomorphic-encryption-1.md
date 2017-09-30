---
title: 'Distributed machine learning and partially homomorphic encryption (part 2)'
summary: How to apply an encrypted model to score remote data
layout: post
date: 2017-08-14
permalink: /posts/2017/08/14/distributed-machine-learning-and-partially-homomorphic-encryption-2/
tags:
  - privacy
  - homomorphic encryption
  - federated learning
---

*The post appeared originally on the [n1analytics blog](https://blog.n1analytics.com/distributed-machine-learning-and-partially-homomorphic-encryption-2/).*

<br>

## Predicting with an Encrypted Model

In [a previous post](/posts/2017/07/13/distributed-machine-learning-and-partially-homomorphic-encryption-1/)
we demonstrated the use of our [python-paillier](https://github.com/n1analytics/python-paillier)
library for implementing a simple secure protocol for federated learning. In this post, we will
explore how an encrypted model can be used to _score_ remote data. The viability of this
technical solution is interesting and relevant for privacy reasons. It means that the owner of
the model (and of the training data) won't need to compromise the privacy of the remote data
owner in order to score their data; and vice-versa, the remote data owner is blind to any
information about the scoring model (and therefore the training data), since the model itself
is encrypted.

We will assume some understanding of the Paillier cryptosystem and also of [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression). This example was inspired by
the excellent [blog post](https://iamtrask.github.io/2017/06/05/homomorphic-surveillance/)
of [@iamtrask](https://twitter.com/iamtrask).

We use a subset of Enron spam email dataset. Alice trains a spam classifier on emails she
owns. She wants to apply it to Bob's personal e-mails, without:

1. Asking Bob to send his e-mails anywhere.
2. Leaking information about the learned model or the dataset she has learned from.
3. Letting Bob know which of his e-mails are spam or not.


The full code is available on [github](https://github.com/n1analytics/python-paillier/blob/master/examples/logistic_regression_encrypted_model.py).
First we make the necessary imports and wrap the code for downloading and preparing the data.

~~~python
import time
import os.path
from zipfile import ZipFile
from urllib.request import urlopen
from contextlib import contextmanager

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import phe as paillier

np.random.seed(42)

# Enron spam dataset hosted by https://cloudstor.aarnet.edu.au
url = [
    'https://cloudstor.aarnet.edu.au/plus/index.php/s/RpHZ57z2E3BTiSQ/download',
    'https://cloudstor.aarnet.edu.au/plus/index.php/s/QVD4Xk5Cz3UVYLp/download'
]


def download_data():
    """Download two sets of Enron1 spam/ham e-mails if they are not here
    We will use the first as trainset and the second as testset.
    Return the path prefix to us to load the data from disk."""

    n_datasets = 2
    for d in range(1, n_datasets + 1):
        if not os.path.isdir('enron%d' % d):

            URL = url[d-1]
            print("Downloading %d/%d: %s" % (d, n_datasets, URL))
            folderzip = 'enron%d.zip' % d

            with urlopen(URL) as remotedata:
                with open(folderzip, 'wb') as z:
                    z.write(remotedata.read())

            with ZipFile(folderzip) as z:
                z.extractall()
            os.remove(folderzip)
~~~

For simplicity, emails are represented as a vector of word in a restricted vocabulary, where each feature value
counts the number of time a word appeared in the email. We use a [CountVectorizer](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
for this.

~~~py
def preprocess_data():
    """
    Get the Enron e-mails from disk.
    Represent them as bag-of-words.
    Shuffle and split train/test.
    """

    print("Importing dataset from disk...")
    path = 'enron1/ham/'
    ham1 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
            for f in os.listdir(path) if os.path.isfile(path + f)]
    path = 'enron1/spam/'
    spam1 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
             for f in os.listdir(path) if os.path.isfile(path + f)]
    path = 'enron2/ham/'
    ham2 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
            for f in os.listdir(path) if os.path.isfile(path + f)]
    path = 'enron2/spam/'
    spam2 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
             for f in os.listdir(path) if os.path.isfile(path + f)]

    # Merge and create labels
    emails = ham1 + spam1 + ham2 + spam2
    y = np.array([-1] * len(ham1) + [1] * len(spam1) +
                 [-1] * len(ham2) + [1] * len(spam2))

    # Words count, keep only frequent words
    count_vect = CountVectorizer(decode_error='replace', stop_words='english',
                                 min_df=0.001)
    X = count_vect.fit_transform(emails)

    print('Vocabulary size: %d' % X.shape[1])

    # Shuffle
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm, :], y[perm]

    # Split train and test
    split = 500
    X_train, X_test = X[-split:, :], X[:-split, :]
    y_train, y_test = y[-split:], y[:-split]

    print("Labels in trainset are {:.2f} spam : {:.2f} ham".format(
        np.mean(y_train == 1), np.mean(y_train == -1)))

    return X_train, y_train, X_test, y_test
~~~

The scenario works as follows. Alice trains a spam classifier with logistic regression on the data she
possesses. After learning, she generates a public/private key pair using the Paillier cryptoscheme.
The model is encrypted using the public key. The public key and the encrypted model are sent to Bob. Bob
applies the encrypted model to his own data, obtaining encrypted scores for each email. Bob sends these
encrypted scores to Alice. Alice decrypts them with the private key to obtain the predictions spam vs.
not spam.

This protocol satisfies the three conditions stated above. In particular, Bob only sees encrypted model
and encrypted scores and cannot get anything out of it without knowledge of the private key.

Now to the implementation. Alice needs to be able to perform logistic regression on plaintext data,
to encrypt the model for remote use and to decrypts encrypted scores using the private key.

~~~py
class Alice:

    def __init__(self):
        self.model = LogisticRegression()

    def generate_paillier_keypair(self, n_length):
        self.pubkey, self.privkey = \
            paillier.generate_paillier_keypair(n_length=n_length)

    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def encrypt_weights(self):
        coef = self.model.coef_[0, :]
        encrypted_weights = [self.pubkey.encrypt(coef[i])
                             for i in range(coef.shape[0])]
        encrypted_intercept = self.pubkey.encrypt(self.model.intercept_[0])
        return encrypted_weights, encrypted_intercept

    def decrypt_scores(self, encrypted_scores):
        return [self.privkey.decrypt(s) for s in encrypted_scores]
~~~

Bob is given the encrypted model and the public key. He must be able to
score local plaintext data with the encrypted model, but cannot decrypt
the scores without the private key held by Alice.

~~~py
class Bob:

  def __init__(self, pubkey):
      self.pubkey = pubkey

  def set_weights(self, weights, intercept):
      self.weights = weights
      self.intercept = intercept

  def encrypted_score(self, x):
      """Compute the score of `x` by multiplying with the encrypted model,
      which is a vector of `paillier.EncryptedNumber`"""
      score = self.intercept
      _, idx = x.nonzero()
      for i in idx:
          score += x[0, i] * self.weights[i]
      return score

  def encrypted_evaluate(self, X):
      return [self.encrypted_score(X[i, :]) for i in range(X.shape[0])]
~~~

Let's see the script in action. We get the data in order first and also inspect the dimensionality of the problem:

~~~py
>>> download_data()
>>> X, y, X_test, y_test = preprocess_data()
>>> X.shape
(500, 7994)
~~~

We are dealing with about 8000 features.
Next we instantiate Alice, who generates the key pair and fits her logistic model on local data.

~~~py
>>> alice = Alice()
>>> alice.generate_paillier_keypair(n_length=1024)
>>> alice.fit(X, y)
~~~

No encryption has been performed yet. Let's just see what the error of
Alice's classifier would be _if_ she had access to Bob's raw
(unencrypted) data. Of course, this would _not_ be possible to know in
a realistic scenario as Bob's data would not be available.

~~~pypy
>>> np.mean(alice.predict(X_test) != y_test)
0.045683350745559882
~~~

Now, Alice encrypts the classifier.

~~~py
>>> encrypted_weights, encrypted_intercept = alice.encrypt_weights()
~~~

We instantiate Bob with Alice's public key. Bob scores by using the encrypted classifier.

~~~py
>>> bob = Bob(alice.pubkey)
>>> bob.set_weights(encrypted_weights, encrypted_intercept)
>>> encrypted_scores = bob.encrypted_evaluate(X_test)
~~~

Let's see how one of those encrypted scores look like.
~~~py
>>> print(encrypted_scores[0].ciphertext())
4975557101598019607333115657955782044002134197013151844631125970114580057948777697681679333578395930647500175104718976826465398554390717765586649503985800812276599674119580862642667636337378406851541955675614078001941547394030888287811317521894539431449722023192072949095429036555137484530752817765976765269293455734683337022787581827841503790798807907517815490376905382493360989832127082449724104557596689227300380104999472764265118788640333048806552912736240459059453425987302997946039793991525213509904102136530661457492688678688561944802008308534596837051863930132631396095952823207091622450117172795188329566587
~~~


Alice decrypts Bob's scores.
~~~py
>>> scores = alice.decrypt_scores(encrypted_scores)
>>> scores[:5]
[-14.511058062671882,
 -9.188384491859484,
 -1.746647646814274,
 -16.91595050694431,
 -6.716934039494412]
~~~

The sign of those scores is equivalent to the predicted class. As a sanity check, let's see what the error of
this model is. Keep in mind that this is not known to Alice, who does not possess Bob's ground truth labels.
The error is the same as above.

~~~py
>>> np.mean(np.sign(scores) != y_test)
0.045683350745559882
~~~

The full code of this second example is available
[here](https://github.com/n1analytics/python-paillier/blob/master/examples/logistic_regression_encrypted_model.py),
when run it will output timing information relative to each step of the protocol.

_Bonus_. You may ask: can this protocol and the one from the previous post be merged? Indeed they can, modulo the fact that the former does classification and the latter regression. In principle, you could set up a federated learning scenario where models trained by a client are deployed remotely in encrypted form and then predictions are sent back to that client.

Thanks to the all [n1analytics](http://www.n1analytics.com/) team for feedback and suggestions
in writing this post.
