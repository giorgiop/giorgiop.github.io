---
title: 'Distributed machine learning and partially homomorphic encryption (part 1)'
summary: A demonstration of using Paillier encryption for federated machine learning
layout: post
date: 2017-07-13
permalink: /posts/2017/07/13/distributed-machine-learning-and-partially-homomorphic-encryption-1/
tags:
  - privacy
  - homomorphic encryption
  - federated learning
---

*The post appeared originally on the [n1analytics blog](https://blog.n1analytics.com/distributed-machine-learning-and-partially-homomorphic-encryption-1/).*

<br>


In this post, we will give a demonstration of the
usage and flexibility of our [python-paillier](https://github.com/n1analytics/python-paillier)
library as a tool for more secure machine learning. We will assume some basic
knowledge about [Paillier partially homomorphic encryption](https://bitsofpy.blogspot.com.au/2016/11/open-source-paillier-libraries.html),
and [linear regression](https://en.wikipedia.org/wiki/Linear_regression).

In particular, we will set up a simple secure protocol for federated machine learning,
inspired by recent [Google's work](https://research.googleblog.com/2017/04/federated-learning-collaborative.html)
on the topic.

## Introduction to the API

Let's start with a quick demo of the API. First thing, let's
create public and private keys by using a key length in bits long enough to get decent
cryptographic guarantees:

~~~py
>>> import phe as paillier
>>> pubkey, privkey = paillier.generate_paillier_keypair(n_length=1024)
~~~

Paillier is an asymmetric cryptoscheme (like RSA), where the public key is used
for encryption and the private key for decryption.

~~~py
>>> secret_numbers = [3.141592653, 300, -4.6e-12]
>>> encrypted_numbers = [pubkey.encrypt(x) for x in secret_numbers]
>>> [privkey.decrypt(x) for x in encrypted_numbers]
[3.141592653, 300, -4.6e-12]
~~~

But what do these encrypted numbers look like? You can open up the object and
look inside at the integer representation.

~~~py
>>> print(encrypted_numbers[0])
<phe.paillier.EncryptedNumber object at 0x7f02f2849dd8>
>>> print(encrypted_numbers[0].ciphertext())
5072752399058920189730182586811912902463474480667712432717959774819587074489325225214240998778150373197112637448816662931016970373407389034275190558182343858721113940870709409924017166597407543355101815707936636905640749963575027963216011646497564724153729103147138747511854121934327406877629294760278241554316409859573065893681767802219202771728963191523152254974808451269262932426358339707361034738737940843867971577772899191177890333880357061518134745146513228505813785268901991647262058355794072849790632418679961213162239495600291127208408082882305219363330327154890172539087918477378211986323886814727480557038
~~~

Paillier encryption is a great tool for preserving simple arithmetic operations in the
encrypted space. We can sum two encrypted numbers and decrypt the result and
that will be equal to the sum of the original numbers.

~~~py
>>> x, y = 2, 0.5
>>> encrypted_x = pubkey.encrypt(x)
>>> encrypted_y = pubkey.encrypt(y)
>>> encrypted_sum = encrypted_x + encrypted_y
>>> privkey.decrypt(encrypted_sum)
2.5
~~~

In the same way, multiplication of an encrypted number by a number in the clear
works.

~~~py
>>> z = 10
>>> privkey.decrypt(z * encrypted_x)
20
~~~

Notice that we cannot multiply two encrypted numbers together. This
is the limit of Paillier cryptosystem which is a _partially_ homomorphic encryption
scheme in contrast to _fully_ homomorphic.
Despite this limit, with those two allowed operations we can already play in
an interesting space, with a subset of linear algebra useful for
implementing machine learning primitives.


## Secure Federated Learning

In this example we assume we have sensitive data of 442 hospital patients,
with different level of progress of diabetes. Recorded variables are age,
gender, body mass index, average blood pressure, and six blood serum
measurements. A last variable is a quantitative measure of the disease
progression which we would like to predict from the previous variables.
Since this measure is continuous, we will solve the problem by performing
linear regression. The original data is hosted [here](http://www4.stat.ncsu.edu/~boos/var.select/diabetes.html)
and we access it via [sklearn](http://scikit-learn.org).

The data is distributed among 3 hospitals, referred as 'clients'. The objective
is to make use of the whole (virtual) training set to improve upon the
model that can be trained locally. Such a scenario is often referred to as
'horizontally partitioned'. Fifty patient records will be kept as a testset and not used
for training. An additional agent is the 'server', who will facilitate the
information exchange among the hospitals under the following constraints. Due
to privacy policy:

1. The individual patients' records data at each hospital cannot leave its premises,
not even in encrypted form
2. Even information/summary derived (read: gradients) from any individual
client's dataset cannot leave a hospital, unless it is first encrypted.
3. None of the parties (clients AND server) must be able to infer WHERE
(in which hospital) a patient in the training set has been treated.

Let's go to the code. We will use `numpy` and `sklearn` for this. The random
number generator is seeded explicitly to enable reproducibility of the experiment.

~~~py
import numpy as np
from sklearn.datasets import load_diabetes

import phe as paillier

seed = 42
np.random.seed(seed)
~~~

Let's prepare the data first, all wrapped into a function.

~~~py
def get_data(n_clients):

    diabetes = load_diabetes()
    y = diabetes.target
    X = diabetes.data

    # Add constant to emulate intercept
    X = np.c_[X, np.ones(X.shape[0])]

    # The features are already preprocessed
    # Shuffle
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm, :], y[perm]

    # Select test at random
    test_size = 50
    test_idx = np.random.choice(X.shape[0], size=test_size, replace=False)
    train_idx = np.ones(X.shape[0], dtype=bool)
    train_idx[test_idx] = False
    X_test, y_test = X[test_idx, :], y[test_idx]
    X_train, y_train = X[train_idx, :], y[train_idx]

    # Split train among multiple clients.
    # The selection is not at random. We simulate the fact that each client
    # sees a potentially very different sample of patients.
    X, y = [], []
    step = int(X_train.shape[0] / n_clients)
    for c in range(n_clients):
        X.append(X_train[step * c: step * (c + 1), :])
        y.append(y_train[step * c: step * (c + 1)])

    return X, y, X_test, y_test
~~~

From the learning viewpoint, notice that we are NOT assuming that each
hospital sees an unbiased sample from the same patients' distribution:
hospitals could be geographically very distant or serve a diverse population.
We simulate this condition by sampling patients NOT uniformly at random,
but in a biased fashion.
The test set is instead an unbiased sample from the overall distribution.

We also define some encrypt/decrypt operations on lists.

~~~py
def encrypt_vector(pubkey, x):
    return [pubkey.encrypt(x[i]) for i in range(x.shape[0])]


def decrypt_vector(privkey, x):
    return np.array([privkey.decrypt(i) for i in x])


def sum_encrypted_vectors(x, y):

    if len(x) != len(y):
        raise Exception('Encrypted vectors must have the same size')

    return [x[i] + y[i] for i in range(len(x))]
~~~

To evaluate the models, we will compute the mean square error between ground
truth and predicted labels.

~~~py
def mean_square_error(y_pred, y):
    return np.mean((y - y_pred) ** 2)
~~~

We perform linear regression by gradient descent. The server owns the private
key and the clients possess the public key. The protocol works as follows.
Until convergence:
1. Hospital 1 computes its gradient, encrypts it and sends it to hospital 2;
2. Hospital 2 computes its gradient, encrypts and sums it to hospital 1's;
3. Hospital 3 does the same and passes the overall sum to the server.
4. The server obtains the gradient of the whole (virtual)
training set; it decrypts it and sends it back in the clear to every client,
who can update the respective local models.

We assume that this aggregate
gradient does not disclose any sensitive information about individuals data
--- otherwise [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy)
 could be used on top of our protocol.

The next two classes implement the primitives necessary to server and clients
for running the protocol.

~~~py
class Server:
    """Hold the private key. Decrypt the average gradient"""

    def __init__(self, key_length=1024):
        self.pubkey, self.privkey = \
            paillier.generate_paillier_keypair(n_length=key_length)

    def decrypt_aggregate(self, input_model, n_clients):
        return decrypt_vector(self.privkey, input_model) / n_clients


class Client:
    """Run linear regression either with local data or by gradient steps,
    where gradients can be send from remotely.
    Hold the private key and can encrypt gradients to send remotely.
    """

    def __init__(self, name, X, y, pubkey):
        self.name = name
        self.pubkey = pubkey
        self.X, self.y = X, y
        self.weights = np.zeros(X.shape[1])

    def fit(self, n_iter, eta=0.01):
        """Linear regression for n_iter"""

        for _ in range(n_iter):
            gradient = self.compute_gradient()
            self.gradient_step(gradient, eta)

    def gradient_step(self, gradient, eta=0.01):
        """Update the model with the given gradient"""

        self.weights -= eta * gradient

    def compute_gradient(self):
        """Return the gradient computed at the current model on all training
        set"""

        delta = self.predict(self.X) - self.y
        return delta.dot(self.X)

    def predict(self, X):
        """Score test data"""
        return X.dot(self.weights)

    def encrypted_gradient(self, sum_to=None):
        """Compute gradient. Encrypt it.
        When `sum_to` is given, sum the encrypted gradient to it, assumed
        to be another vector of the same size
        """

        gradient = encrypt_vector(self.pubkey, self.compute_gradient())

        if sum_to is not None:
            if len(sum_to) != len(gradient):
                raise Exception('Encrypted vectors must have the same size')
            return sum_encrypted_vectors(sum_to, gradient)
        else:
            return gradient
~~~

Now we have all the necessary scaffolding. Let's set up a bunch of
parameters and get the data ready.

~~~py
>>> n_iter, eta = 50, 0.01

>>> names = ['Hospital 1', 'Hospital 2', 'Hospital 3']
>>> n_clients = len(names)
>>>
>>> X, y, X_test, y_test = get_data(n_clients=n_clients)
~~~

We instantiate server and clients. Each client gets the public key at creation
and its own local dataset.

~~~py
>>> server = Server(key_length=1024)
>>>
>>> clients = []
>>> for i in range(n_clients):
>>>     clients.append(Client(names[i], X[i], y[i], server.pubkey))
~~~

Each client trains a linear regressor on its own data.
What is the error (MSE) that each client would get on test set by training only
on its own local data?

~~~py
>>> for c in clients:
>>>     c.fit(n_iter, eta)
>>>     y_pred = c.predict(X_test)
>>>     print('{:s}:\t{:.2f}'.format(c.name, mean_square_error(y_pred, y_test)))
Hospital 1:	3933.78
Hospital 2:	4176.48
Hospital 3:	3795.95
~~~

Finally, the federated learning with gradient descent.

~~~py
>>> for i in range(n_iter):
>>>     # Compute gradients, encrypt and aggregate
>>>     encrypt_aggr = clients[0].encrypted_gradient(sum_to=None)
>>>     for i in range(1, n_clients):
>>>         encrypt_aggr = clients[i].encrypted_gradient(sum_to=encrypt_aggr)
>>>
>>>     # Send aggregate to server and decrypt it
>>>     aggr = server.decrypt_aggregate(encrypt_aggr, n_clients)
>>>
>>>     # Take gradient steps
>>>     for c in clients:
>>>         c.gradient_step(aggr, eta)
~~~

What is the error (MSE) that each client gets after running the protocol?

~~~py
>>> for c in clients:
>>>     y_pred = c.predict(X_test)
>>>     print('{:s}:\t{:.2f}'.format(c.name, mean_square_error(y_pred, y_test)))
Hospital 1:	3695.77
Hospital 2:	3855.14
Hospital 3:	3598.63
~~~

As expected, the MSE has decreased for every client. (They are not the same,
because the initial model for client was different, i.e. the best model on
local data.)

From the security viewpoint, we consider all parties to be "honest but curious".
Even by seeing the aggregated gradient in the clear, no participant can pinpoint
where patients' data originated. This is true if this RING protocol
is run by at least 3 clients, which prevents reconstruction of each others' gradient
simply by taking differences.

You can find the code of the full example [here](https://github.com/n1analytics/python-paillier/blob/master/examples/federated_learning_with_encryption.py).
Thanks to the all [n1analytics](http://www.n1analytics.com/) team for feedback and suggestions
in writing this post.
