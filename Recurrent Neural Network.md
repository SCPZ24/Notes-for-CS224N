# Language Modeling
The task of a language model is to predict the next word given the formal several context words.
For example, when typing some words in a keyboard, the phone will predict some potential next words.
**Recurrent Neural Network(RNN)** is a common tool to build a language model.
# Brief Structure of RNN
![[截屏2025-10-23 01.08.40.png]]
A sentence is actually a sequence of words.
RNN can take in previous words and predict the upcoming words.
# Generating Words
## Viewing the Context
Look up the embedding of each word in the sequence.
Initialize the hidden layer $h^{(0)}$.
From the beginning of the first known tokens, recurrently do a fixed calculation.
For the $i^{th}$ word $x^{(i)}$, look up for its word vector $e^{(i)}$. Then, multiply it with a fixed Matrix $W_e$.
Multiply the previous hidden layer $h^{(i-1)}$ with a fixed Matrix $W_h$.
Then, sum them up and add a bias $b_1$.
*$W_e,W_h,b_1$ are fixed parameters in the model.*
$$
z^{(i)}=W_hh^{(i-1)}+W_ee^{(i)}+b_1
$$
Then, to get the hidden layer of state $i$, we activate with a non-linear function $\sigma$.
Note that the activate function here is **tanh** instead of sigmoid. Briefly, sigmoid is always mapping values to positive, which will cause the values in hidden layers accumulatively become larger as the model propagates. **tanh** is fair for both positive and negative values.
$$
h^{(i)}=\sigma(z^{(i)})
$$
## Predicting
After processing the last known word, we can begin predicting.
First, linear the last hidden layer to a $|V|\times 1$ vector, in which $|V|$ is the size of the whole corpus(the number of tokens in the corpus).
$$
Z^{(t)}=Uh^{(t)}+b_2
$$
Activate $Z^{(t)}$ with a softmax. Use the largest value's index to do predict the next word.
$$
\hat{y}^{(t)}=softmax(Z)
$$
Then, use the predicted word's embedding $e^{(t+1)}$ as input for next roll's prediction.
# Training
## Loss Function for One Step
For single step $t$, the Loss Function is a Cross Entropy Loss.
$$
J^{(t)}(\theta)=CE(y^{(t)},\hat y^{(t)})=-\sum_{w\in V}y_w^{(t)}\log\hat y_w^{(t)}=-\log\hat y_{x_{t+1}}^{(t)}
$$
## Loss Function for a sequence
Suppose the length of a sequence is $T$.
Then, the loss of the sequence is the average of the loss of all steps.
$$
J(\theta)=\frac{1}{T}\sum_{t=1}^TJ^{(t)}(\theta)=\frac{1}{T}\sum_{t=1}^T-\log\hat y_{x_{t+1}}^{(t)}
$$
## Back Propagation Through Time
![[截屏2025-10-23 20.57.19.png]]
RNN uses same parameters to forward propagate. So we take derivatives to same Parameters each step.
Take $W_h$ as an example.
$$
\frac{\partial J^{(t)}}{\partial W_h}=\sum_{i=1}^t\frac{\partial J^{(t)}}{\partial W_h}\bigg|_{(i)}
$$
It's in the form of sum because $J^{(t)}$ is influenced by $h^{(t)}$, and $h^{(t)}$ has relationship with $h^{(1)},h^{(2)},h^{(3)},......,h^{(t-1)}$.
While training, the workload to calculate the gradient rockets up in exponential scale.
So we need to truncate the derivatives(use a pseudo derivative for latter layers) and update the parameters of the model after a fixed number of steps.
## Batching
Scanning through the whole corpus takes up a lot of time. We can use batches to do it faster.
Naturally, we use a sentence or a fixed number of sentences to do batch training.
Compute loss for a batch(or a truncate point), compute gradients and update weights. Repeat on a new batch of sentences.
# Attributes of RNN
1. RNN can process any length of context(but some words far from the current word are easy to forget).
2. Model size is constant. It won't increase for longer input context.
3. Same weights applied on every time step, so there is **symmetry** in how inputs are processed.
4. Recurrent computation is slow(repeat doing matrix multiplication).
# Perplexity
The standard evaluation metric for Language Models is perplexity.
A bigger perplexity indicates that the model feels more confused when predicting the next word given previous words.
$$
perplexity=\prod_{t=1}^T(\frac{1}{P_{LM}(x^{(t+1)}|x^{(t)},x^{(t-1)},......,x^{(1)})})^{1/T}$$
Specifically,
$$
perplexity=\prod_{t=1}^T(\frac{1}{\hat y_{x_{t+1}}^{(t)}})^{1/T}=\exp(\frac{1}{T}\sum_{t=1}^T-\log\hat y_{x_{t+1}}^{(t)})=\exp(J(\theta))
$$
it is directly related to the Loss.
So, the smaller the perplexity is, the better the model predicts.
# Vanishing and Exploding Gradients
Intuitively, an RNN model can't learn from contexts that are far away.
In math, one of the explain is vanishing and exploding of gradients.
## Vanish
![[截屏2025-10-24 22.24.04.png]]
Chain Rule is used when working out gradients in RNN.
When $x\in (-1,1)$, $\tanh(x)$ is approximately equal to $x$.
Approximately, we have
$$
\frac{\partial h^{(t)}}{\partial h^{(t-1)}}\approx W_h
$$
After propagating for $t$ layers, we'll get $W_h^t$, and we consider eigenvalues $\lambda_1,\lambda_2,......,\lambda_n$ and eigenvectors $q_1,q_2,......,q_n$, we have
$$
W_h^t=\sum_{i=1}^nc_i\lambda_i^tq_i
$$
As $t$ get larger and $|\lambda_i|<1$, $\lambda_i^t$ is sure to be pretty small.
It's obvious that some sub-directions of the gradient will be close to $0$, leading to what's called **Vanish**.
The model can't learn because the gradient vanishes.
## Explode
The same factor will also contribute to gradient **Explode**, which means the gradient goes much more big in one step.
This will lead to bad steps.
![[截屏2025-10-24 22.25.37.png]]
One way to solve Explode is to set a threshold.
If the norm of the gradient is larger than the threshold, re-scale the gradient to a small one(but pointing to the same direction).
## Generalize
Gradient vanish and explode is not just a problem of RNN.
As the layers go deep, the problems are common to meet.
Some knows solutions such as "ResNet" and "DenseNet" are all based on creating **skip-connections** between layers that are not close to each other.