# Distributed Representation
Naturally, we use context to learn about word.
So, we can build a dense vector for each word in **corpus**(word library/vocabulary).
By dot product, we can detect the similarity of given two words. Large result stands for large similarity while small one stands for small similarity.
Usually, a word vector is 300~500 dimensional. For some modern models, it can be around 1000.
# Word2Vector
Word2Vector is the algorithm to turn words in corpus to vectors.
There are two methods to achieve it: **Skip-grams(SG)** and **Continuous Bag of Words(CBOW)**.
SG predicts context words based on given center word.
CBOW predicts center word based on given context words.
Following is a demonstration of SG.
The process on the corpus is a sliding window.
![[截屏2025-10-13 19.16.55.png]]
## Loss Function
We use $\theta$ to represent all the parameters in the set of all vectors.
For a given sentence, the Likelihood for its appearance in our model is
$$
L(\theta)=\prod_{t=1}^T\prod_{-m\leq j\leq m,j\neq0}P(w_{t+j}|w_t;\theta)
$$
The sentence is already appeared. So to optimize out parameter, we should let it become bigger.
So, take the average of negative log likelihood as loss. The loss of out model is
$$
J(\theta)=-\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m,j\neq 0}\log P(w_{t+j},w_t;\theta)
$$
## Definition of $P(w_{t+j},w_t;\theta)$
For a word $w$, we should maintain 2 vectors.
$v_w$: $w$ is a center word.
$u_w$: $w$ is a context word.
So, for a corpus with $v$ words, each vector is $d$ dimensional, then we'll have $2vd$ words in total. All the parameters are initialized in random before optimizing.
Let $o$ be one of the context words while $c$ be the central word. We have
$$
P(o|c)=\frac{exp(u_o^Tv_c)}{\sum_{w\in V}exp(u_w^Tv_c)}
$$
It is a **`softmax`** function $R^n\rightarrow (0,1)^n$.
### Gradient of $v_c$
By taking derivatives using chain rule, we have
$$
\frac{\partial(\log P(o|c))}{\partial{v_c}}=u_o-\sum_{w\in v}P(w|c)u_w
$$
The outcome is in the form of $observed-expected$.
observed: $u_o$
expected: $\sum_{w\in v}p(w|c)u_w$ which is the expected vector of possible results when the center word is $c$.
### Gradient of every $u_w$
when $w = o$, we have
$$
\frac{\partial(\log P(o|c))}{\partial{u_o}}=v_c\cdot(1-P(o|c))
$$
when $w \neq o$, we have
$$
\frac{\partial(\log P(o|c))}{\partial{u_w}}=-v_c\cdot P(w|c)
$$
## Gradient Descent
Basing on this, we can form a descent for the whole $\theta$.
Given the learning rate $\alpha$, we have
$$
\theta^{new}=\theta^{old}-\alpha\nabla_\theta J(\theta)
$$
## Forming Word Vector
After training, it usually turns out that every word's $u$ and $v$ vectors are very close. 
We can take average of them to form a final vector for each word.
# More Sophisticated Way for Word2Vector
In fact, It's impossible to do GD or SGD on the mode above, because that process needs to iterate over all word in the corpus.
Instead, we can use **Negative Sampling**.
## Sample
Instead of pulling out every word, we just sample some words.
The words which are more frequently used in human language are more easily to be sampled out.
For every word $w$ in the corpus $V$ with total possibility of $U(w)$ to appear in human language, let
$$
p(w)=U^{3/4}(w)
$$
Then we work out the total possibility to Standard all the $p(w)$ to let the sum of them is $1$.
$$
Z=\sum_{w\in V}p(w)
$$
Just divide with $Z$, and get the final output for sampling
$$
P(w)=\frac{p(w)}{Z}
$$
With the operation of taking $U^{3/4}(w)$ in the beginning, we can let the words that have small possibility to appear weigh a little more(a bit more likely to be sampled) when sampling.
## Loss Function
Only for words that are not context of the given center word, we sample them.
Then, we need to let the machine know they are not similar with the center word(by do positive contribution to the loss).
We sample out $K$ such words per batch. They are called negative samples.
So, we can have the loss function
$$
J_{neg-sample}(u_o,v_c,U)=-\log\sigma(u_o^Tv_c)-\sum_{k\in K\,sampled\,words}\log\sigma(-u_k^Tv_c)
$$
Here, $\sigma(x)$ is the **Sigmoid** function.
The context word's vector $u_o$ make negative contributions to the loss while sampled word's vectors make positive contributions. That is suitable for optimizing.
## Stochastic Gradient Descent
We only need to update a few vectors related to the loss function: $u_o$, $v_c$, and every sampled word $w$'s $u_w$.
$$
\frac{\partial{J}}{\partial u_o}=[\sigma(u_o^T​v_c​)−1]⋅v_c​
$$
$$
\frac{\partial{J}}{\partial v_c}=[\sigma(u_o^T​v_c​)−1]⋅u_o​+\sum_{k\in K\,sampled\,words}​[\sigma(u_w^T​v_c​)]⋅u_w​
$$
$$
\frac{\partial{J}}{\partial u_w}=\sigma(u_w^T​v_c​)⋅v_c
$$
# Word Sense Ambiguity
Most of common words have different meanings.
Sure we can insert different copies of a same word into a corpus, and do some clustering to make each copy belongs to its relative area(train multiple word vectors for a single word).
But that is too difficult. Actually we can just do linear functions.
For a word $w$ which has $n$ meanings, let $f_1$,$f_2$,......,$f_n$ be frequency of each meaning.$v_1$,$v_2$,......,$v_n$ is each meaning's vector. So we have a final meaning vector
$$
v=\frac{1}{\sum_{i=1}^nf_i}\sum_{i=1}^nf_iv_i
$$
Although this process discards lots of information, but it usually works in a high-dimensional space.
# Named Entity Recognition(NER)
We can build a neural network to sign a word with some labels.
For example, labels like LOC,PER,DATE(location, person, date) to sign what a word refers to.
![[截屏2025-10-14 16.42.48.png]]
## Window Classification Using Binary Logistic Classifier
We can use a sliding window of a fixed size, sliding over the sentences and detect whether a word in the center position is a LOC, or a PER, or a DATE.
![[截屏2025-10-14 16.47.01.png]]
For the input of the neural network, we view the word vectors in the window to a big vector.
Then, we do some regression.
![[截屏2025-10-14 16.58.49.png]]
*$h=f(Wx+b)$ is a process that do Linear Transformation and then Activate the data. Actually we can use more than one layer of this rather than the process shown in the graph.*
## Cross-entropy Loss
The output of the neural network should be a vector that demonstrates whether the central word in the window is a LOC/PER/DATE.
For example, if the word is LOC, then the target vector to output is `[1,0,0]`.
Let $p$ be the target vector, and $q$ is the predicted vector output by the neural network. Both are n-dimensional vectors. $v_i$ is the $i^{th}$ dimension's value.
The loss is
$$
H(p,q)=-\sum_{i=1}^np_i\log q_i
$$
*Note that this formula of cross-entropy loss is only suitable for the circumstance that there is only 1 `1` in the target vector and the else are all `0`.*
