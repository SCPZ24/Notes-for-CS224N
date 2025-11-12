# ConvNet
## Basics
Apart from doing convolution to a 2-D image, we can do convolution to languages.
Typically, it should be using a 1-D convolutional layer.
To the first layer, the number of input channel is equal to the dimension of word embeddings.
![[截屏2025-11-12 09.03.47.png]]
After a layer of convolution, we can then apply **MaxPool** or **AveragePool** to the features.
Also, we can set strides to convolutional filters or pools.
![[截屏2025-11-12 09.09.07.png]]
Note that pooling can do globally(take average or max through all output features) or locally(only do pooling in a $n$-gram sliding windows)
We can use $k$-max pooling: not use max number of a channel, but use top $k$ largest values in a channel, to capture some more informations.
## Single Layer CNN
A single CNN can be just one layer: convolution-pooling.
It can do some sentence classification.
The inner process of doing this is just doing vector product.
If word embedding is $d$-dimensional, and kernel size is $h$.
We concatenate $h$ adjacent vectors and form a longer $hd\times 1$ vector.
$$
x_{k:k+h-1}=[x^T_k,x^T_{k+1},...,x^T_{k+h-1}]^T
$$
Then, we do convolution.
$$
c_k=f(W^Tx_{k:k+h-1}+b)
$$
in which, $W$ is $hd\times 1$ vector while $f$ is activation function.
Then, we get the result for this full channel.
$$
c=[c_1,c_2,...,c_{n-h+1}]\in R^n-h+1
$$
We pool it to extract some information.
$$
\hat{c}=max(c)
$$
We can use multi-channels($m$ as example). So output of this layer can be
$$
z=[\hat c_1,\hat c_2,...,\hat c_m]
$$
Then do linear and logistic regression or softmax to adapt for downstream tasks.
## Mixed Kernel Size
Sometimes we can mix some kernel size, do different grams and mix them.
It can capture information of different scales of language.
![[截屏2025-11-12 11.58.16.png]]
## Batch Normalization
Transform the convolution output of a batch by scaling the activations to have zero mean and unit variance.
To be specific, for each channel, we do normalization among all elements in one channel.
If the activation feature is $batchSize\times channel\times n$, Then we do normalization to each $n$-elements.
Different from Layer Normalization in [[Transformers]], BatchNorm normalizes across all elements and items in a batch for each feature independently. While LayerNorm calculates statistics across all feature dimensions for each instance independently.
## Size 1 Convolution
In CV, size 1 convolution can serve as a **Full-connection** layer.
In NLP, intuitively, it can re-comprehend the meaning of a word.
Also, it can be used to map from many channels to fewer channels.
## Deep Convolutional Neural Networks
Here we take VD-CNN as example.
![[截屏2025-11-12 12.19.09.png]]
ResNets are applied here.
A single **Convolutional Block** looks like as follows.
![[截屏2025-11-12 12.20.32.png]]
# Tree Recursive Neural Network
## Constituency Sentence Parsing
A sentence can be parsed into a tree, according to relationships of different words.
![[截屏2025-11-12 12.22.55.png]]
Compared to RNN, Tree Recursive Neural Network can capture more prefix context.
![[截屏2025-11-12 12.25.19.png]]
## Parsing a Sentence
Get a pair-wise score for each adjacent token pair.
![[截屏2025-11-12 12.36.42.png]]
Form a binary tree struct on pair that wins the highest score.
![[截屏2025-11-12 12.37.38.png]]
Continue to form a complete tree.
## Score and New Vector
There is a single way to get a new score and a new vector of a combination of two origin vectors.
The new combined vector $p$ is
$$
p=\tanh (W\begin{bmatrix}c_1\\c_2\end{bmatrix}+b)
$$
The score can be mapped out from $p$.
$$
s=U^Tp
$$
note that same $W$, $b$ and $U$ are applied to every node pair.
Here, some multiplicative relations of $c_1$ and $c_2$ may be missed.
We can also work out the $p$ as follows
$$
p=f(\begin{bmatrix}c_1\\c_2\end{bmatrix}^TV\begin{bmatrix}c_1\\c_2\end{bmatrix}+W\begin{bmatrix}c_1\\c_2\end{bmatrix})
$$
Here $f$ can be sigmoid, serving as a logistic regression to indicate whether to combine the two nodes.(This can make training easier: prepare some manual labeled parsing process, logistic regress which nodes to combine every step.)
## Usage
Tree Recursive Neural Network can be used to do some sentence classification.
![[截屏2025-11-12 13.01.42.png]]
With a tree structure, the classification of positive/negative can be more naturally indicated. Because the meanings of human languages are more like a tree structure rather than a linear structure.