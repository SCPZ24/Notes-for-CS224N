# Long Short-Term Memory RNNs
**LSTM** introduces some gates into the model.
![[截屏2025-10-24 23.03.38.png]]
Cell content is the container vector to store long-term memories.
The sigmoids in the forget gate, input gate and output gate are used to turn switch on($1$) or off($0$), in order decide which value in the coming vector remains and which discards.
For example, if the model judges that a certain dimension of $c^{(t-1)}$ should be forgotten, then the same dimension in $f^{(t)}$ should be nearly $0$. Then after element-wise product, the output in this dimension is nearly $0$.
So, the three gates serve as masks, filter datas according to which value should be passed down and which should be blocked.
The data flow in one layer can be shown in a graph.
![[截屏2025-10-24 23.16.44.png]]
Vanilla RNN has memory almost no more than 7 layers. But LSTM can enhance that value to about 100 layers.
# Usages of RNN
RNNs are sophisticated in processing sequences.
Some usages of RNN are all based on its capability on sequence processing tasks.
## Sequence Tagging
For example, we can sign Part-of-speech for every word in a sentence.
![[截屏2025-10-24 23.29.56.png]]
## Sentence Encoder Model
We can use RNN to verdict whether the sentence is positive or negative.
![[截屏2025-10-24 23.31.43.png]]
# Bidirectional RNN
## Implement
The RNN shown above only processes the sentence in one direction. This is only how humans accept information.
Moreover, this method only makes use of contexts from only one side.
We can reversely process the sequence and generate another sequence of vectors(a vector represents a hidden layer here).
Then, concatenate two vectors and gain a more comprehensive vector.
![[截屏2025-10-24 23.37.48.png]]
## Occasions to Use
Bidirectional RNN should be applied by default when the model can already know the whole sequence.
But when it comes to language models, this is of no use.
# Multi-layer RNN
The implement of Multi-layer RNN is shown below.
![[截屏2025-10-24 23.47.42.png]]
Often 2-layers is a lot better than 1-layer RNN, while 3-layers RNN might be only a little better than 2-layers RNN.
Usually, skip-connections/dense-connections are needed to train deeper RNNs(e.g., 8 layers).
# Neural Machine Translation
## Seq2seq Model
Neural Machine Translation(NMT) is a way to do **Machine Translation** in a single end-to-end neural network.
The neural network architecture is called a sequence-to-sequence model(a.k.a seq2seq) and it involves two RNNs: **Encoder RNN** and **Decoder RNN**.
![[截屏2025-10-24 23.54.37.png]]
## Predicting
First, go through the Encoder RNN and get the hidden layer(vector) of the last token.
Then, use this vector as the initial input layer of the Decoder RNN. Use this RNN to predict the translated texts step by step(`hidden layers->linear->softmax->choose the word->pass its embedding to next layer`).
## Training
To train the two RNNs, we should prepare a parallel dataset.
For example, enough Chinese-English parallel texts, in which a English sentence and a Chinese sentence with the same meaning as a pair.
![[截屏2025-10-25 00.57.47.png]]
The concept of multi-layer can also be applied here.
![[截屏2025-10-25 00.58.30.png]]
# Attention
Seq2seq described above can only use features in the last layer in the Encoder RNN. But usually, hidden features in last layer can't completely cover the message from previous layers.
**Attention** is a method to process some previous layers of the Encoder RNN. Just like a human translator, he won't focus on the last word of the sentence. Instead, he'll skip from one word to any another word while translating.
Attention builds shortcuts to faraway layers, which is helpful to handle gradient vanishing problem.
## Basic Implement
![[截屏2025-10-26 12.09.22.png]]
Given the Decoder RNN's hidden layers and Encoder RNN's hidden layers are same-dimensional vectors.
Suppose the encoder RNN has $N$ layers, each is a $h$-dimensional vector. They are $h_1,h_2,h_3,......,h_N$. When it comes to the $t$ layer of decoder RNN, suppose the hidden layer is a $h$-dimensional vector $s_t$.
Then, we can work out the attention score for each layer of every input tokens(a token refers to a layer in encoder RNN) in respect to $s_t$.
$$
e^t=[s_t^Th_1,s_t^Th_2,......,s_t^Th_N]=s_t^T[h_1,h_2,......,h_N]
$$
We have $e^t\in R^N$. Then we can map it with softmax to form Attention Distribution.
$$
\alpha^t=softmax(e^t)
$$
$\alpha^t$ is the weight for layers in encoder layer. It interprets how much attention should the model pay to each layer.
We can then based on this work out the additional $h\times 1$ input vector for the current layer in decoder RNN.
$$
a_t=\sum_{i=1}^N\alpha_i^th_i
$$
Then we can concatenate it with the original $s_t$.
$$
\begin{bmatrix}a_t\\s_t\end{bmatrix}\in R^{2h}
$$
Use this to predict and pass to next layer.
## Multiplicative Attention
Sometimes just use $e_i=s_t^Th_i$ can't fully represent the attention score. And it forces the $s_t$ and $h_i$ has same dimension.
We can use a mid matrix $W$.
If $s_t$ is $d_1\times 1$, $h_i$ is $d_2\times 1$. Then, $W$ is $d_1\times d_2$.
This time, we have
$$
e_i=s_t^TWh_i
$$
This method is called **Multiplicative Attention** or **Bilinear Attention**.
However, hidden layers are large. So the $W$ could be large. Then, many multiplications will be involved in only one layer. So, there's Reduced-rank Multiplicative Attention below.
## Reduced-rank Multiplicative Attention
We can decompress $W$ to two low rank matrices.
$$
W = U^TV
$$
In which, $U$ is $k\times d_1$, $V$ is $k\times d_2$.
$k$ is far more smaller than $d_1$ and $d_2$, to ensure that both $U$ and $V$ are low-rank matrices, which contains far less parameters.
Then we have
$$
e_i=s_t^T(U^TV)h_i=(Us_t)^T(Vh_i)
$$
Calculating in this order(after the second $=$ in expression above) can save much time.