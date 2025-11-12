For RNNs, it's difficult to handle word that are far from each other but have dependencies.
Moreover, GPUs work well in doing parallelizable tasks such as matrix multiplication. For recurrent jobs, GPUs can't work well.
**Transformers** can take advantages of parallel computing of GPUs.
Time cost for transformers is $O(dn^2)$ while for RNN is $O(d^2n)$.
In which, $d$ is the dimension of word vectors and $n$ is the sequence length.
# Self-attention
## Similarity with RNN's Attention
Recall the Attention process in [[Advanced Topics of RNN]]
The whole process of **Query**-**Key**-**Value**.
$$
e_i=s_t^T(U^TV)h_i=(Us_t)^T(Vh_i)
$$
To the processing token $s_t$, we map it with $U$, and get a query vector of $s_t$.
Then mapping $h_i$ with $V$ is the process of get the key.
Multiply query vector and key vector, we get the weight for each value.
$$
\alpha^t=softmax(e^t)
$$
$$
a_t=\sum_{i=1}^N\alpha_i^th_i
$$
## Q-K-V for Transformers
Let $w_{1:n}$ be a sequence of words in corpus $V$.
$E$ is the whole word Embeddings $d\times |V|$.
For each word $w_i$, we have its word vector $x_i(d\times 1)=Ew_i$.
Then, we have three mapping matrices $Q,K,V$, which are all $d\times d$.
For every word in the sentence $w_i$, we take the mapping
$$
q_i=Qx_i
$$
$$
k_i=Kx_i
$$
$$
v_i=Vx_i
$$
Then, we can build a attention distribution from any one word $i$ to all words in the sentence.
$$
e_{ij}=q_i^Tk_j
$$
Use softmax to map the results to a distribution
$$
a_{ij}=\frac{\exp(e_{ij})}{\sum_{j^{'}}\exp(e_{ij^{'}})}
$$
Do a weighted sum of output softmax
$$
o_i=\sum_ja_{ij}v_j
$$
## Vectorization
The math process can be vectorized.
Let $X=[x_1;x_2;......;x_n]$ which is a $n\times d$ matrix that holds the all word embeddings.
The same with before, we have $Q,K,V\in R^{d\times d}$.
So we have $XQ,XK,XV\in R^{n\times d}$.
Then we can work out the output $o$ of this Q-K-V layer directly.
$$
o=softmax(XQ(XK)^T)XV
$$
It's really fast for GPUs.
## Piling Up for more Layers
A layer of QKV takes in a sequence of word vectors and outputs a same-length sequence of word vectors.
Only one layer can't work the best. We can use more layers of QKV self-attention.
But in fact, there are no element-wise nonlinearities so far. In math, stacking more such layers are just re-averaging the **value vectors**.
The solution is to add a **Feed-forward Network** to post-process each output vector.
$$
m_i=MLP(o_i)=W_2ReLU(W_1o_i+b_1)+b_2
$$
So the big picture looks like
![[截屏2025-10-27 23.33.03.png]]
We can call a layer of attention(including Q-K-V and Feed-forward Network) a Transformer **Building Block**.
# Sequence Order
In math, in the process of Q-K-V, every word is dealt with equally.
In another word, no matter in what order the original sequence is, the outputs are all the same.
## Sequence Index Vector
We need to give the model a notation of sequence.
To trade off, we have to fix the max length of the input sequence to $n$, and pad the rest places with default vectors.
We initialize $n$ $d\times 1$ vectors $p_i$ for $i\in \begin{Bmatrix}1,2,......,n\end{Bmatrix}$.
For the input word vector $x_i$ in index $i$, we add the $p_i$.
$$
\tilde x_i=x_i+p_i
$$
Pass the modified vector $\tilde x$ to the self-attention layer.
We can learn all the $p_i$ while training.
Then, each $p_i$ is a distinct sign for every index, which can be used to represent the sequence order.
## Rotary Position Embedding(RoPE)
### 2-d Vector Rotation
To a 2-d vector $v=(a,b)$, we have its complex $c_v=a+bi$.
Then $e^{i\theta}c_v$ is the complex of the vector that is the $v$ rotated by angle $\theta$.
The process can also be replaced by a matrix multiplication(a $2\times 2$ matrix multiplies a $2\times 1$ vector).
If two 2-d vectors are rotated by a same angle $\theta$, their inner product doesn't change.
If $v_1$ is rotated $\theta_1$ and $v_2$ rotated $\theta_2$, the only value that influences the their inner product  is $\theta_1-\theta_2$.
### Rotation of High-dimension Vectors for Transformers
![[截屏2025-10-28 09.52.06.png]]
Break the $d$-dimension vector into $d/2$ $2$-dimension vectors.
For each $2$-dimension vector, do rotation respectively.
Then concatenate all the results back to a $d$-dimension vector.
This process can also be vectorized.
### Rotate for Sequence
For every $q_i$ and $k_i$, we rotate them with a angle $\theta_i = mi$, in which $m$ is a constant, and $i$ is the index of their token in the original sentence.
So, the sequence order of the input sentence is perfectly described.
Note that what matters in this process is the distance of two words in the sentence(**Relative Position**) instead of the **Absolute Position** of each word.
# Masking
To use self-attention in decoders, the model should not know the future while predicting next words.
We only take out the already generated words to Query.
While dealing with future words, we mask them by setting the attention score to $-\infty$.
$$
e_{ij}=\begin{cases}q_i^Tk_j,j\leq i\\-\infty,j>i\end{cases}
$$
![[截屏2025-10-28 10.25.58.png]]
Also, instead of setting $-\infty$ scores, we can pad some `<EMPTY>` tokens to the blank that haven't been generated yet.
# Multi-head Self-attention
## Independent $Q,K,V$s
In self-attention, we only have a group of $Q,K,V$.
The Multi-head Self-attention introduces several groups of $Q_l,K_l,V_l$s.
Each group extract information from the input features independently(some catches syntax and some catches dependency for example).
## Implement
For word vectors with dimension $d$, and $h$ heads, we let
$$
Q_l,K_l,V_l\in R^{d\times \frac{d}{h}}
$$
in which $l$ ranges from $1$ to $h$.
Each attention has head performs attention and give output independently
$$
o_i=softmax(XQ_l(XK_l)^T)XV_l
$$
$XQ_l,XK_l,XV_l$ are all $n\times \frac{d}{h}$.
So it is $n\times n$ in the softmax.
Note that softmax always do the mapping along the last axis. For this matrix, it takes each row in the matrix as input and outputs a mapped row.
$o_i$ is $n\times \frac{d}{h}$.
Concatenate all $o_i$ for $i\in\begin{Bmatrix}1,2,......,h\end{Bmatrix}$, we get an output $n\times d$ vector. Them objective with a $d\times d$ matrix $Y$(can help integrate outcomes from different heads).
$$
o=[o_1,o_2,......,o_h]Y\in R^{n\times d}
$$
## Vectorization
Take $Q$ as an example.
First, concatenate all $Q_l$ to form a $d\times d$ matrix $Q$.
Compute $XQ$, which is $n\times d$.
Break the second dimension down to 2 new dimensions, then the shape goes to $n\times h\times\frac{d}{h}$.
Transpose the tensor to $h\times n\times\frac{d}{h}$.
The $h$ is the "Heads" axis(piles up the matrices of each heads).
Do the same operations to $K$ and $V$.
![[截屏2025-10-28 11.43.06.png]]
The whole expression is still
$$
o=softmax(XQ(XK)^T)XVY
$$
Now, $XQ$ is $h\times n\times\frac{d}{h}$. $(XK)^T$ is $h\times\frac{d}{h}\times n$.
$XQ(XK)^T$ is $h\times n\times n$.
Still, softmax is done to the last dimension.
*View the tensor $XQ(XK)^T$ as a $h$-floor building. On each floor there is a distinct matrix. The softmax is done to each row in each floor.*
Then we multiply it with $XV$ which is also $h\times n\times\frac{d}{h}$.
The output is a $h\times n\times\frac{d}{h}$ tensor.
Then, pile the first and third dimension up(just like the building fall horizontally) and form a $n\times d$ matrix.
Multiply it with $Y$ to get the outcome of this layer.
*The $Y$ are all optional.*
## Scaled Dot Product
As $d$ becomes larger, dot products between vectors tend to become large.
We should rescale it in each attention layer.
So, the update version of attention layer is
$$
o=softmax(\frac{XQ(XK)^T}{\sqrt{d/h}})XVY
$$
# Transformer Architecture
## Basic
The basic architecture is after several building blocks, we connect to Linear Layer and Softmax Layer to get some outputs.
![[截屏2025-10-28 18.15.48.png]]
## Residual Connection
Before and After a Layer, we have $X^{(i-1)}$ and $X^{(i)}$, we have
![[截屏2025-10-28 18.49.39.png]]
$$
X^{(i)}=Layer(X^{(i-1)})
$$
The additional Residual Connection links directly $X^{(i-1)}$ and $X^{(i)}$. Which makes training more smooth.
![[截屏2025-10-28 18.52.18.png]]
$$
X^{(i)}=Layer(X^{(i-1)})+X^{(i-1)}
$$
![[截屏2025-10-28 18.55.02.png]]
With some residual connections, we are more likely to find the global optimize by gradient descent.
![[截屏2025-10-28 18.54.08.png]]
## Layer Normalization
After each layer, some word vectors may be larger in scale than other vector.
Then, we should do a independent normalization for each word.
For each word's vector $x\in R^d$.
Let the mean
$$
\mu=\frac{1}{d}\sum_{i=1}^dx_i
$$
and the standard deviation
$$
\sigma=\sqrt{\frac{1}{d}\sum_{i=1}^d(x_i-\mu)^2}
$$
Then, we standardize the initial $x$ and use $\gamma,\beta\in R^d$ to map it.
$$
o_x=\frac{x-\mu}{\sqrt{\sigma}+\epsilon}\odot\gamma+\beta
$$
in which, $\epsilon$ is a small scalar to prevent against divide zero problem.
## Encoder and Decoder
The encoder needs no mask. It needs to generate features based on whole inputs.
![[截屏2025-10-28 19.40.55.png]]
The decoder needs mask.
![[截屏2025-10-28 19.36.37.png]]
## Cross-attention
![[截屏2025-10-28 19.47.08.png]]
Suppose in the current state, the decoder has generated $t$ tokens $z_1,z_2,......,z_t$. We concatenate these $d$-dimensional vectors to generate a whole matrix $Z=[z_1;z_2;......,z_t]\in R^{t\times d}$.
The encoder has $T$ tokens $h_1,h_2,......,h_T$. We concatenate these $d$-dimensional vectors to generate a whole matrix $H=[h_1;h_2;......;h_T]\in R^{T\times d}$.
Now, we use $Z$ to Query, and use $H$ to Key and Value.
First, we calculate $ZQ$, which is $t\times d$.
Then, calculate $HK$ which is $T\times d$.
Multiply $ZQ(HK)^T$, which is $t\times T$. Each row refers to the all attention scores in terms of each $z_i$. This indicates which input token should each output(docoder-generated) token should focus more in.
Do softmax to each row to get the attention distribution, multiply the result matrix with $HV$ and get the output.
The whole process is described as
$$
o=softmax(ZQ(HK)^T)HV
$$
Sometimes, we can pad some tokens to make the number of decoder features from $t$ tokens to $T$ tokens which is the same as encoder(like masking).