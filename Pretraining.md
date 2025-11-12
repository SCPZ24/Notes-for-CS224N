# Subword
A trained embedding can't cover every word in human language(humans are keeping generating new words, and there misspelling of human).
If we keep signing this words with `<Unk>`(unknown) token, then great amount of information will be missed.
## Byte-pair Encode Algorithm
Then, we can generate some subword tokens according to the appearing frequency in the corpus of specific character combinations.
In detail, we do
1. Separate the whole text into smallest units(usually one character). Each adding a special sign(e.g. `</w>`) to stand for the end of the word.
2. Count the appearing frequency of all the adjacent characters. Use most common adjacent characters to form a subword.(e.g. `s,u,b` are usually appearing together so we add `sub` as a new subword)
3. Replace instances of the character pair with the new subword; repeat until desired vocab size.
Below is the process of forming subword `est` as an example
![[截屏2025-10-31 21.30.46.png]]
# Inspiration of Pretraining
## The Concept of Pretraining
Before, we train our word embeddings first.
Then, the vector of each token is fixed through the whole training process.
![[截屏2025-10-31 21.33.02.png]]
Instead, we can train the whole model parameters.
To elaborate, we can train the word embeddings and neural layers jointly.
This process is called **Pretraining**.
![[截屏2025-10-31 21.36.14.png]]
The model can't be used after pretraining.
It's just a method of initializing the parameters of the model to lay the foundation for the latter training(may be a profile of human language or knowledge of some basic human syntax).
## The Pretraining-finetuning Process
After the initializing process called pretraining, we can do **Finetuning** to the model, which makes the model be able to handle some task of more specific areas and functions.
Starting from the outcome $\hat{\theta}$ of pretraining, we finetune process can go better
- Maybe the finetuning local minima near $\hat{\theta}$ tend to generalize well!
- Maybe the gradients of finetuning loss near $\hat{\theta}$ propagate nicely(less explosion or vanishment)!
![[截屏2025-10-31 21.46.48.png]]
Pretraining costs a lot. But the outcome of the pretraining can be used(finetuned) many times.
Finetuning costs far less. It's common to run finetuning on a single GPU.
# Pretraining for Encoder
The core method of pretraining encoders is randomly mask some tokens in the source sentence and let the model to predict what masked token is.
## Bidirectional Encoder Representations from Transformers(BERT)
The process is called "Masked LM".
We randomly choose 15% tokens in the source sentence. To the chosen tokens
- Replace 80% of them with a special `<Mask>` token.
- Replace 10% of them with another randomly chosen token.
- Else 10% we don't change them as input,but the model still needs to predict them as its output.
![[截屏2025-10-31 22.13.06.png]]
This teaches the model to make use of context tokens, and find some rules of human language.
## SpanBERT
The hidden tokens are randomly chosen for the BERT above.
In SpanBERT, we tend to choose adjacent tokens to do some masking.
This teaches the model to make use farther and longer contexts, and understand the text structure.
# Pretraining for Encoder-decoder
The classical process is divide the source sentence into two parts.
- Prefix: need to feed into the encoder part with no mask operations.
- Target: the task for decoder to predict, and based on the prediction to do back-propagation.
![[截屏2025-10-31 23.46.08.png]]
Another method used the mask.
![[截屏2025-11-01 00.21.04.png]]
As the picture shows, we mask some tokens and predict the other tokens.
# Pretraining for Decoder
*The Generative Pretrained Transformer(GPT) is a pure Decoder model.*
## Target Output as a Sequence
The precess must be done recurrently.
![[截屏2025-11-01 00.48.21.png]]
Forward propagation with known tokens $w_1,w_2,......,w_t$, and generate a hidden layer $h_t$.
Do linear and maybe softmax to $h_t$, work out the loss and back propagation.
![[截屏2025-11-01 00.49.11.png]]
Pass the right $w_{t+1}$ to continue training.
## Other Downstream Tasks
Just recurrently go through all the given tokens, generating a sequence of hidden layers.
![[截屏2025-11-01 00.50.39.png]]
Use the last hidden layer only.
Do linear and softmax to it, and use it to predict some target labels such as the emotion(positive/ negative) of the sentence.
The $A$ and $b$ here to do linear are randomly and normally generated.
![[截屏2025-11-01 00.53.19.png]]
Basing on this, back propagate through the whole network.