# Prompting for Pretrained Models
*This involves no gradient steps!*
Pretrained models have learned some basic tools and knowledges to handle human language tasks.
Take GPT as example. GPT is a decoder that can generated a sequence of tokens based on input tokens(may be human's questions).
When it's not finetuned for specific tasks, we can still use some prompts(ask it explicitly in the input sentence) to let it handle specific tasks with already learned general skills.
The state that the model is trained on enough amount of data and is big enough, then it can handle some specific task without finetuning is called **Emergent**.
## Zero-shot Learning
Give the model a prompt, and just ask following questions.
```
Your following task is to translate the sentences to Chinese.

I love china.
I want to eat fish.
```
Then, the model can generate response that translates user's request.
## Few-shot Learning
Give the model a prompt, and some example inputs and outputs. Then the model can do the task more precisely.
A shot is an example for the model.
![[截屏2025-11-01 22.17.45.png]]
The picture shows the user gives the model a task and give some examples. And let the model to do new task of the same kind.
### Chain-of-Thought Prompting
For some tasks that needs a chain of thought(can't work out directly), we can give the examples in the form of Question-Thinking Chain-Answer.
![[截屏2025-11-01 22.21.10.png]]
In this pattern, the model tends to give more precise answer with its thinking chain.
Another way to implement Chain-of-Thought is force the model to output a sentence like "Let's think step by step."
Then the model will output step-by-step thinking.
# Scaling Up Finetuning
We can feed the data each in the form of **Initial Input-Task Label-Output**.
Then we can let the model do multi-tasks.
This finetuning doesn't need the model to do Chatting with user, but do various kinds of tasks with clear label in the input.
# Instruction Finetuning
## Implement
![[截屏2025-11-01 23.18.39.png]]
Finetuning the model with the dataset in the from of **Question-Answer**.
![[截屏2025-11-01 23.18.56.png]]
## Limitation
language modeling penalizes all token-level mistakes equally. But tasks like open-ended creative generation have no right answer, and even with instruction finetuning, there are mismatches between the LM objective and the objective of “satisfy human preferences”.
# Direct Preference Optimization(DPO)
After doing the instruction finetuning, we can do some extra optimization for the model, to let the model more possible to generate the answer that satisfy humans.
## Dataset Demand
We need the data in the form of
- Input sequence $x$
- Some possible output sequence $y$s to the given $x$
- The human labeled satisfied rank of the $y$s.
The model itself can calculate the log-likelihood $p_\theta(y|x)$ for an input $x$ and a specific $y$ as follows
$$
\log p_\theta(y|x)=\log\prod_{t=1}^np_\theta(y_t|x,y_1,y_2,...,y_{t-1})=\sum_{t=1}^n\log p_\theta(y_t|x,y_1,y_2,...,y_{t-1})
$$
*Note that $y$ is a token sequence here. $y_i$ is the $i^{th}$ token.*
## The Loss to Optimize
We can have a reward model to evaluate the satisfying score of a pair of $x,y$ which can be represented as $RM(x,y)$. Use this model to help the original model to optimize.
But in the pattern of DPO, during the calculation, we can cancel out all the $RM(x,y)$.
For a $x$ and its $y$s, we can take out all pairs $y^w$ and $y^l$ that $y^w$ is more satisfying than $y^l$ for humans
Then we can have the DPO Loss
$$
J_{DPO}(\theta)=-E_{(x,y^w,y^l)\sim D}[\log\sigma(RM_\theta(x,y^w)-RM_\theta(x,y^l))]
$$
in which, $E_{(x,y^w,y^l)\sim D}$ means sampling and out all pairs of demanding $y^w,y^l$ in dataset add up the calculating result.
$RM_\theta(x,y)$ is the division of original model's(right after finetuning) predicted possibility to current-step model's(the model updates its parameters every gradient step) predicted possibility. *So we need to remain a copy of the original model.*
$$
RM_\theta=\beta\log\frac{p_\theta^{RL}(y|x)}{p^{PT}(y|x)}
$$
$\beta$ is the **Temperature Coefficient**, a hyper-parameter scalar.
$p_\theta^{RL}$ is the current model prediction(after some turns of parameter update).
$p^{PT}$ is the original model prediction(right after the finetuning process).
Using this form of division can prevent the model goes too far away from the finetuned version, generating unexpected answers.