# Dependency Structure
The structure of a sentence can usually be shown in a tree.
Sentence structure consists of relations between words, which are normally binary asymmetric relations("arrows") called dependencies.
For example, the sentence “`[ROOT]` Bills on ports and immigration were submitted by Republican Senator Brownback of Kansas.” can be represented by a tree
![[截屏2025-10-20 09.47.44.png]]
The root is a assist node to help parse the sentence. There is a tree in an array to describe the whole structure.
![[截屏2025-10-20 09.54.42.png]]
In the sentence above, "completed" is the central word(head); "was", "Discussion" is dependent of "completed". "Discussion" then has some own sub-structures.
## Projectivity
Use some arcs(arrows) to demonstrate the tree structure, and find if there are any crosses among the arcs.
If there exists an cross, the sentence is of no projectivity.
Normally a sentence is of no projectivity.
But when it comes to sentences that have **displaced constituents**, the sentence becomes non-projective, which is more challenging to analyse.
# Arc-standard Transition-based Parser
We need to use a parser to parse a linear sentence into tree structure. A parser maintains a stack $\sigma$, a buffer $\beta$, a set of dependency arcs $A$, and some actions.
## Actions of a Basic Transition-based Parser
In the following demonstration, $w_i,w_j$ are some tokens(words) as a part of the whole sentence.
$\sigma$ is the stack that added to the beginning of the sentence. The token `[ROOT]` is in the bottom of $\sigma$.
$A$ is the set of dependency arcs.
$r(w_x,w_y)$ is the dependency of $w_y$ to $w_x$(an arrow from $w_x$ to $w_y$ in the graph)
$\rightarrow$ means from former value change to latter value.
$|$ is the link. $w_x|w_y$ means $w_x$ and $w_y$ are both in the stack or both in the buffer.
$B$ is the buffer, which contains all the tokens in the sentence that haven't been added to the stack.
1. Shift: push the front word of the buffer to stack.
$$
\sigma,w_i|B\rightarrow\sigma|W_i,B
$$
2. Left-Arc: create a dependency between the two tokens on the top of $\sigma$.
$$
\sigma|w_i|w_j\rightarrow\sigma|w_j
$$
$$
A\rightarrow A\cup r(w_j,w_i)
$$
3. Right-Arc: create a dependency between the two tokens on the top of $\sigma$.
$$
\sigma|w_i|w_j\rightarrow\sigma|w_i
$$
$$
A\rightarrow A\cup r(w_i,w_j)
$$
## An Example With a Sequence of Actions
![[截屏2025-10-20 21.15.00.png]]![[截屏2025-10-20 21.15.38.png]]
The process ends when the buffer is empty and the stack has only the root.
## Neural Dependency Parser
We can use a neural network to decide which action to take on a certain state.
Input the word vectors of the top 2(or more) tokens in the stack and 1(or more) front tokens in the buffer to the model as input features and some other features such as POS(Part of Speech,词性).
![[截屏2025-10-20 22.55.11.png]]
There are 7 words involve.
	$s_1,s_2,b_1$: top 2 tokens in the stack and the front word in the buffer.
	$lc(s_1)$: the left child of $s_1$ in the already build dependency in $A$. If there are multi left children, pick the nearest one or pick the dependency that are more close.
	$rc(s_1),lc(s_2),rc(s_2)$: same with $lc(s_1)$.
There are 3 attributes of a token involve.
	The token itself(word vector).
	The labeled POS.
	The already build dependency. For example, the subject is signed as "nsubj".
The output of the model is a 3-dimensional vector after a Softmax, indicating which action to take.
There is no searching in the process. So roughly the cost is $O(n)$. Some extra search operations can be used to rise the accuracy but it will cost time. But in general, the accuracy of this model is enough.
![[截屏2025-10-20 22.45.19.png]]
## Advantages of Neural Dependency Parser
1. The non-linear feature of neural network which allows to learn more complex nonlinear decision boundaries.
2. Use dense word vector rather than one-hot encoding for every circumstance.
# Evaluation of Dependency Parser
After building a dependency of a sentence, we can use some values to measure the accuracy of the parser.
![[截屏2025-10-20 23.24.46.png]]
For example, Gold is the right answer assigned by human. Parsed is the outcome if machine.
## Unlabeled attachment score(UAS)
Measure how much **arcs** the model predicts right.
In the picture, tokens `1,2,4,5` are right. So the UAS is 80%.
## Unlabeled attachment score(LAS)
Measure how much tokens that both the **arcs** and the **dependencies** are predicted right.
In the picture, tokens `1,2` are right. So the LAS is 40%.