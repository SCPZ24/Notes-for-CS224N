# Tensor
Exactly the same action with `numpy.array`.
Doing sections returns a smaller tensor even it's only one number. Use `tensor.item()` to extract the single value.
# Auto Grad
Pytorch works out gradient for every tensor automatically.
We can activate the auto grad system for a tensor when defining it.
```Python
x = torch.tensor([2.], requires_grad=True)

y = x * x * 3 #define a function y=f(x) here, and y is also a tensor.

y.backward()
print(x.grad) # returns 12
```
If x is forward propagated to multiple variables, it's gradient will be added up from all gradients passed back.
```Python
x.grad = None

z = x * x * 3 # 3x^2

z.backward() #passes gradient 12 to x
y.backward() #passes gradient 12 to x
# y = x * x * 3

print(x.grad) #returns 24 = 12 + 12.
```
So during backward propagation, the gradient to a node is accumulated up.
Run `zero_grad()` to clear all the grad before every training iteration.
# Neural Network Layers
We need to import the `nn` module.
```Python
import torch.nn as nn
```
## Linear Layer
```Python
# Create the inputs
input = torch.ones(2,3,4)

# Make a linear layers transforming N,*,H_in dimensinal inputs to N,*,H_out dimensional outputs
linear = nn.Linear(4, 2)

linear_output = linear(input)

print(linear_output.shape)
#output:torch.Size([2, 3, 2])
```
In the code above, input size is `2,3,4`.
The first number `2` is the batch size dimension. The after are matrices that contains datas.
## Activation Function Layer
Some examples of activations functions are `nn.ReLU()`, `nn.Sigmoid()` and `nn.LeakyReLU()`.
## Encapsulating the Layers
We can use `nn.Sequential(layers...)` to capture different layers in a container.
```Python
block = nn.Sequential(
	nn.Linear(4, 2),
	nn.Sigmoid()
)

input = torch.ones(2,3,4)

output = block(input)
```
# Neural Network Implement
A neural network should extend the class `nn.Module`.
Implement the `__init__()` function. Call the constructor of super class first. Define some variables to store some information(e.g. size). Then declare the layers in the network.
Override the forward() function to predict an output based on the inputs.
```Python
class MultilayerPerceptron(nn.Module):

  def __init__(self, input_size, hidden_size):
    # Call to the __init__ function of the super class
    super(MultilayerPerceptron, self).__init__()

    # Bookkeeping: Saving the initialization parameters
    self.input_size = input_size
    self.hidden_size = hidden_size

    # Defining of our model
    # There isn't anything specific about the naming of `self.model`. It could
    # be something arbitrary.
    self.model = nn.Sequential(
        nn.Linear(self.input_size, self.hidden_size),
        nn.ReLU(),
        nn.Linear(self.hidden_size, self.input_size),
        nn.Sigmoid()
    )

  def forward(self, x):
    output = self.model(x)
    return output
```
Another way to declare the layers are as follows.
```Python
class MultilayerPerceptron(nn.Module):

  def __init__(self, input_size, hidden_size):
    # Call to the __init__ function of the super class
    super(MultilayerPerceptron, self).__init__()

    # Bookkeeping: Saving the initialization parameters
    self.input_size = input_size
    self.hidden_size = hidden_size

    # Defining of our layers
    self.linear = nn.Linear(self.input_size, self.hidden_size)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(self.hidden_size, self.input_size)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    linear = self.linear(x)
    relu = self.relu(linear)
    linear2 = self.linear2(relu)
    output = self.sigmoid(linear2)
    return output
```
To use our network, create an instance of the class.
```Python
# Make a sample input
input = torch.randn(2, 5)

# Create our model
model = MultilayerPerceptron(5, 3)

# Pass our input through our model
predict = model(input)

print(predict)
```
# Procedure of Optimizing
Once the model class is defined and dataset is ready, we can optimize the parameters in a fixed procedure.
We need to import the module first.
```Python
import torch.optim as optim
```
## Initialize the Variables
Create instances of out model, optimizer, loss function.
```Python
# Instantiate the model
model = MultilayerPerceptron(5, 3)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-1)

# Define loss using a predefined loss function
loss_function = nn.MSELoss()
```
## Train in Loops
```Python
# Set the number of epoch, which determines the number of training iterations
n_epoch = 10

for epoch in range(n_epoch):
  # Set the gradients to 0
  optimizer.zero_grad()

  # Get the model predictions
  y_pred = model(x)

  # Get the loss
  loss = loss_function(y_pred, y)

  # Print stats
  print(f"Epoch {epoch}: traing loss: {loss}")

  # Compute the gradients
  loss.backward()

  # Take a step to optimize the weights
  optimizer.step()
```
Some other code such as calculating and printing the accuracy of training in every epoch can be added to the loop statement.
# DataLoader
DataLoader is a tool that can yield datas in batches.
```Python
from torch.utils.data import DataLoader
```
## Create a Loader
The statement is
```Python
loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
```
in which, `collate_fn` is the function that implements to a batch of data. It should have only one parameter: the batched data.
Different loaders can be integrate to train a model such as `train_x`, `train_y`, `test_x`, `test_y`.
Also, data and labels can be zipped into a same loader.
## Yield from a Loader
When training, load data with loader.
Use a for loop. For each iteration, we have inputs, labels, and batch size.
```Python
def train_epoch(loss_function, optimizer, model, loader):
# Keep track of the total loss for the batch
total_loss = 0

for batch_inputs, batch_labels, batch_lengths in loader:
	# Clear the gradients
	optimizer.zero_grad()
	
	# Run a forward pass
	outputs = model.forward(batch_inputs)
	
	# Compute the batch loss
	loss = loss_function(outputs, batch_labels, batch_lengths)
	
	# Calculate the gradients
	loss.backward()
	
	# Update the parameteres
	optimizer.step()
	
	total_loss += loss.item()

return total_loss
```