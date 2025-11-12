Basic python tools are in transformers library.
```
pip install transformers
```
# Tokenizer
A Large Language Model has its own tokenizer.
Tokenizer is the algorithm to divide sentences into a sequence of tokens.
## Load Tokenizer
```Python
from transformers import DistilBertTokenizer, DistilBertTokenizerFast, AutoTokenizer
```
*DistilBert is the distilled version of BERT Model.(distill makes the model smaller but has similar function)*
*DistilBertTokenizer is its Tokenizer.*
*DistilBertTokenizerFast the version rewrite in Rust which goes faster than the Python version.*
Use these tokenizer models to contain tokenizer from internet.
```Python
name = "distilbert/distilbert-base-cased"

tokenizer = DistilBertTokenizer.from_pretrained(name) # written in Python
print(tokenizer)

tokenizer = DistilBertTokenizerFast.from_pretrained(name) # written in Rust
print(tokenizer)

tokenizer = AutoTokenizer.from_pretrained(name) # convenient! Defaults to Fast
print(tokenizer)
```
```
DistilBertTokenizer(name_or_path='distilbert/distilbert-base-cased', vocab_size=28996, model_max_length=1000000000000000019884624838656, ...... } 

DistilBertTokenizerFast(name_or_path='distilbert/distilbert-base-cased', vocab_size=28996, model_max_length=1000000000000000019884624838656, ...... } 

DistilBertTokenizerFast(name_or_path='distilbert/distilbert-base-cased', vocab_size=28996, model_max_length=1000000000000000019884624838656, ...... }
```
## Call Tokenizer
Just feed String sentence into it.
The output is a dict include `input_ids` and `attention_mask`.
```Python
input_str = "Hugging Face Transformers is great!"

tokenized_inputs = tokenizer(input_str)

# Two ways to access:
print(tokenized_inputs.input_ids)
print(tokenized_inputs["input_ids"])

print(tokenized_inputs.attention_mask)
print(tokenized_inputs.["attention_mask"])
```
```
[101, 20164, 10932, 10289, 25267, 1110, 1632, 106, 102]
[101, 20164, 10932, 10289, 25267, 1110, 1632, 106, 102]

[1, 1, 1, 1, 1, 1, 1, 1, 1]
[1, 1, 1, 1, 1, 1, 1, 1, 1]
```
The inner process is like as follows
```
start:Hugging Face Transformers is great!
tokenize:['Hu', '##gging', 'Face', 'Transformers', 'is', 'great', '!']
convert_tokens_to_ids:[20164, 10932, 10289, 25267, 1110, 1632, 106]
add special tokens:[101, 20164, 10932, 10289, 25267, 1110, 1632, 106, 102]
--------
decode:[CLS] Hugging Face Transformers is great! [SEP]
```
We can let the tokenizer output pytorch tensors.
```Python
model_inputs = tokenizer("Hugging Face Transformers is great!", return_tensors="pt")
```
We can feed in batches of sentences, and ask it to pad the tokens(make all the tokenized sequence same-length, by padding special `<Pad>` tokens to end of not-longest sequences, or just pad every sequence to the model's max length)
```Python
model_inputs = tokenizer(
	["Hugging Face Transformers is great!",
	"The quick brown fox jumps over the lazy dog." +\
	"Then the dog got up and ran away because she didn't like foxes.",
	],
	return_tensors="pt",
	padding=True,
	truncation=True)
```
truncation means that when the sequence is longer than limited length of the model(transformers model has fixed max input length, according to [[Transformers]]), the tokenizer will cut and discard the rest tokens in the end.
# Model
## Load Model
Models in transformers can be loaded in similar ways.
```Python
from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification, DistilBertModel

base_model = DistilBertModel.from_pretrained('distilbert-base-cased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)
```
We can also load model with random weights.
Here we load a model with a random classifier with output dimension $2$.
Note that the parameters of the classifier haven't been trained right after it's loaded.
```Python
from transformers import DistilBertConfig, DistilBertModel

# Initializing a DistilBERT configuration
configuration = DistilBertConfig()
configuration.num_labels=2
# Initializing a model (with random weights) from the configuration
model = DistilBertForSequenceClassification(configuration)

# Accessing the model configuration
configuration = model.config
```
## Feed Data
We can feed datas in form of key-value.
```Python
model_inputs = tokenizer(input_str, return_tensors="pt")

# Option 1
model_outputs = model(
	input_ids=model_inputs.input_ids,
	attention_mask=model_inputs.attention_mask
)

# Option 2 - the keys of the dictionary the tokenizer returns are the same as the keyword arguments
model_outputs = model(**model_inputs)

print(model_outputs)
```
```
SequenceClassifierOutput(loss=None, logits=tensor([[0.0368, 0.0659]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
```
For training, we can concatenate the loaded model with torch's APIs.
```Python
# We can calculate the loss like normal
label = torch.tensor([1])
loss = torch.nn.functional.cross_entropy(model_outputs.logits, label)

print(loss)

loss.backward()

# You can get the parameters
print(list(model.named_parameters())[0])
```
```
tensor(0.6787, grad_fn=<NllLossBackward0>)

('distilbert.embeddings.word_embeddings.weight',
 Parameter containing:
 tensor([[-2.5130e-02, -3.3044e-02, -2.4396e-03,  ..., -1.0848e-02,
          -4.6824e-02, -9.4855e-03],
         [-4.8244e-03, -2.1486e-02, -8.7145e-03,  ..., -2.6029e-02,
          -3.7862e-02, -2.4103e-02],
         ...,
         [ 1.1905e-02, -2.3293e-02, -2.2506e-02,  ..., -2.7136e-02,
          -4.3556e-02,  1.0529e-04]], requires_grad=True))
```
To extract the output attention and hidden state in the model calculation, we can set some parameters when load the model.
```Python
from transformers import AutoModel

model = AutoModel.from_pretrained("distilbert-base-cased", output_attentions=True, output_hidden_states=True)
model.eval()

model_inputs = tokenizer(input_str, return_tensors="pt")
with torch.no_grad():
    model_output = model(**model_inputs)


print("Hidden state size (per layer):  ", model_output.hidden_states[0].shape)
print("Attention head size (per layer):", model_output.attentions[0].shape)
```
# Finetuning
## Load the Dataset
```Python
from datasets import load_dataset, DatasetDict

# DataLoader(zip(list1, list2))
dataset_name = "stanfordnlp/imdb"
imdb_dataset = load_dataset(dataset_name)

# Just take the first 50 tokens for speed/running on cpu
def truncate(example):
    return {
        'text': " ".join(example['text'].split()[:50]),
        'label': example['label']
    }

print(imdb_dataset)
```
```
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
})
```
Here, we randomly load some of the data
```Python
# Take 128 random examples for train and 32 validation
small_imdb_dataset = DatasetDict(
	train=imdb_dataset['train'].shuffle(seed=1111).select(range(128)).map(truncate),
    val=imdb_dataset['train'].shuffle(seed=1111).select(range(128, 160)).map(truncate),
)
print(small_imdb_dataset)
```
```
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 128
    })
    val: Dataset({
        features: ['text', 'label'],
        num_rows: 32
    })
})
```
Then, we tokenize the dataset and turn it into batched tensors.
```Python
# Prepare the dataset - this tokenizes the dataset in batches of 16 examples.
small_tokenized_dataset = small_imdb_dataset.map(
    lambda example: tokenizer(example['text'], padding=True, truncation=True),
    batched=True,
    batch_size=16
)

small_tokenized_dataset = small_tokenized_dataset.remove_columns(["text"])
small_tokenized_dataset = small_tokenized_dataset.rename_column("label", "labels")
small_tokenized_dataset.set_format("torch")
```
Feed the data into `dataloader`, ready to do tensor jobs in `pytorch`.
```Python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_tokenized_dataset['train'], batch_size=16)
eval_dataloader = DataLoader(small_tokenized_dataset['val'], batch_size=16)
```
## Train
Hugging Face models are also `torch.nn.Module`s so backpropagation happens the same way.
For optimization, we're using the AdamW Optimizer, which is almost identical to Adam except it also includes weight decay. And we're using a linear learning rate scheduler, which reduces the learning rate a little bit after each training step over the course of training.
```Python
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)

num_epochs = 1
num_training_steps = len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

best_val_loss = float("inf")
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    # training
    model.train()
    for batch_i, batch in enumerate(train_dataloader):

        # batch = ([text1, text2], [0, 1])

        output = model(**batch)

        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

    # validation
    model.eval()
    for batch_i, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            output = model(**batch)
        loss += output.loss

    avg_val_loss = loss / len(eval_dataloader)
    print(f"Validation loss: {avg_val_loss}")
    if avg_val_loss < best_val_loss:
        print("Saving checkpoint!")
        best_val_loss = avg_val_loss
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'val_loss': best_val_loss,
        #     },
        #     f"checkpoints/epoch_{epoch}.pt"
        # )
```
`transformers` also provides us with a `Trainer` class and a `TrainingArguments` class.
We first create the arguments object, then define the trainer.
```Python
from transformers import TrainingArguments, Trainer

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)

arguments = TrainingArguments(
    output_dir="sample_hf_trainer",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    evaluation_strategy="epoch", # run validation at the end of each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    seed=224
)


def compute_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculates the accuracy
    return {"accuracy": np.mean(predictions == labels)}


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=small_tokenized_dataset['train'],
    eval_dataset=small_tokenized_dataset['val'], # change to test when you do your final evaluation!
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```
## Evaluate
Using the trainer class, evaluating the model is very easy.
```Python
# results = trainer.evaluate() # just gets evaluation metrics
results = trainer.predict(small_tokenized_dataset['val']) # also gives you predictions
```
## Load
To load the model, we can just type in the directory and check point.
```Python
# To load our saved model, we can pass the path to the checkpoint into the `from_pretrained` method:
test_str = "I enjoyed the movie!"

finetuned_model = AutoModelForSequenceClassification.from_pretrained("sample_hf_trainer/checkpoint-8")
model_inputs = tokenizer(test_str, return_tensors="pt")
prediction = torch.argmax(finetuned_model(**model_inputs).logits)
print(["NEGATIVE", "POSITIVE"][prediction])
```
## Suggested Hyper Parameters for Finetuning
- Epochs: {2, 3, 4} (larger amounts of data need fewer epochs)
- Batch size (bigger is better: as large as you can make it)
- Optimizer: AdamW
- AdamW learning rate: {2e-5, 5e-5}
- Learning rate scheduler: linear warm up for first {0, 100, 500} steps of training
- weight_decay (l2 regularization): {0, 0.01, 0.1}