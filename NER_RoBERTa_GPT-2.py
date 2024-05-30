####-----------------------------------------------PYTHON CODE TO TRAIN A NER MODEL USING ROBERTA---------------------------------------####

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer, RobertaForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import pandas as pd

# Load the training data
new_data_path = r"C:\Users\ZA40142720\Downloads\test_data_conll.conll"

sentences = []
labels = []

current_sentence = []
current_labels = []

with open(new_data_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line:  # If the line is not empty
            parts = line.split()  # Split the line into parts
            if len(parts) > 1:  # Ensure there's enough parts
                current_sentence.append(parts[0])  # Assume the first part is the word
                current_labels.append(parts[-1])  # Assume the last part is the label
        else:
            if current_sentence and current_labels:  # Ensure they are not empty
                sentences.append(current_sentence)
                labels.append(current_labels)
                current_sentence = []
                current_labels = []

# Load the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Define the label mapping
label_map = {"O": 0, "B-DATE": 1, "I-DATE": 2, "B-AMOUNT": 3, "I-AMOUNT": 4, "B-CREDIT_CARD_NO": 5, "I-CREDIT_CARD_NO": 6, "B-MOBILE_NO": 7, "I-MOBILE_NO": 8, "B-ACCOUNT_NO": 9, "I-ACCOUNT_NO": 10, "B-DATE_RANGE": 11, "I-DATE_RANGE": 12, "B-MODE_OF_PAYMENT": 13, "I-MODE_OF_PAYMENT": 14}

# Tokenize the sentences and convert labels to integers
tokenized_sentences = [tokenizer.encode(sent, add_special_tokens=True) for sent in sentences]
labels = [[label_map[label] for label in sent] for sent in labels]

# Pad the sequences to the same length
max_len = max(len(seq) for seq in tokenized_sentences)
tokenized_sentences = [seq + [tokenizer.pad_token_id] * (max_len - len(seq)) for seq in tokenized_sentences]
labels = [label + [-100] * (max_len - len(label)) for label in labels]

# Convert the tokenized sentences and labels to PyTorch tensors
input_ids = torch.tensor(tokenized_sentences)
labels = torch.tensor(labels)

# Load the pre-trained RoBERTa model for token classification
model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=len(label_map))

# Define the TrainingArguments
training_args = TrainingArguments(
    output_dir="roberta-token-classification",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=3,
)

# Define the Dataset for the training data
class CustomDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "attention_mask": self.input_ids[index], "labels": self.labels[index]}

    def __len__(self):
        return len(self.input_ids)

train_dataset = CustomDataset(input_ids, labels)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
)

# Train the model
trainer.train()

# Evaluate the model on the training data
eval_result = trainer.evaluate()
print(eval_result)

# Save the trained model
trainer.save_model("roberta-token-classification")

####-----------------------------------PYTHON CODE TO PREDICT A REOBERTA MODEL FOR TOKEN CLASSIFICATION---------------####

df = pd.read_excel(r"C:\Users\ZA40142720\Downloads\entity_smapleTest\entity_smapleTest.xlsx")
test_sentences = df['text'].apply(lambda x: x.split()).tolist()

import torch
from transformers import RobertaTokenizer, RobertaForTokenClassification

model_path = "roberta-token-classification"  
model = RobertaForTokenClassification.from_pretrained(model_path)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def predict_labels(sentences):
    tokenized_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    # Perform inference
    with torch.no_grad():
        outputs = model(**tokenized_sentences)
    # Get the predicted labels
    predicted_labels = torch.argmax(outputs.logits, dim=2).tolist()
    return predicted_labels

# Example sentences for prediction
input_sentences = [
    "Hello, how are you?",
    "My credit card number is 1234.",
    "Please transfer $100 from my account to John's account."
]

# Perform prediction
predicted_labels = predict_labels(input_sentences)

# Print the predicted labels
for sentence, labels in zip(input_sentences, predicted_labels):
    print("Sentence:", sentence)
    print("Predicted Labels:", labels)


####------------------------------------PYTHON CODE TO TRAIN A GPT2 FOR NER ------------------------------------------####

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast, GPT2ForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load the data from conll file
def read_conll(file_path):
    sentences = []
    labels = []
    sentence = []
    label = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            else:
                parts = line.strip().split()
                token = parts[0]
                tag = parts[-1]
                sentence.append(token)
                label.append(tag)

    if sentence:
        sentences.append(sentence)
        labels.append(label)

    return sentences, labels

sentences, labels = read_conll(r"C:\Users\ZA40142720\Downloads\test_data_conll.conll")

# Create a label to ID mapping
unique_labels = set(label for sublist in labels for label in sublist)
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

# Split into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)

# Load the fast tokenizer with add_prefix_space=True
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the data and preserve labels
def tokenize_and_preserve_labels(texts, labels):
    tokenized_inputs = tokenizer(
        texts,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=1024,
        return_tensors="pt"
    )

    labels_out = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels_out.append(label_ids)

    tokenized_inputs["labels"] = torch.tensor(labels_out, dtype=torch.long)
    return tokenized_inputs

train_encodings = tokenize_and_preserve_labels(train_texts, train_labels)
test_encodings = tokenize_and_preserve_labels(test_texts, test_labels)

class TokenClassificationDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

def data_collator(features):
    batch = {}
    for key in features[0].keys():
        batch[key] = torch.nn.utils.rnn.pad_sequence([feature[key] for feature in features], batch_first=True, padding_value=tokenizer.pad_token_id if key != "labels" else -100)
    return batch

train_dataset = TokenClassificationDataset(train_encodings)
test_dataset = TokenClassificationDataset(test_encodings)

# Number of unique labels
num_labels = len(label2id)

# Load the GPT-2 model
model = GPT2ForTokenClassification.from_pretrained('gpt2', num_labels=num_labels)
model.config.pad_token_id = model.config.eos_token_id
model.config.label2id = label2id
model.config.id2label = id2label

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=4,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    data_collator=data_collator
)

trainer.train()

results = trainer.evaluate()
print(results)

model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')

####--------------------------------PYTHON CODE TO PREDICT A GPT-2 MODEL FOR NER------------------------------####

import torch
from transformers import GPT2TokenizerFast, GPT2ForTokenClassification

# Load the trained model and tokenizer
model = GPT2ForTokenClassification.from_pretrained('./trained_model')
tokenizer = GPT2TokenizerFast.from_pretrained('./trained_model')

# Load the label mapping
id2label = model.config.id2label

# Define a new sentence for prediction
new_sentences = "non-receipt of 4% fuel cashback on hp super saver credit card dear icici bank customer service team, thank you for your prompt response and acknowledgment of my concerns regarding the 4% fuel cashback on my hp super saver credit card for the period from august 2023 to december 2023. i appreciate your teams efforts in reviewing my account and understanding the discrepancies in the cashback amounts, as outlined in my initial email. however, i note that the amounts mentioned in your response are consistent with the details i provided earlier. which 123.14/- was missed by me"

# Tokenize the new sentence
inputs = tokenizer(new_sentences, padding=True, truncation=True, return_tensors="pt")

# Get the model's predictions
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted labels
predicted_ids = torch.argmax(logits, dim=-1).squeeze().tolist()
predicted_labels = [id2label[idx] for idx in predicted_ids]

# Get the combined text for each named entity
combined_entities = []
current_entity = []
current_label = None

for i, token in enumerate(tokenizer.convert_ids_to_tokens(inputs["input_ids"].tolist()[0])):
    label = predicted_labels[i]
    if label != "O":
        if current_label is None:
            current_label = label
        if current_label == label:
            current_entity.append(token.replace("Ġ", " "))
        else:
            combined_entities.append(" ".join(current_entity))
            current_entity = [token.replace("Ġ", " ")]
            current_label = label
    else:
        if current_entity:
            combined_entities.append("".join(current_entity))
            current_entity = []
            current_label = None

# Format the results as a dictionary
named_entities = {entity: current_label for entity, current_label in zip(combined_entities, predicted_labels)}

# Print the named entities and their labels
print("Named entities and their labels:", named_entities)
