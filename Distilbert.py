import pandas as pd
import numpy as np
import torch
import logging  # Import the logging module
from sklearn.model_selection import train_test_split
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

# Initialize the logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename='/home/wipro/NLP_RnD/PB_EC/EDA/DISTILLBERT/OUTPUT_20_116_CATS/training.log', level=logging.INFO)  # Specify the log file

output_dir = "/home/wipro/NLP_RnD/PB_EC/dataset/phase2_pb_116_subcats_26092023.csv"
df = pd.read_csv(output_dir)
df = df.dropna()

df['encoded_label'] = df['label'].astype('category').cat.codes
data_texts = df['text'].tolist()
data_labels = df['encoded_label'].tolist()
NUM_LABELS = len(df['encoded_label'].drop_duplicates())

train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size=0.2, random_state=42)

model_checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=NUM_LABELS)

# train_encodings = tokenizer(train_texts, truncation=True, max_length=512, return_tensors='pt', padding='max_length', return_attention_mask=True, return_token_type_ids=False)

# val_encodings = tokenizer(val_texts, truncation=True, max_length=512, return_tensors='pt', padding='max_length', return_attention_mask=True, return_token_type_ids=False)

# # Tokenize the data padding=True, padding=True,
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

# Define a custom PyTorch dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings.input_ids[idx],
            'attention_mask': self.encodings.attention_mask[idx],
            'labels': torch.tensor(self.labels[idx])
        }
        return item

metric = load_metric("accuracy")
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    
    # Calculate F1 score for multi-class classification (macro-average)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1['f1']
    }

train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

training_args = TrainingArguments(
    output_dir ='./results',
    num_train_epochs=20,  
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    load_best_model_at_end=True,
    warmup_steps=500,
    weight_decay=1e-5,
    logging_dir='./logs',
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=500,  # Add this line to specify the logging frequency
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Log the training output
for line in trainer.state.log_history:
    logger.info(line)

results = trainer.evaluate()

model.save_pretrained('/home/wipro/NLP_RnD/PB_EC/EDA/DISTILLBERT/OUTPUT_20_116_CATS')
tokenizer.save_pretrained('/home/wipro/NLP_RnD/PB_EC/EDA/DISTILLBERT/OUTPUT_20_116_CATS')