import torch
import torch.nn as nn
import numpy as np 
import pandas as pd 
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utilities import print_metrics, plot_confusion_matrix

# load dataset
dataset = load_dataset("tweet_eval", "irony")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(
        batch['text'], 
        padding='max_length', 
        truncation=True, 
        max_length=64
    )

encoded_dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])
encoded_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# compute class weights from training labels
train_labels = np.array(dataset['train']['label'])
classes = np.unique(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=classes,y=train_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# define model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# custom Trainer with weighted loss
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs = False, **kwargs):
        labels = inputs.get('labels')
        outputs = model(
            **{k: v for k, v in inputs.items() if k != 'labels'}, 
            return_dict=True
        )
        logits = outputs.logits
        loss_func = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_func(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
# metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy='epoch',
    save_strategy="no",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    logging_dir="./logs",
    report_to=[],
    seed=42
)

# trainer
trainer = WeightedTrainer(
    class_weights=class_weights_tensor,
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
    compute_metrics=compute_metrics
)

# train
trainer.train()

# evaluate
results = trainer.evaluate()
print("Evaluation Results:", results)

# predict on validation set
predictions_output = trainer.predict(encoded_dataset['validation'])

# store validation texts for error analysis
test_texts = dataset['validation']['text']

preds = np.argmax(predictions_output.predictions, axis=1)
true_labels = predictions_output.label_ids

print_metrics(true_labels, preds, target_names=['Not Sarcastic', 'Sarcastic'])
plot_confusion_matrix(true_labels, preds, labels=['Not Sarcastic', 'Sarcastic']) 

if __name__ == '__main__':
    
    # train and evaluate
    trainer.train()
    results = trainer.evaluate()
    print("Evaluation Results: ", results)
    
    # predict on validation set
    predictions_output = trainer.predict(encoded_dataset['validation'])
    preds = np.argmax(predictions_output.predictions, axis=1)
    true_labels = predictions_output.label_ids
    
    print_metrics(true_labels, preds, target_names=['Not Sarcastic', 'Sarcastic'])
    plot_confusion_matrix(true_labels, preds, labels=['Not Sarcastic', 'Sarcastic'])
    
    # save predictions to csv to import into error_analysis easier
    pd.DataFrame({
        'text': dataset['validation']['text'],
        'true_labels': true_labels,
        'pred_label': preds
    }).to_csv('bert_predictions.csv', index=False)
    print("Predictions saved to bert_predictions.csv")