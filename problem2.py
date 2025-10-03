from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import random

# load imdb dataset
dataset = load_dataset('imdb')
test_data = dataset['test']

# sample 5000 reviews for faster testing
sample_size = 5000
indices = random.sample(range(len(test_data)), sample_size)
test_texts = [test_data[i]['text'] for i in indices]
true_labels = ['POSITIVE' if test_data[i]['label'] == 1 else 'NEGATIVE' for i in indices]

# load pre-trained DistilBERT model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=512,
    )

# run inference on the sampled texts
print("Running inference...")
predictions = [res['label'] for res in sentiment_pipeline(test_texts, batch_size=32)]

# evaluate the results
accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions)

print(f'Accuracy: {accuracy:.4f}\n')
print('Classification Report:\n')
print(report)