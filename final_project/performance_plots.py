import matplotlib.pyplot as plt 
import numpy as np 

# results from svm and bert
metrics_svm = {'F1': 0.6454, 'Accuracy': 0.6416, 'Precision': 0.6609, 'Recall': 0.6416}
metrics_bert = {'F1': 0.6705, 'Accuracy': 0.6712, 'Precision': 0.6774, 'Recall': 0.6712}

labels = list(metrics_svm.keys())
svm_scores = list(metrics_svm.values())
bert_scores = list(metrics_bert.values())

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, svm_scores, width, label='SVM')
ax.bar(x + width/2, bert_scores, width, label='BERT')

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.ylim(0,1)
plt.show()