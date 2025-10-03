import pandas as pd 

df = pd.read_csv('bert_predictions.csv')

misclassified_df = df[df['true_labels'] != df['pred_label']]

print(misclassified_df.head(10))