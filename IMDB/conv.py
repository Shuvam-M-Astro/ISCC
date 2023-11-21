import os
import pandas as pd

def convert_to_csv(data_dir, output_csv):
    reviews = []
    sentiments = []

    for sentiment in ['pos', 'neg']:
        directory = os.path.join(data_dir, sentiment)
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                with open(os.path.join(directory, filename), 'r', encoding='utf8') as file:
                    reviews.append(file.read())
                    sentiments.append(1 if sentiment == 'pos' else 0)

    df = pd.DataFrame({
        'review': reviews,
        'sentiment': sentiments
    })

    df.to_csv(output_csv, index=False)

# Adjust these paths according to your file structure
convert_to_csv('C:/Users/shuva/Downloads/Bodvar/aclImdb_v1/aclImdb/train', 'imdb_reviews_train.csv')
convert_to_csv('C:/Users/shuva/Downloads/Bodvar/aclImdb_v1/aclImdb/test', 'imdb_reviews_test.csv')
