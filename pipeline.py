import os
import re
import string

import nltk
import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

lemma = WordNetLemmatizer()
stemmer = PorterStemmer()


def preprocess_text(text):
    words = word_tokenize(text)
    words = [lemma.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)


def clean_text(text):
    text = str(text).lower()  # Lowering the case
    text = re.sub('\[.*?\]', '', text)  # Remove any text in the square brackets
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove any links present
    text = re.sub('<.*?>+', '', text)  #
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub('\n', '', text)  # Removing the next line character
    text = re.sub('\w*\d\w*', '', text)  # Removing the words containing numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text


def load(input_dir: str, input_csv: str = "data.csv"):
    filepath_or_buffer = os.path.join(input_dir, input_csv)
    df = pd.read_csv(filepath_or_buffer)
    df['summary'] = df['summary'].apply(clean_text)
    df['summary'] = df['summary'].apply(preprocess_text)
    df.info()
    return df


def encode(df):
    one_hot = pd.get_dummies(df['genre'])
    df = df.drop('genre', axis=1)
    df = df.join(one_hot)
    df.info()
    return df


def split(df):
    columns_base = ['index', 'title', 'summary']
    columns_genres = df.columns.difference(columns_base)
    df_base = df[columns_base]
    df_genres = []
    for column_genre in columns_genres:
        df_genre = df_base.copy().join(df[column_genre])
        df_genres.append(df_genre)
    return df_genres


def organize(output_dir: str, df_genres):
    columns_base = ['index', 'title', 'summary']
    for df_genre in df_genres:
        [column_genre] = df_genre.columns.difference(columns_base)
        filepath_genre = os.path.join(output_dir, column_genre)
        if not os.path.exists(filepath_genre):
            os.makedirs(filepath_genre)
            for _, row in df_genre.iterrows():
                index = row['index']
                summary = row['summary']
                filename_genre = f"{index}.txt"
                with open(os.path.join(filepath_genre, filename_genre), 'w') as summary_file:
                    summary_file.write(summary)
    pass


def pipe(input_dir: str, input_csv: str, output_dir: str):
    df = load(input_dir, input_csv)
    df = encode(df)
    organize(output_dir, df_genres=split(df))
    pass
