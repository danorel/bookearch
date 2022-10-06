import os
import pandas as pd


def load(input_dir: str, input_csv: str = "data.csv"):
    filepath_or_buffer = os.path.join(input_dir, input_csv)
    df = pd.read_csv(filepath_or_buffer)
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
                title = row['title']
                summary = row['summary']
                filename_genre = f"{title}.txt"
                with open(os.path.join(filepath_genre, filename_genre), 'w') as summary_file:
                    summary_file.write(summary)
    pass


def preprocess(input_dir: str, input_csv: str, output_dir: str):
    df = load(input_dir, input_csv)
    df = encode(df)
    organize(output_dir, df_genres=split(df))
    pass
