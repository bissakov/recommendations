import os
from os.path import exists, join, dirname

import gensim.downloader as api
import joblib
import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils import vectorize, create_interaction_matrix, apply_svd, combine, evaluate

project_folder = dirname(dirname(__file__))
joblib_folder = join(project_folder, 'joblib')
os.makedirs(joblib_folder, exist_ok=True)


def run(system_joblib_folder: str) -> None:
    os.makedirs(system_joblib_folder, exist_ok=True)

    connection_string = 'mongodb://192.168.0.19:27017'
    client = MongoClient(connection_string)

    if not exists(join(joblib_folder, 'books_df.joblib')):
        db = client['library']

        if not exists(join(joblib_folder, 'model.joblib')):
            model = api.load('fasttext-wiki-news-subwords-300')
            joblib.dump(model, join(joblib_folder, 'model.joblib'))
        else:
            model = joblib.load(join(joblib_folder, 'model.joblib'))
        print('Model loaded')

        books_collection: Collection = db['books']

        books = books_collection.find({'language': 'English'},
                                      {'_id': 0, 'book_id': 1, 'title': 1, 'author_id': 1,
                                       'average_rating': 1, 'ratings_count': 1,
                                       'genres': 1, 'places': 1, 'characters': 1, 'description': 1})
        books_df = pd.DataFrame(list(books))

        books_df['normalized_rating'] = books_df['average_rating'] / 5
        books_df['normalized_count'] = ((books_df['ratings_count'] - books_df['ratings_count'].min()) /
                                        (books_df['ratings_count'].max() - books_df['ratings_count'].min()))
        books_df['popularity_score'] = (books_df['normalized_rating'] + books_df['normalized_count']) / 2

        books_df['genres'] = books_df['genres'].fillna('').str.join(' ')
        books_df['places'] = books_df['places'].fillna('').str.join(' ')
        books_df['characters'] = books_df['characters'].fillna('').str.join(' ')
        books_df['description'] = books_df['description'].fillna('')

        books_df['features'] = (books_df['genres'] + ' ' + books_df['places'] +
                                ' ' + books_df['description'] + ' ' + books_df['characters'])
        books_df['features'] = books_df['features'].fillna('').str.join(' ')

        with tqdm(total=len(books_df['features']), desc='Vectorizing features') as pbar:
            books_df['features_vector'] = books_df['features'] \
                .apply(lambda text: vectorize(text=text, model=model, pbar=pbar))

        # with tqdm(total=len(books_df['genres']), desc='Vectorizing genres') as pbar:
        #     books_df['genres_vector'] = books_df['genres'] \
        #         .apply(lambda x: vectorize_keyword(x, model, pbar))
        #
        # with tqdm(total=len(books_df[books_df['places'] != '']), desc='Vectorizing places') as pbar:
        #     books_df['places_vector'] = books_df['places'] \
        #         .apply(lambda x: vectorize_keyword(x, model, pbar))

        books_df.drop(['genres', 'places', 'average_rating', 'ratings_count',
                       'normalized_rating', 'normalized_count'], axis=1, inplace=True)

        joblib.dump(books_df, join(joblib_folder, 'books_df.joblib'))
        if model:
            del model
    else:
        books_df = joblib.load(join(joblib_folder, 'books_df.joblib'))
    print('Books loaded')

    if not exists(join(joblib_folder, 'ratings_df.joblib')):
        db = client['app_library']
        ratings_collection: Collection = db['ratings']
        ratings_df = pd.DataFrame(list(ratings_collection.find({}, {'book_id': 1, 'user_id': 1, 'rating': 1})))
        ratings_df['book_id'] = ratings_df['book_id'].astype(int)
        ratings_df['user_id'] = ratings_df['user_id'].astype(int)
        ratings_df = ratings_df[ratings_df['book_id'].isin(books_df['book_id'])]
        assert len(ratings_df) > 0
        ratings_df = ratings_df.groupby('user_id').filter(lambda x: len(x) >= 10)
        ratings_df = ratings_df[ratings_df['rating'] > 0]
        ratings_df.drop('_id', axis=1, inplace=True)
        joblib.dump(ratings_df, join(joblib_folder, 'ratings_df.joblib'))
    else:
        ratings_df = joblib.load(join(joblib_folder, 'ratings_df.joblib'))
    print('Ratings loaded')

    books_df = books_df[books_df['book_id'].isin(ratings_df['book_id'].unique())]

    book_id_map = {book_id: idx for idx, book_id in enumerate(books_df['book_id'].unique())}
    user_id_map = {user_id: idx for idx, user_id in enumerate(ratings_df['user_id'].unique())}
    print('Maps created')

    _, test_ratings = train_test_split(ratings_df, test_size=0.01, random_state=32)
    print('Train and test ratings created')

    interaction_matrix = create_interaction_matrix(ratings_df,
                                                   user_id_map,
                                                   book_id_map)
    print('Train interaction matrix created')

    if not exists(join(system_joblib_folder, 'user_factors.joblib')):
        user_factors = apply_svd(interaction_matrix)
        joblib.dump(user_factors, join(system_joblib_folder, 'user_factors.joblib'))
    else:
        user_factors = joblib.load(join(system_joblib_folder, 'user_factors.joblib'))
    print('User factors created')

    features = user_factors

    if 'hybrid' in system_joblib_folder:
        if not exists(join(system_joblib_folder, 'combined_features.joblib')):
            combined_features = combine(books_df, features, user_id_map, interaction_matrix)
            joblib.dump(combined_features, join(system_joblib_folder, 'combined_features.joblib'))
        else:
            combined_features = joblib.load(join(system_joblib_folder, 'combined_features.joblib'))
        print('Combined features created')

        features = combined_features

    evaluate(ratings_df=test_ratings, features=features,
             interaction_matrix=interaction_matrix,
             user_id_map=user_id_map, book_id_map=book_id_map,
             joblib_folder=system_joblib_folder)
    print('Evaluation completed')


def main() -> None:
    run(join(joblib_folder, 'hybrid'))
    # run(join(joblib_folder, 'collaborative'))


if __name__ == '__main__':
    main()
