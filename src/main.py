import os
from math import sqrt
from os.path import exists, join
from typing import Dict, List

import gensim.downloader as api
import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import word_tokenize
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

import joblib


def connect_to_mongo(connection_string: str, db_name: str) -> Database:
    client = MongoClient(connection_string)
    db = client[db_name]
    return db


def vectorize_keyword(keyword: str, model: KeyedVectors,
                      pbar: tqdm) -> np.ndarray:
    words = word_tokenize(keyword.lower())
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(model.vector_size)
    keyword_vector = np.mean(word_vectors, axis=0)
    pbar.update(1)
    return keyword_vector


def create_interaction_matrix(df: pd.DataFrame,
                              user_id_map: Dict[int, int],
                              book_id_map: Dict[int, int]) -> lil_matrix:
    matrix = lil_matrix((len(user_id_map), len(book_id_map)))
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        matrix[user_id_map[row.user_id], book_id_map[row.book_id]] = row.rating
    return matrix


def apply_svd(interaction_matrix: lil_matrix, n_components: int = 150) -> np.ndarray:
    svd = TruncatedSVD(n_components=n_components)
    return svd.fit_transform(interaction_matrix)


def combine_features(user_id: int, user_factors: np.ndarray, user_id_map: Dict[int, int],
                     feature_vectors: List[np.ndarray],
                     interaction_matrix: lil_matrix) -> np.ndarray:
    user_idx = user_id_map.get(user_id)
    if user_idx is None:
        return np.zeros(user_factors.shape[1])

    rated_books_indices = interaction_matrix[user_idx].nonzero()[1]

    weights = [0.6, 0.3, 0.1, 0.1]

    avg_vectors = []
    for vec, weight in zip(feature_vectors, weights):
        if rated_books_indices.size > 0 and vec.shape[1] > 0:
            avg_vector = np.mean(vec[rated_books_indices], axis=0) * weight
        else:
            avg_vector = np.zeros(vec.shape[1]) if vec.shape[1] > 0 else np.zeros(user_factors.shape[1])
        avg_vectors.append(avg_vector)

    combined_content_features = np.concatenate(avg_vectors)

    if np.linalg.norm(combined_content_features) > 0:
        norm_combined_content_features = combined_content_features / np.linalg.norm(combined_content_features)
    else:
        norm_combined_content_features = np.zeros_like(combined_content_features)

    user_feature = user_factors[user_idx] * 0.7
    combined_features = np.concatenate([user_feature, norm_combined_content_features * 0.3])
    return combined_features


def combine(books_df: pd.DataFrame, user_factors: np.ndarray, user_id_map: Dict[int, int],
            train_interaction_matrix: lil_matrix) -> np.ndarray:
    def extract_vectors(column_name):
        vectors = np.array(books_df[column_name].tolist())
        return vectors if vectors.ndim == 2 else np.zeros((len(books_df), user_factors.shape[1]))

    feature_vectors = [
        extract_vectors('genres_vector'),
        extract_vectors('places_vector'),
        extract_vectors('normalized_avg_rating'),
        extract_vectors('normalized_ratings_count')
    ]

    combined_features = []
    for user_id in tqdm(user_id_map):
        user_feature = combine_features(
            user_id,
            user_factors,
            user_id_map,
            feature_vectors,
            train_interaction_matrix
        )
        combined_features.append(user_feature)
    combined_features = np.array(combined_features)

    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)
    return combined_features


def get_top_n_indices(arr, n):
    """Return the indices of the top n values in the array."""
    return np.argpartition(arr, -n)[-n:]


def recommend_books(user_id: int, combined_features: np.ndarray, interaction_matrix: lil_matrix,
                    user_id_map: Dict[int, int], inverse_book_id_map: Dict[int, int],
                    top_n: int = 10, include_rated_books: bool = False):
    user_id = str(user_id) if isinstance(user_id, int) else user_id

    if user_id not in user_id_map:
        return []

    user_idx = user_id_map[user_id]
    user_feature = combined_features[user_idx]
    similarities = cosine_similarity([user_feature], combined_features)[0]
    top_users_indices = get_top_n_indices(similarities, top_n + 1)[1:]

    recommended_books = set()
    for idx in top_users_indices:
        rated_books_by_similar_user = interaction_matrix[idx].nonzero()[1]
        for book_idx in rated_books_by_similar_user:
            if len(recommended_books) >= top_n:
                break
            if include_rated_books or interaction_matrix[user_idx, book_idx] == 0:
                recommended_books.add(book_idx)

    recommended_book_ids = [inverse_book_id_map[book_idx] for book_idx in list(recommended_books)[:top_n]]
    return recommended_book_ids


def predict_rating_batch(user_ids: List[int], book_ids: List[int],
                         combined_features: np.ndarray,
                         interaction_matrix: lil_matrix,
                         user_id_map: Dict[int, int], book_id_map: Dict[int, int]) -> np.ndarray:
    predicted_ratings = np.zeros(len(user_ids))

    user_indices = np.array([user_id_map[user_id] if user_id in user_id_map else -1 for user_id in user_ids])
    book_indices = np.array([book_id_map[book_id] if book_id in book_id_map else -1 for book_id in book_ids])

    user_features = combined_features[user_indices]
    similarities = cosine_similarity(user_features, combined_features)

    for i, (user_idx, book_idx) in enumerate(zip(user_indices, book_indices)):
        if user_idx == -1 or book_idx == -1:
            predicted_ratings[i] = 0
            continue

        users_indices = interaction_matrix[:, book_idx].nonzero()[0]

        if len(users_indices) == 0:
            predicted_ratings[i] = 2.5
            continue

        ratings = interaction_matrix[users_indices, book_idx].toarray().flatten()
        user_similarities = similarities[i, users_indices]

        if np.sum(user_similarities) == 0:
            predicted_ratings[i] = 2.5
        else:
            predicted_rating = np.dot(ratings, user_similarities) / np.sum(user_similarities)
            predicted_ratings[i] = min(max(predicted_rating, 1), 5)

    return predicted_ratings


def evaluate(test_ratings_df: pd.DataFrame, combined_features: np.ndarray,
             interaction_matrix: lil_matrix, user_id_map: Dict[int, int],
             book_id_map: Dict[int, int]):
    predictions = []
    targets = []
    computed_batches = set()

    batch_size = 5
    max_batch_count = 100

    all_batches = set(range(0, len(range(0, batch_size * max_batch_count, batch_size))))
    new_batches = sorted(list(all_batches - computed_batches))

    with tqdm(total=len(new_batches), smoothing=0) as pbar:
        for idx in new_batches:
            start_idx = idx * batch_size

            batch = test_ratings_df.iloc[start_idx:start_idx + batch_size]
            user_ids = batch['user_id'].values
            book_ids = batch['book_id'].values
            actual_ratings = batch['rating'].values

            predicted_ratings = predict_rating_batch(user_ids, book_ids,
                                                     combined_features,
                                                     interaction_matrix,
                                                     user_id_map, book_id_map)

            predictions.extend(predicted_ratings)
            targets.extend(actual_ratings)

            computed_batches.add(idx)

            pbar.update()
            if idx > max_batch_count:
                break

    predictions = np.array(predictions)
    targets = np.array(targets)

    rmse_value = sqrt(mean_squared_error(predictions, targets))
    mae_value = np.mean(np.abs(predictions - targets))

    print(f"RMSE: {rmse_value}")
    print(f"MAE: {mae_value}")


def main():
    project_folder = os.path.dirname(__file__)
    joblib_folder = join(project_folder, 'joblib')

    connection_string = 'mongodb://localhost:27017'
    db_name = 'app_library'
    db = connect_to_mongo(connection_string, db_name)

    if not exists(join(joblib_folder, 'books_df_vectorized.joblib')):
        if not exists(join(joblib_folder, 'model.joblib')):
            model = api.load('fasttext-wiki-news-subwords-300')
            joblib.dump(model, join(joblib_folder, 'model.joblib'))
        else:
            model = joblib.load(join(joblib_folder, 'model.joblib'))
        print('Model loaded')

        books_collection: Collection = db['books']

        books = books_collection.find({}, {'book_id': 1, 'title': 1, 'author_id': 1, 'average_rating': 1,
                                           'ratings_count': 1, 'genres': 1, 'places': 1, 'characters': 1})
        books_df = pd.DataFrame(list(books))
        books_df['genres'] = books_df['genres'].fillna('').str.join(' ')
        books_df['places'] = books_df['places'].fillna('').str.join(' ')
        books_df['characters'] = books_df['characters'].fillna('').str.join(' ')

        books_df['normalized_avg_rating'] = MinMaxScaler().fit_transform(books_df[['average_rating']])
        books_df['normalized_ratings_count'] = MinMaxScaler().fit_transform(books_df[['ratings_count']])

        with tqdm(total=len(books_df['genres']), desc='Vectorizing genres') as pbar:
            books_df['genres_vector'] = books_df['genres'] \
                .apply(lambda x: vectorize_keyword(x, model, pbar))

        with tqdm(total=len(books_df[books_df['places'] != ' ']), desc='Vectorizing places') as pbar:
            books_df['places_vector'] = books_df['places'] \
                .apply(lambda x: vectorize_keyword(x, model, pbar))

        with tqdm(total=len(books_df[books_df['characters'] != '']), desc='Vectorizing characters') as pbar:
            books_df['characters_vector'] = books_df['characters'] \
                .apply(lambda x: vectorize_keyword(x, model, pbar))

        books_df.drop(['genres', 'places', 'characters', 'average_rating', 'ratings_count'], axis=1, inplace=True)

        joblib.dump(books_df, join(joblib_folder, 'books_df_vectorized.joblib'))
        if model:
            del model
    else:
        books_df = joblib.load(join(joblib_folder, 'books_df_vectorized.joblib'))
    print('Books loaded')

    if not exists(join(joblib_folder, 'ratings_df.joblib')):
        ratings_collection: Collection = db['ratings']
        ratings_df = pd.DataFrame(list(ratings_collection.find()))
        ratings_df = ratings_df[ratings_df['book_id'].isin(books_df['book_id'])]
        ratings_df = ratings_df.groupby('user_id').filter(lambda x: len(x) >= 10)
        ratings_df = ratings_df[ratings_df['rating'] > 0]
        ratings_df.drop('_id', axis=1, inplace=True)
        joblib.dump(ratings_df, join(joblib_folder, 'ratings_df.joblib'))
    else:
        ratings_df = joblib.load(join(joblib_folder, 'ratings_df.joblib'))
    print('Ratings loaded')

    book_id_map = {book_id: idx for idx, book_id in enumerate(books_df['book_id'].unique())}
    inverse_book_id_map = {idx: book_id for book_id, idx in book_id_map.items()}
    user_id_map = {user_id: idx for idx, user_id in enumerate(ratings_df['user_id'].unique())}
    print('Maps created')

    train_ratings, test_ratings = train_test_split(ratings_df, test_size=0.2, random_state=42)
    print('Train and test ratings created')

    train_interaction_matrix = create_interaction_matrix(train_ratings,
                                                         user_id_map,
                                                         book_id_map)
    print('Train interaction matrix created')

    if not exists(join(joblib_folder, 'user_factors.joblib')):
        user_factors = apply_svd(train_interaction_matrix, n_components=200)
        joblib.dump(user_factors, join(joblib_folder, 'user_factors.joblib'))
    else:
        user_factors = joblib.load(join(joblib_folder, 'user_factors.joblib'))
    print('User factors created')

    if not exists(join(joblib_folder, 'combined_features.joblib')):
        combined_features = combine(books_df, user_factors, user_id_map,
                                    train_interaction_matrix)
        joblib.dump(combined_features, join(joblib_folder, 'combined_features.joblib'))
    else:
        combined_features = joblib.load(join(joblib_folder, 'combined_features.joblib'))
    print('Combined features created')

    evaluate(test_ratings, combined_features,
             train_interaction_matrix,
             user_id_map, book_id_map)
    print('Evaluation completed')

    user_id = 1
    recommended_books = recommend_books(user_id, combined_features,
                                        train_interaction_matrix, user_id_map,
                                        inverse_book_id_map)

    for book_id in recommended_books:
        print(f'https://www.goodreads.com/book/show/{book_id}')


if __name__ == '__main__':
    main()
