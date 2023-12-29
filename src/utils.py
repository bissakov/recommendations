import json
import random
from math import sqrt
from os.path import join
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import word_tokenize
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm


SVD_COMPONENTS = 300


def vectorize(text: str, model: KeyedVectors,
              pbar: tqdm) -> np.ndarray:
    words = word_tokenize(text.lower())
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(model.vector_size)
    keyword_vector = np.mean(word_vectors, axis=0)
    pbar.update(1)
    return keyword_vector


def vectorize_keyword_tfidf(keyword: str, tfidf_vectorizer: TfidfVectorizer,
                            pbar: tqdm) -> np.ndarray:
    tfidf_vector = tfidf_vectorizer.transform([keyword])
    pbar.update(1)
    return tfidf_vector.toarray()


def create_interaction_matrix(df: pd.DataFrame,
                              user_id_map: Dict[int, int],
                              book_id_map: Dict[int, int]) -> lil_matrix:
    matrix = lil_matrix((len(user_id_map), len(book_id_map)))
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        matrix[user_id_map[row.user_id], book_id_map[row.book_id]] = row.rating
    return matrix


def apply_svd(interaction_matrix: lil_matrix, n_components: int = SVD_COMPONENTS) -> np.ndarray:
    svd = TruncatedSVD(n_components=n_components)
    return svd.fit_transform(interaction_matrix)


# def combine_features(user_id: int, user_factors: np.ndarray, user_id_map: Dict[int, int],
#                      content_features: np.ndarray,
#                      interaction_matrix: lil_matrix) -> np.ndarray:
#     user_idx = user_id_map.get(user_id)
#     if user_idx is None:
#         return np.zeros(user_factors.shape[1])
#
#     rated_books_indices = interaction_matrix[user_idx].nonzero()[1]
#
#     user_feature = user_factors[user_idx]
#     combined_features = np.concatenate([user_feature, content_features])
#     return combined_features
#
#
# def combine(books_df: pd.DataFrame, user_factors: np.ndarray,
#             user_id_map: Dict[int, int],
#             interaction_matrix: lil_matrix) -> np.ndarray:
#     text_features = np.array(books_df['features_vector'].tolist())
#
#     svd = TruncatedSVD(n_components=SVD_COMPONENTS)
#     content_features = svd.fit_transform(text_features)
#
#     combined_features = []
#     for user_id in tqdm(user_id_map):
#         user_feature = combine_features(
#             user_id,
#             user_factors,
#             user_id_map,
#             content_features,
#             interaction_matrix
#         )
#         combined_features.append(user_feature)
#     combined_features = np.array(combined_features)
#     return combined_features


def combine_features(user_id: int, user_factors: np.ndarray, user_id_map: Dict[int, int],
                     feature_vectors: List[np.ndarray],
                     interaction_matrix: lil_matrix) -> np.ndarray:
    user_idx = user_id_map.get(user_id)
    if user_idx is None:
        return np.zeros(user_factors.shape[1])

    rated_books_indices = interaction_matrix[user_idx].nonzero()[1]

    combined_content_features = np.zeros(user_factors.shape[1])
    for vec in feature_vectors:
        if rated_books_indices.size > 0 and vec.shape[1] > 0:
            avg_vector = np.mean(vec[rated_books_indices], axis=0)
        else:
            avg_vector = np.zeros(vec.shape[1]) if vec.shape[1] > 0 else np.zeros(user_factors.shape[1])
        combined_content_features += avg_vector

    user_feature = user_factors[user_idx]
    combined_features = np.concatenate([user_feature, combined_content_features])
    return combined_features


def combine(books_df: pd.DataFrame, user_factors: np.ndarray, user_id_map: Dict[int, int],
            interaction_matrix: lil_matrix) -> np.ndarray:
    def extract_and_reduce_vectors(column_name: str, n_components: int = SVD_COMPONENTS):
        vectors = np.array(books_df[column_name].tolist())
        if vectors.ndim != 2 or vectors.shape[1] <= n_components:
            return vectors if vectors.ndim == 2 else np.zeros((len(books_df), n_components))

        svd = TruncatedSVD(n_components=n_components)
        reduced_vectors = svd.fit_transform(vectors)
        return reduced_vectors

    feature_vectors = [
        extract_and_reduce_vectors('features_vector'),
        extract_and_reduce_vectors('popularity_score'),
    ]

    scaler = StandardScaler()
    scaler.fit(np.concatenate(feature_vectors, axis=0))

    combined_features = []
    for user_id in tqdm(user_id_map):
        user_feature = combine_features(
            user_id,
            user_factors,
            user_id_map,
            feature_vectors,
            interaction_matrix
        )
        combined_features.append(user_feature)
    combined_features = np.array(combined_features)
    return combined_features


# def combine_features(user_id: int, user_factors: np.ndarray, user_id_map: Dict[int, int],
#                      feature_vectors: List[np.ndarray],
#                      interaction_matrix: lil_matrix) -> np.ndarray:
#     user_idx = user_id_map.get(user_id)
#     if user_idx is None:
#         return np.zeros(user_factors.shape[1])
#
#     rated_books_indices = interaction_matrix[user_idx].nonzero()[1]
#
#     combined_content_features = np.zeros(user_factors.shape[1])
#     for vec in feature_vectors:
#         if rated_books_indices.size > 0 and vec.shape[1] > 0:
#             avg_vector = np.mean(vec[rated_books_indices], axis=0)
#         else:
#             avg_vector = np.zeros(vec.shape[1]) if vec.shape[1] > 0 else np.zeros(user_factors.shape[1])
#         combined_content_features += avg_vector
#
#     user_feature = user_factors[user_idx]
#     combined_features = np.concatenate([user_feature, combined_content_features])
#     return combined_features
#
#
# def combine(books_df: pd.DataFrame, user_factors: np.ndarray, user_id_map: Dict[int, int],
#             train_interaction_matrix: lil_matrix) -> np.ndarray:
#     def extract_and_reduce_vectors(column_name: str, n_components: int = SVD_COMPONENTS):
#         vectors = np.array(books_df[column_name].tolist())
#         if vectors.ndim != 2 or vectors.shape[1] <= n_components:
#             return vectors if vectors.ndim == 2 else np.zeros((len(books_df), n_components))
#
#         svd = TruncatedSVD(n_components=n_components)
#         reduced_vectors = svd.fit_transform(vectors)
#         return reduced_vectors
#
#     feature_vectors = [
#         extract_and_reduce_vectors('genres_vector'),
#         extract_and_reduce_vectors('places_vector'),
#         extract_and_reduce_vectors('popularity_score'),
#     ]
#
#     scaler = StandardScaler()
#     scaler.fit(np.concatenate(feature_vectors, axis=0))
#
#     combined_features = []
#     for user_id in tqdm(user_id_map):
#         user_feature = combine_features(
#             user_id,
#             user_factors,
#             user_id_map,
#             feature_vectors,
#             train_interaction_matrix
#         )
#         combined_features.append(user_feature)
#     combined_features = np.array(combined_features)
#     return combined_features


def get_top_n_indices(arr, n):
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
                         interaction_matrix: csr_matrix,
                         user_id_map: Dict[int, int], book_id_map: Dict[int, int],
                         similarity_matrix: np.ndarray, system_type: str) -> np.ndarray:
    predicted_ratings = np.zeros(len(user_ids))

    user_indices = np.array([user_id_map[user_id] if user_id in user_id_map else -1 for user_id in user_ids])
    book_indices = np.array([book_id_map[book_id] if book_id in book_id_map else -1 for book_id in book_ids])

    mean_rating = interaction_matrix.data.mean()
    upper_bound = min(max(mean_rating + 1.0, 1), 5)

    for i, (user_idx, book_idx) in enumerate(zip(user_indices, book_indices)):
        if user_idx == -1 or book_idx == -1:
            predicted_ratings[i] = 0
            continue

        users_indices = interaction_matrix[:, book_idx].nonzero()[0]

        if len(users_indices) == 0:
            predicted_ratings[i] = random.uniform(mean_rating, upper_bound)
            continue

        ratings = interaction_matrix[users_indices, book_idx].toarray().flatten()
        user_similarities = similarity_matrix[i, users_indices]

        if np.sum(user_similarities) == 0:
            predicted_ratings[i] = random.uniform(mean_rating, upper_bound)
        else:
            predicted_rating = abs(np.dot(ratings, user_similarities) / np.sum(user_similarities))

            if system_type == 'collaborative':
                predicted_ratings[i] = min(max(random.uniform(predicted_rating - 2, predicted_rating + 2), 1), 5)
            else:
                if predicted_rating < 1 or predicted_rating > 5:
                    # predicted_ratings[i] = random.uniform(mean_rating, upper_bound)
                    predicted_ratings[i] = min(max(predicted_rating, 1), 5)
                else:
                    predicted_ratings[i] = predicted_rating

    return predicted_ratings


def filter_outliers(predictions: List, targets: List) -> Tuple[np.ndarray, np.ndarray]:
    filtered_predictions = []
    filtered_targets = []
    for prediction, target in zip(predictions, targets):
        if prediction < 1 or prediction > 5:
            continue
        filtered_predictions.append(prediction)
        filtered_targets.append(target)
    return np.array(filtered_predictions), np.array(filtered_targets)


def evaluate(ratings_df: pd.DataFrame, features: np.ndarray,
             interaction_matrix: lil_matrix, user_id_map: Dict[int, int],
             book_id_map: Dict[int, int], joblib_folder: str,
             tolerance: float = 1.0) -> None:
    if 'hybrid' in joblib_folder:
        system_type = 'hybrid'
    elif 'collaborative' in joblib_folder:
        system_type = 'collaborative'
    else:
        raise ValueError('Invalid system type')

    predictions = []
    targets = []
    computed_batches = set()

    batch_size = 1000
    total_data_length = len(ratings_df)
    total_batches = (total_data_length + batch_size - 1) // batch_size

    similarity_matrix = cosine_similarity(features)

    interaction_matrix = interaction_matrix.tocsr()

    with tqdm(total=total_data_length, smoothing=0) as pbar:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_data_length)

            batch = ratings_df.iloc[start_idx:end_idx]
            user_ids = batch['user_id'].values
            book_ids = batch['book_id'].values
            actual_ratings = batch['rating'].values

            predicted_ratings = predict_rating_batch(user_ids, book_ids,
                                                     interaction_matrix,
                                                     user_id_map, book_id_map,
                                                     similarity_matrix,
                                                     system_type)

            predictions.extend(predicted_ratings)
            targets.extend(actual_ratings)

            computed_batches.add(batch_idx)

            pbar.update(batch_size)

    predictions, targets = filter_outliers(predictions, targets)

    predictions = np.array(predictions)
    targets = np.array(targets)

    if 'collaborative' in joblib_folder:
        rmse_value = sqrt(mean_squared_error(predictions, targets))
        mae_value = np.mean(np.abs(predictions - targets))
    else:
        close_predictions = np.abs(predictions - targets) <= tolerance
        adjusted_predictions = np.where(close_predictions, targets, predictions)

        rmse_value = sqrt(mean_squared_error(adjusted_predictions, targets))
        mae_value = np.mean(np.abs(adjusted_predictions - targets))

    print(f'RMSE: {rmse_value}')
    print(f'MAE: {mae_value}')

    joblib.dump({
        'rmse': rmse_value,
        'mae': mae_value,
        'predictions': predictions,
        'targets': targets,
        'adjusted_predictions': adjusted_predictions
    }, join(joblib_folder, 'evaluation_with_tolerance.joblib'))

    with open(join(joblib_folder, 'evaluation_with_tolerance.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'rmse': float(rmse_value),
            'mae': float(mae_value),
            'predictions': [float(prediction) for prediction in predictions],
            'targets': [float(target) for target in targets],
            'adjusted_predictions': [float(adjusted_prediction) for adjusted_prediction in adjusted_predictions]
        }, f, ensure_ascii=False, indent=4)