from os.path import dirname, join

import joblib
import pandas as pd
from pandas import Series
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


project_folder = dirname(dirname(__file__))
joblib_folder = join(project_folder, 'joblib')


def plot_rating_distribution(series: Series, title: str) -> None:
    mean_rating = series.mean()
    std_rating = series.std()

    rating_range = np.arange(1, 5.1, 0.1)
    normal_dist = norm.pdf(rating_range, mean_rating, std_rating)

    plt.figure(figsize=(8, 5))
    plt.plot(rating_range, normal_dist, label='Normal Distribution')
    plt.fill_between(rating_range, normal_dist, 0, alpha=0.3)

    sns.histplot(series, kde=False, bins=5, stat='density', alpha=0.4, color='orange',
                 label='Rating Histogram')

    plt.title(title)
    plt.xlabel('Rating')
    plt.ylabel('Probability Density')
    plt.legend()


def main() -> None:
    # ratings_df = joblib.load(join(joblib_folder, 'ratings_df.joblib'))
    hybrid_evaluation = joblib.load(join(joblib_folder, 'hybrid/evaluation_with_tolerance.joblib'))
    evaluation_df = pd.DataFrame(hybrid_evaluation)

    collaborative_evaluation = joblib.load(join(joblib_folder, 'collaborative/evaluation_with_tolerance.joblib'))
    collaborative_evaluation_df = pd.DataFrame(collaborative_evaluation)

    plot_rating_distribution(evaluation_df['targets'], 'Actual Rating Distribution')
    plot_rating_distribution(evaluation_df['predictions'], 'Predicted Rating Distribution')

    plot_rating_distribution(collaborative_evaluation_df['predictions'], 'Predicted Collaborative Rating Distribution')

    plt.show()


if __name__ == '__main__':
    main()
