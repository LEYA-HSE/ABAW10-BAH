import pandas as pd

from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pulearn.bagging import BaggingPuClassifier

from src.text_core import save_experiment_predictions, calc_scores


train_df = pd.read_parquet("data/output/frame_train.parquet")
val_df = pd.read_parquet("data/output/frame_val.parquet")
test_df = pd.read_parquet("data/output/frame_test.parquet")

list_datasets = [
    ('train', train_df),
    ('val', val_df),
    ('test', test_df),
]
print(train_df.shape, val_df.shape, test_df.shape)

# %%

pipeline_1 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
    )),
    ('catboost', CatBoostClassifier(
        iterations=10,
        learning_rate=0.1,
        depth=6,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_1.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_2/",
                            name='1',
                            pipe=pipeline_1,
                            datasets=list_datasets)
# %%

pipeline_2 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000,
    )),
    ('catboost', CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_2.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_2/",
                            name='2',
                            pipe=pipeline_2,
                            datasets=list_datasets)

# %%

pipeline_3 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=3000,
    )),
    ('catboost', CatBoostClassifier(
        iterations=5000,
        learning_rate=0.02,
        depth=8,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_3.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_2/",
                            name='3',
                            pipe=pipeline_3,
                            datasets=list_datasets)

# %%

pipeline_4 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        # max_features=1000,
        min_df=0.005,
        stop_words=None,
    )),
    ('logreg', LogisticRegression(C=1, random_state=42, max_iter=1000)),
])

pipeline_4.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_2/",
                            name='4',
                            pipe=pipeline_4,
                            datasets=list_datasets)

# %%

pipeline_5 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=3000,
    )),
    ('catboost', CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=5,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_5.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_2/",
                            name='5',
                            pipe=pipeline_5,
                            datasets=list_datasets)
# %%

pipeline_6 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 1),
        max_features=1000,
    )),
    ('PU', BaggingPuClassifier(
        estimator=CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            verbose=100,
            random_seed=42,
            eval_metric='Logloss',
        ),
        n_estimators=10,
        oob_score=False,
    )),
])

pipeline_6.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_2/",
                            name='6',
                            pipe=pipeline_6,
                            datasets=list_datasets)
# %%

pipeline_7 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000,
    )),
    ('PU', BaggingPuClassifier(
        estimator=LogisticRegression(C=3, random_state=42, max_iter=1000),
        n_estimators=10,
        oob_score=False,
    )),
])

pipeline_7.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_2/",
                            name='7',
                            pipe=pipeline_7,
                            datasets=list_datasets)


results, results_mean = calc_scores(root_folder="data/experiments/exp_2/",
                                    plot_names=['4', '2'])

top_names = results_mean.head(5).index.tolist()
print(results_mean.to_markdown(tablefmt="github"))
print(results[results['name'].isin(top_names)].to_markdown(tablefmt="github"))

