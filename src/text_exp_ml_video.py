import pandas as pd

from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

from src.text_core import save_experiment_predictions, calc_scores

train_df = pd.read_csv("data/input/video-text/BAH_train_Qwen3-VL-4B-Instruct.csv")
val_df = pd.read_csv("data/input/video-text/BAH_val_Qwen3-VL-4B-Instruct.csv")
test_df = pd.read_csv("data/input/video-text/BAH_test_Qwen3-VL-4B-Instruct.csv")


list_datasets = [
    ('train', train_df),
    ('val', val_df),
    ('test', test_df),
]
print(train_df.shape, val_df.shape, test_df.shape)

# %%

pipeline_1 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
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

pipeline_1.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='1',
                            pipe=pipeline_1,
                            datasets=list_datasets,
                            )
# %%

pipeline_2 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
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

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='2',
                            pipe=pipeline_2,
                            datasets=list_datasets,
                            )

# %%

pipeline_3 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100,
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

pipeline_3.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='3',
                            pipe=pipeline_3,
                            datasets=list_datasets,
                            )
# %%

pipeline_4 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=500,
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

pipeline_4.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='4',
                            pipe=pipeline_4,
                            datasets=list_datasets)

# %%

pipeline_5 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
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

pipeline_5.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='5',
                            pipe=pipeline_5,
                            datasets=list_datasets)
# %%

pipeline_6 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
    )),
    ('catboost', CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_6.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='6',
                            pipe=pipeline_6,
                            datasets=list_datasets)
# %%

pipeline_7 = Pipeline([
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

pipeline_7.fit(train_df['text'], train_df['label'])

coef = pipeline_7.steps[-1][1].get_feature_importance()
columns = pipeline_7.steps[0][1].get_feature_names_out()
imp = pd.Series(coef, index=columns)
imp = imp.sort_values(ascending=False)
imp.to_csv("data/experiments/exp_6/7_imp.csv")

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='7',
                            pipe=pipeline_7,
                            datasets=list_datasets)
# %%

pipeline_8 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
    )),
    ('catboost', CatBoostClassifier(
        iterations=50,
        learning_rate=0.05,
        depth=3,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_8.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='8',
                            pipe=pipeline_8,
                            datasets=list_datasets)
# %%

pipeline_9 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
    )),
    ('catboost', CatBoostClassifier(
        iterations=500,
        learning_rate=0.002,
        depth=4,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_9.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='9',
                            pipe=pipeline_9,
                            datasets=list_datasets)
# %%

pipeline_10 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 1),
        max_features=100,
        stop_words=None,
    )),
    ('catboost', CatBoostClassifier(
        iterations=500,
        learning_rate=0.002,
        depth=4,
        # verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_10.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='10',
                            pipe=pipeline_10,
                            datasets=list_datasets)
# %%

pipeline_11 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 1),
        # max_features=100,
        min_df=0.005,
        stop_words=None,
    )),
    ('catboost', CatBoostClassifier(
        iterations=500,
        learning_rate=0.002,
        depth=4,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_11.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='11',
                            pipe=pipeline_11,
                            datasets=list_datasets)
# %%

pipeline_12 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 1),
        # max_features=1000,
        min_df=0.005,
        stop_words=None,
    )),
    ('svd', TruncatedSVD(n_components=200, n_iter=100, random_state=42)),
    ('catboost', CatBoostClassifier(
        iterations=500,
        learning_rate=0.002,
        depth=4,
        verbose=100,
        random_seed=42,
        eval_metric='Logloss'
    ))
])

pipeline_12.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='12',
                            pipe=pipeline_12,
                            datasets=list_datasets)

# %%

pipeline_13 = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 1),
        max_features=300,
        # min_df=0.005,
        stop_words=None,
    )),
    ('logreg', LogisticRegression(C=0.3, random_state=42, max_iter=1000)),
])

pipeline_13.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='13',
                            pipe=pipeline_13,
                            datasets=list_datasets)

coef = pipeline_13.steps[-1][1].coef_
columns = pipeline_13.steps[0][1].get_feature_names_out()
imp = pd.Series(coef[0], index=columns)
imp = imp.sort_values()
imp.to_csv("data/experiments/exp_6/13_imp.csv")

# %%

pipeline_14 = Pipeline([
    ('count', CountVectorizer(
        ngram_range=(1, 2),
        max_features=500,
        # min_df=0.005,
        stop_words=None,
    )),
    ('logreg', LogisticRegression(C=0.3, random_state=42, max_iter=1000)),
])

pipeline_14.fit(train_df['text'], train_df['label'])

save_experiment_predictions(root_folder="data/experiments/exp_6/",
                            name='14',
                            pipe=pipeline_14,
                            datasets=list_datasets)


results, results_mean = calc_scores(root_folder="data/experiments/exp_6/",
                                    plot_names=['6', '7'])

top_names = results_mean.head(5).index.tolist()
print(results_mean.to_markdown(tablefmt="github"))
print(results[results['name'].isin(top_names)].to_markdown(tablefmt="github"))

