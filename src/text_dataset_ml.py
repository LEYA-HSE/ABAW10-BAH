import pandas as pd
import polars as pl


def read_split_dataset(filepath):
   raw_df = pl.read_csv(filepath, has_header=False,
                        separator="\n", new_columns=["raw_line"])
   raw_df = raw_df.select(
                       pl.col("raw_line").str.extract_groups(r"^([^,]+),([0,1]),(.*)$")
                       ).unnest("raw_line")
   raw_df.columns = ['file', 'label', 'text']
   return raw_df.to_pandas()


train_df = read_split_dataset("data/input/split/train.txt")
val_df = read_split_dataset("data/input/split/val.txt")
test_df = read_split_dataset("data/input/split/test.txt")

print(train_df.shape, val_df.shape, test_df.shape)

train_df.to_parquet("data/output/split_train.parquet")
val_df.to_parquet("data/output/split_val.parquet")
test_df.to_parquet("data/output/split_test.parquet")

oot_df = read_split_dataset("data/test-set-30-participants-unlabeled/data/split/test.txt")
imp = pd.read_csv("data/experiments/exp_1/13_imp.csv", index_col=0)

# %%

def word_stats(df, words_list):
    all_names = list(words_list)

    df['len'] = df['text'].map(len)
    all_names.append('len')

    for x in words_list:
        name = f"first_{x}"
        df[name] = df['text'].map(lambda y: y.find(x))
        all_names.append(name)

        name = f"last_{x}"
        df[name] = df['text'].map(lambda y: y.rfind(x))
        all_names.append(name)

        df[x] = df['text'].map(lambda y: x in y).astype(int)

    return df, all_names

# words_list = ['but', 'been', 'stop', 'usually', 'love',
              # 'my', 'want', 'much', 'up', 'off']
pd.options.display.max_rows = 1000

num_top_words = 100
words_list = (
   imp['0'].abs()
   .sort_values().tail(num_top_words)
   .index.tolist()
)

concat = []
for name, dataset in [('train', train_df),
                      ('oot', oot_df)]:
   # print(name)
   dataset, all_names = word_stats(dataset, words_list)
   freq = dataset[all_names].describe().round(2)[words_list].loc['mean']
   freq = freq.to_frame()
   freq['dtype'] = name
   mean_label = dataset[words_list + ['label']].corr()['label'].round(2)
   # print(freq)
   concat.append(freq)
   # print(mean_label)

main_df = pd.concat(concat).reset_index()
main_df = main_df.groupby(['dtype', 'index'])['mean'].sum().unstack().T
print(main_df)

# %%

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_vocab_dataset(train_dataset, transformed_dataset_list,
                      ngram_range, max_features, stop_words):

   def use_vocab_only(x):
      z = str(x).lower().split(" ")
      string = " ".join([y if y in vocab else "" for y in z])
      return string.strip()

   tfidf = TfidfVectorizer(
      ngram_range=ngram_range,
      max_features=max_features,
      stop_words=stop_words,
   )
   tfidf.fit(train_dataset['text']) 
   vocab = tfidf.vocabulary_

   new_list = []
   for dataset in transformed_dataset_list:
   # use_vocab_only("hello run it he music")
      new_dataset = dataset.copy(True)
      new_dataset['text'] = new_dataset['text'].map(lambda x: use_vocab_only(x))
      new_list.append(new_dataset)

   return new_list


whole_df = pd.concat([train_df, val_df, test_df])

train_new, val_new, test_new = get_vocab_dataset(train_dataset=whole_df,
                                                 transformed_dataset_list=[train_df, val_df, test_df],
                                                 ngram_range=(1, 1),
                                                 max_features=1000,
                                                 stop_words=None)

print(train_new.shape, val_new.shape, test_new.shape)
train_new.to_parquet("data/output/split_train_1.parquet")
val_new.to_parquet("data/output/split_val_1.parquet")
test_new.to_parquet("data/output/split_test_1.parquet")

