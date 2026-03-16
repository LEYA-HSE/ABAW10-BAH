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

pd.options.display.max_rows = 1000

num_top_words = 100
words_list = (
   imp['0'].abs()
   .sort_values().tail(num_top_words)
   .index.tolist()
)

concat = []
for name, dataset in [('train', train_df)]:
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
      new_dataset = dataset.copy(True)
      new_dataset['text'] = new_dataset['text'].map(lambda x: use_vocab_only(x))
      new_list.append(new_dataset)

   return new_list


whole_df = pd.concat([train_df, val_df, test_df])

train_new, val_new, test_new = get_vocab_dataset(train_dataset=whole_df,
                                                 transformed_dataset_list=[train_df, val_df, test_df],
                                                 ngram_range=(1, 1),
                                                 max_features=2000,
                                                 stop_words=None)

print(train_new.shape, val_new.shape, test_new.shape)
train_new.to_parquet("data/output/split_train_1.parquet")
val_new.to_parquet("data/output/split_val_1.parquet")
test_new.to_parquet("data/output/split_test_1.parquet")

# %%

import nltk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

# %%

import nlpaug.augmenter.word as naw


def get_augmented_df(df, aug):
   new_df = df.copy(True)
   new_texts = []
   for text in new_df['text'].values:
      augmented_text = aug.augment(text, n=1)
      new_texts.append(augmented_text[0])
   
   new_df['text'] = new_texts
   return new_df


# Create a synonym replacement augmenter
aug = naw.SynonymAug(aug_p=0.3)

train_2_df = get_augmented_df(train_df, aug)
train_2_df = pd.concat([train_df, train_2_df])
train_2_df.to_parquet("data/output/split_train_2.parquet")

# %%

import nlpaug.augmenter.word as naw

# text = "The quick brown fox jumps over the lazy dog."
aug = naw.RandomWordAug(action="swap")

train_3_df = get_augmented_df(train_df, aug)
train_3_df = pd.concat([train_df, train_3_df])
train_3_df.to_parquet("data/output/split_train_3.parquet")

# %%

aug = naw.SynonymAug(aug_p=0.3)
df_1 = get_augmented_df(train_df, aug)

aug = naw.RandomWordAug(action="swap")
df_2 = get_augmented_df(train_df, aug)

train_4_df = pd.concat([train_df, df_1, df_2, train_df])
train_4_df.to_parquet("data/output/split_train_4.parquet")
