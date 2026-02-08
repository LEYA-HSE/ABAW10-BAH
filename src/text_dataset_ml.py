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
