import os
import yaml
import json

import pandas as pd
import polars as pl


def read_frames(filepath):

    df = pd.read_csv(filepath, names=["path", "label"])
    df['frame'] = (
        df['path']
        .str.replace(".jpg", "")
        .str.split("/").str[-1:].str.join("")
        .str.split("-").str[-1:].str.join("")
    )
    df['user'] = (
        df['path'].str.split("/").str[1:3].str.join("/")
    )

    df['question'] = (
        df['path'].str.split("/").str[3:4].str.join("/")
    )
    return df


# raw_df = read_frames("data/input/split-frames/train.txt")
#
# df_uq = (
#     raw_df.groupby(['user', 'question'])['label'].agg(['mean', 'count'])
#     .reset_index()
# )
# df_uq.to_csv("data/experiments/exp_2/user_question_stats.csv")
#
# df_u = (
#     raw_df.groupby(['user']).agg({'label': ['mean', 'count'], 'question': 'nunique'})
#     .reset_index()
# )
# df_u.columns = [f"{x}_{y}" for x, y in df_u.columns.ravel()]
# df_u.to_csv("data/experiments/exp_2/user_stats.csv")


def read_yaml_transcription(directory):
    def construct_python_tuple(loader, node):
        return tuple(loader.construct_sequence(node))

    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', construct_python_tuple)

    data = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            _, ext = os.path.splitext(filename)
            main_path = root.replace(directory, "").replace("Videos/", "")
            user = "/".join(main_path.split("/")[:2])
            question = main_path.split("/")[-1]
            filepath = os.path.join(root, filename)
            # print(f"{user} {question}")
            with open(filepath, "r") as f:
                data_yaml = yaml.safe_load(f)

            for indx, chunk in enumerate(data_yaml['chunks']):
                data.append({
                    "user": user,
                    "question": question,
                    "chunk_id": indx,
                    "start_sec": chunk['timestamp'][0],
                    "end_sec": chunk['timestamp'][1],
                    "duration": chunk['timestamp'][1] - chunk['timestamp'][0],
                    "text": chunk['text'],
                })

    data = pd.DataFrame(data)
    return data


def read_frame_dataset(filepath, df_script):
    raw_df = pl.read_csv(filepath,
                         has_header=False,
                         separator="\n",
                         new_columns=["raw_line"])
    raw_df = raw_df.select(
        pl.col("raw_line").str.extract_groups(r"^([^,]+),([0,1]),(.*)$")
    ).unnest("raw_line")
    raw_df.columns = ['file', 'label', 'text']
    raw_df = raw_df.to_pandas()
    raw_df['user'] = (
        raw_df['file'].str.split("/").str[1:3].str.join("/")
    )

    raw_df['question'] = (
        raw_df['file'].str.split("/").str[3:4].str.join("/")
    )

    raw_df = raw_df.drop(['text', 'file'], axis=1)
    df = raw_df.merge(df_script, on=['user', 'question'], how='left')

    return df

# %%


df_script = read_yaml_transcription("data/input/transcription/")
train_df = read_frame_dataset("data/input/split/train.txt", df_script)
val_df = read_frame_dataset("data/input/split/val.txt", df_script)
test_df = read_frame_dataset("data/input/split/test.txt", df_script)

train_df.to_parquet("data/output/frame_train.parquet")
val_df.to_parquet("data/output/frame_val.parquet")
test_df.to_parquet("data/output/frame_test.parquet")
