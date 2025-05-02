import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split
random.seed(420)


def hardmath_upsample():
    with open("./HARDMATH/hardmath_output.jsonl", 'r', encoding='utf-8') as infile:
        with open("./HARDMATH/hardmath_output2.jsonl", 'r', encoding='utf-8') as infile2:
            with open("./HARDMATH/hardmath_output3.jsonl", 'r', encoding='utf-8') as infile3:
                with open("./HARDMATH/hardmath_output4.jsonl", 'r', encoding='utf-8') as infile4:
                    data1 = [json.loads(line) for line in infile]
                    data2 = [json.loads(line) for line in infile2]
                    data3 = [json.loads(line) for line in infile3]
                    data4 = [json.loads(line) for line in infile4]
                    data = data1 + data2 + data3 + data4
                    random.shuffle(data)
                    split_idx = int(len(data) * 0.8)
                    train_data = data[:split_idx]
                    val_data = data[split_idx:]
                    zeros = [line for line in train_data if line['correct'] == 0]
                    ones = [line for line in train_data if line['correct'] == 1]
                    twos = [line for line in train_data if line['correct'] == 2]
                    threes = [line for line in train_data if line['correct'] == 3]

                    print("0: ", len(zeros))
                    print("1: ", len(ones))
                    print("2: ", len(twos))
                    print("3: ", len(threes))

                    # Upsample the smaller classes with replacement
                    # max_size = max(len(ones), len(twos), len(threes))
                    # zeros = random.sample(zeros, k=max_size)
                    # ones = random.choices(ones, k=max_size)
                    # twos = random.choices(twos, k=max_size)
                    # threes = random.choices(threes, k=max_size)

                    print("0 (upsampled): ", len(zeros))
                    print("1 (upsampled): ", len(ones))
                    print("2 (upsampled): ", len(twos))
                    print("3 (upsampled): ", len(threes))

                    upsampled_train = zeros + ones + twos + threes
                    random.shuffle(upsampled_train) 

                    # Save the upsampled train data
                    with open("./HARDMATH/hardmath_output_upsampled_train.jsonl", 'w', encoding='utf-8') as outfile:
                        for row in upsampled_train:
                            json.dump(row, outfile)
                            outfile.write('\n')
                    # Save the untouched validation data
                    with open("./HARDMATH/hardmath_output_upsampled_val.jsonl", 'w', encoding='utf-8') as outfile:
                        for row in val_data:
                            json.dump(row, outfile)
                            outfile.write('\n')


def neurips_upsample():
    with open("./NEURIPS/neurips_output.jsonl", 'r', encoding='utf-8') as infile:
        df = pd.read_json(infile, lines=True)
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['correct'], random_state=42)
        df_0 = train_df[train_df['correct'] == 0]
        df_1 = train_df[train_df['correct'] == 1]
        df_2 = train_df[train_df['correct'] == 2]
        df_3 = train_df[train_df['correct'] == 3]

        print(train_df['correct'].value_counts())
        # Upsample the smaller classes with replacement
        # max_size = max(len(df_1), len(df_2), len(df_3))
        # df_0 = df_0.sample(n=max_size, replace=True, random_state=42)
        # df_1 = df_1.sample(n=max_size, replace=True, random_state=42)
        # df_2 = df_2.sample(n=max_size, replace=True, random_state=42)
        # df_3 = df_3.sample(n=max_size, replace=True, random_state=42)

        upsampled_train = pd.concat([df_0, df_1, df_2, df_3], ignore_index=True)
        upsampled_train = upsampled_train.sample(frac=1, random_state=42)

        print("0 (upsampled): ", len(df_0))
        print("1 (upsampled): ", len(df_1))
        print("2 (upsampled): ", len(df_2))
        print("3 (upsampled): ", len(df_3))

        # Save the upsampled train data
        with open("./NEURIPS/neurips_output_upsampled_train.jsonl", 'w', encoding='utf-8') as outfile:
            for row in upsampled_train.to_dict(orient='records'):
                json.dump(row, outfile)
                outfile.write('\n')
        # Save the untouched validation data
        with open("./NEURIPS/neurips_output_upsampled_val.jsonl", 'w', encoding='utf-8') as outfile:
            for row in val_df.to_dict(orient='records'):
                json.dump(row, outfile)
                outfile.write('\n')


def deepmind_upsample():
    with open("./DEEPMIND/output.jsonl", 'r', encoding='utf-8') as infile:
        df = pd.read_json(infile, lines=True)
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['correct'], random_state=42)
        df_0 = train_df[train_df['correct'] == 0]
        df_1 = train_df[train_df['correct'] == 1]
        df_2 = train_df[train_df['correct'] == 2]
        df_3 = train_df[train_df['correct'] == 3]

        print(train_df['correct'].value_counts())
        # Upsample the smaller classes with replacement
        # max_size = max(len(df_1), len(df_2), len(df_3))
        # df_0 = df_0.sample(n=max_size, replace=True, random_state=42)
        # df_1 = df_1.sample(n=max_size, replace=True, random_state=42)
        # df_2 = df_2.sample(n=max_size, replace=True, random_state=42)
        # df_3 = df_3.sample(n=max_size, replace=True, random_state=42)

        upsampled_train = pd.concat([df_0, df_1, df_2, df_3], ignore_index=True)
        upsampled_train = upsampled_train.sample(frac=1, random_state=42)

        print("0 (upsampled): ", len(df_0))
        print("1 (upsampled): ", len(df_1))
        print("2 (upsampled): ", len(df_2))
        print("3 (upsampled): ", len(df_3))

        # Save the upsampled train data
        with open("./DEEPMIND/output_upsampled_train.jsonl", 'w', encoding='utf-8') as outfile:
            for row in upsampled_train.to_dict(orient='records'):
                json.dump(row, outfile)
                outfile.write('\n')
        # Save the untouched validation data
        with open("./DEEPMIND/output_upsampled_val.jsonl", 'w', encoding='utf-8') as outfile:
            for row in val_df.to_dict(orient='records'):
                json.dump(row, outfile)
                outfile.write('\n')

if __name__ == "__main__":
    hardmath_upsample()
    neurips_upsample()
    deepmind_upsample()
