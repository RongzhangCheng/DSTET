from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day, add_day_in_week, steps_per_day
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    df = pd.DataFrame(df)
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]

    if add_time_in_day:
        # numerical time_of_day
        tod = [i % steps_per_day /
               steps_per_day for i in range(data.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)


    if add_day_in_week:
        # numerical day_of_week
        dow = [(i // steps_per_day) % 7 for i in range(data.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    # df = pd.read_hdf(args.traffic_df_filename)
    df = np.load(args.traffic_df_filename)['data']
    df = df[..., args.target_channel]
    df = df[..., -1]
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=True,
        steps_per_day=args.steps_per_day
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * args.valid_ratio)
    num_train = round(num_samples * args.train_ratio)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":

    TRAIN_RATIO = 0.6
    TEST_RATIO = 0.2
    TARGET_CHANNEL = [0]
    FUTURE_SEQ_LEN = 12

    DATASET_NAME = "PEMS03"  # PEMS03 PEMS04 PEMS07 PEMS08
    OUTPUT_DIR = "data/" + DATASET_NAME
    DATA_FILE_PATH = "data/{0}/{0}.npz".format(DATASET_NAME)


    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default=DATA_FILE_PATH, help="Raw traffic readings.", )
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.", )
    parser.add_argument("--seq_length_y", type=int, default=FUTURE_SEQ_LEN, help="Sequence Length.", )
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--steps_per_day", type=int, default=288, help="Sequence Length.")
    parser.add_argument("--target_channel", type=list, default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float, default=TEST_RATIO, help="Validate ratio.")

    args = parser.parse_args()
    print('Sequence Length:', args.seq_length_y)
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
