import argparse
import pickle
import numpy as np
import pandas as pd

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--csv', required=True, help="CSV path to read as dataset")
    prs.add_argument('--out', required=True, help="pickle path to save as dataset")
    prs.add_argument('--split', type=float, default=0.8, help="Train side of Train/Test split (default: %(default)s)")
    prs.add_argument('--sample', type=int, default=None, help="Number of samples (total between Train AND Test) to use from dataset (default: ALL from CSV)")
    prs.add_argument('--seed', type=int, default=1234, help="RNG seed (default: %(default)s)")
    prs.add_argument('--y-column', default='objective', help="Name of column to use as y-values (default: %(default)s)")
    prs.add_argument('--y-transform', choices=['none','rank','quantile'], default='none', help="Modification to y-column values (original values: none, rank: least-to-greatest, quantile: rank/len(data)); (default: %(default)s)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    return args

def main(args=None):
    args = parse(args)
    initial_data = pd.read_csv(args.csv)

    # Transform y-column data as directed
    if args.y_transform in ['rank','quantile']:
        initial_data[args.y_column] = initial_data[args.y_column].rank(method='first').astype(int)
        if args.y_transform == 'quantile':
            initial_data[args.y_column] /= len(initial_data)

    # Get subset in a random order with seeding
    if args.sample is None:
        # Use fraction = 1 to hardcode everything
        subset = initial_data.sample(frac=1.0, random_state=args.seed, ignore_index=True)
    else:
        subset = initial_data.sample(n=args.sample, random_state=args.seed, ignore_index=True)

    # Make the split between X/Y values
    x_cols = [_ for _ in subset.columns if _ != args.y_column]
    x = subset[x_cols].to_numpy()
    y = subset[args.y_column].to_numpy()

    # Make train/test split
    train_test_split_index = int(y.shape[0] * args.split)
    x_train = x[ :train_test_split_index, :]
    x_test  = x[train_test_split_index: , :]
    y_train = y[ :train_test_split_index]
    y_test  = y[train_test_split_index: ]

    # Make dictionary
    pickle_me = {'train_x': x_train,
                 'train_y': y_train,
                 'test_x':  x_test,
                 'test_y':  y_test,
                 }

    # Save to file
    with open(args.out, 'wb') as f:
        pickle.dump(pickle_me, f)

if __name__ == '__main__':
    main()


