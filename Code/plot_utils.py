import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def plot_input_chars(df, script = 'all', n_rows=3, n_cols=10, n_samples=10, save=False):

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))

    n_samples = n_cols
    if script == 'all':
        keys = ['katakana', 'hiragana','kanji'] # list(df.true_script.unique())
        n_rows = len(keys)
        for i, key in enumerate(keys):
            df_sampled = df.groupby('true_script').get_group(key).sample(n_samples)
            row_ind = i
            for col_ind, (ind, row) in enumerate(df_sampled.iterrows()):
                axes[row_ind, col_ind].imshow(row['image_data'], cmap='gray')
                axes[row_ind, col_ind].set_title(key)
                axes[row_ind, col_ind].axis('off'

    else:
        df_in = df[df['true_script'] == script]
        chars = random.sample(list(df_in['hex_char']), n_rows)
        for i, char in enumerate(chars):
            df_sampled = f_in.groupby('hex_char').get_group(char).sample(n_samples)
            row_ind = i
            for col_ind, (ind, row) in enumerate(df_sampled.iterrows()):
                axes[row_ind, col_ind].imshow(row['image_data'], cmap='gray')
                axes[row_ind, col_ind].set_title(char)
    for ax in axes:
        ax.axis('off')
