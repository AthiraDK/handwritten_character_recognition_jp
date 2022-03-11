import os
import glob

import numpy as np
import pandas as pd

import io
import struct
import yaml

class ETLDataset:

    def __init__(self, datapath='../Data/', yaml_path='../data_formats.yaml'):
        self.datapath = datapath

    def get_dataframe(self, script='all'):

        return df_data

    def get_train_test(self, script='all',task='task1'):
        from sklearn.model_selection import train_test_split
        
        return (X_train, y_train, X_test, y_test)
