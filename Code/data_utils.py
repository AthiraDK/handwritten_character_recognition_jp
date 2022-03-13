import os
import glob

import numpy as np
import pandas as pd

import io
import struct
import yaml

from sklearn.model_selection import train_test_split
from preprocessing_utils import *

class ETLDataset:

    def __init__(self, datapath='/home/athira//Data/', yaml_path='..//data_formats.yaml'):
        self.datapath = datapath
        self.yaml_path = yaml_path
        self.map_hex = {}

    def read_data_yaml(self, key='ETL-1'):
        with open(self.yaml_path, 'r') as stream:
            try:
                self.parsed_yaml = yaml.safe_load(stream)
                meta_dict = self.parsed_yaml[key]
                return meta_dict
            except yaml.YAMLError as e:
                print(f'Exception occured : {e} \n Empty dictionary returned')
                return {}

    def update_dict_hex(self, script, script_meta):

        start_hex = script_meta['start_hex']
        end_hex = script_meta['end_hex']
        script_range = [hex(jap) for jap in np.arange(start_hex, end_hex)]
        self.dict_hex[script] = script_range

    def map_hex_to_script(self, hex_char):

        test_script = []
        for script in self.dict_hex.keys():

            if hex_char in self.dict_hex[script]:
                test_script.append(script)

        if len(test_script) == 0:
            return 'not_japanese'
        elif len(test_script) == 1:
            return test_script[0]
        else:
            ScriptError(f'Script identity is ambiguous, matched with multiple ({len(test_script)}) scripts')
            return None# hi


    def get_dataframe(self, script='all'):

        if script == 'all':
            yaml_keys = ['katakana', 'hiragana','kanji']
        else:
            yaml_keys = [script]

        folder_names = []
        list_etl_files = []
        for i, key in enumerate(yaml_keys):
            script_meta = self.read_data_yaml(key)
            self.update_dict_hex(key, script_meta)
            data_source = script_meta['data_source']

            if data_source not in folder_names:
                folder_names.append(data_source)
                file_format = script_meta['file_format']

                etl_meta = self.read_data_yaml(file_format)
                folder_path = os.path.join(self.datapath, data_source)
                file_paths = sorted([path for path in glob.glob(folder_path) if 'INFO' not in path])

                for fpath in file_paths:
                    list_etl_chars = self.read_etl_file(key, fpath, etl_meta)
                    list_etl_files.extend(list_etl_chars)

        df_etl_data = pd.DataFrame(list_etl_files)
        # find true script
        df_etl_data['true_script'] = df_etl_data['hex_char'].apply(lambda x:map_hex_to_script(x))
        if script == 'all':
            return df_etl_data[df_etl_data['true_script'] != 'not_japanese']
        else :
            return df_etl_data[df_etl_data['true_script'] == script]


    def get_train_test(self, script='all', task='identify_script'):

        df_data = self.get_dataframe(script)
        df_train, df_test = train_test_split(df_data, test_size= 0.2, stratify=df_data['char_uid'])

        X_train = np.concatenate(list(df_train['image_data']))
        X_test = np.concatenate(list(df_test['image_data']))
        X_train = X_train.reshape((-1, 1, 64, 64))
        X_test = X_test.reshape((-1, 1, 64, 64))

        if task == 'identify_script':
            map_scripts = {script: ind for ind, script in enumerate(list(df_data['true_script'].unique()))}
            n_classes = len(map_scripts.keys())
            if (script == 'all') & (n_classes == 3):
                y_train = [map_scripts[label] for label in list(df_train['true_script'])]
                y_test = [map_scripts[label] for label in list(df_test['true_script'])]
            else:
                ScriptError(f'Wrong dataset!, {script} script_type, {n_classes} categories found')

        elif task == f'identify_char':
            map_chars = {char: ind for ind, char in enumerate(list(df_data['char_uid'].unique()))}
            y_train = [map_chars[label] for label in list(df_train['char_uid'])]
            y_test = [map_chars[label] for label in list(df_test['char_uid'])]
            n_classes = len(map_chars.keys())
        else:
            print('Unfamiliar task name !')
        return (X_train, y_train, X_test, y_test)

    def read_etl_file(self, script, fpath, meta_data):
        # One single etl file contains multiple records. n_records to be exact
        # this function unpacks the bytefile and returns the data in the form of a list of dictionaries

        # Inner functions
        def count_records():
            file_size = os.path.getsize(fpath)
            n_records = int(file_size/ record_size)
            print(f'Number of records in {fpath}: {n_records}')
            return n_records

        def extract_image():
            from PIL import Image
            W, H = meta_data['resolution'][0],meta_data['resolution'][1]
            im_size = (W, H)
            img_ind = meta_data['image_data_index']
            bit_depth = meta_data['bit_depth']
            img = Image.frombytes('F',im_size, record_data[img_ind], 'bit', bit_depth)
            img_ = img.convert('L')

            bin_img = binarize_img(img_)
            bin_img = Image.fromarray(bin_img)
            bin_img = bin_img.resize((64,64))

            if script =='katakana':
                bin_img = center_img(bin_img)
            image_array =  resize_img(bin_img)#
            return image_array


        record_size = meta_data["record_size"]
        n_records = count_records()
        file_ind = os.path.basename(fpath)

        with open(fpath, 'rb') as f_obj:
            etl_file_data = []
            for i in range(n_records):
                record = f_obj.read(record_size)
                record_data = struct.unpack(meta_data['record_format'],record)
                image_as_array = extract_image()

                char_ind = meta_data['char_code_index']
                char_uid = record_data[char_ind]
                hex_char = hex(char_uid)
                dict_char = {'script': script,
                             'hex_char': hex_char,
                             'char_uid': char_uid,
                             'file_ind': file_ind,
                             'record_data': record_data,
                             'image_data': image_as_array
                            }
                etl_file_data.append(dict_char)
        return etl_file_data

class ScriptError(Exception):
    def __init__(self, message):
        print(message)
