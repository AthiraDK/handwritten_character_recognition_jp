from data_utils import *
from models import M11
import os
from tensorflow import keras

project_path = '/home/athira/Code_Mania/Deep_dive/handwritten_character_recognition_jp/'
data_path = os.path.join(project_path, 'Data')
yaml_path = os.path.join(project_path, 'data_formats.yaml')
etl_data_obj = ETLDataset(data_path, yaml_path)
(X_train, y_train, X_test, y_test) = etl_data_obj.get_train_test(script='all', task='identify_script')

# A binary matrix representation of the input. (One hot encoding)
Y_train = keras.utils.to_categorical(y_train)
Y_test = keras.utils.to_categorical(y_test)

n_classes = max(y_train) + 1

model_task1 = M11(n_classes = n_classes)
model_task1.compile(optimizer='adam',
              loss= 'categorical_crossentropy',
              metrics=['accuracy'])
# history = model_task1.fit(X_train, Y_train, epochs=20,  batch_size = 16, validation_data=(X_test, Y_test))
