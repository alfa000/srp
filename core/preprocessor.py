import tensorflow as tf

class DataPreprocessor:
    def preprocess(self, raw_data):
        input_data = {
            'crop_type': tf.constant([raw_data['crop_type']]),
            'humidity': tf.constant([int(raw_data['humidity'])], dtype=tf.int64),
            'moisture': tf.constant([int(raw_data['moisture'])], dtype=tf.int64),
            'nitrogen': tf.constant([int(raw_data['nitrogen'])], dtype=tf.int64),
            'phosphorous': tf.constant([int(raw_data['phosphorous'])], dtype=tf.int64),
            'potassium': tf.constant([int(raw_data['potassium'])], dtype=tf.int64),
            'soil_type': tf.constant([raw_data['soil_type']]),
            'temparature': tf.constant([int(raw_data['temparature'])], dtype=tf.int64),
        }

        return input_data
