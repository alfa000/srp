from strategy import PredictionStrategy
import tensorflow as tf

class SVMStrategy(PredictionStrategy):
    def execute(self, data):
        options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        model = tf.keras.models.load_model('./save_model/2', options=options)
        return model.predict(data)