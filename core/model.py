import tensorflow as tf
import tensorflow_decision_forests as tfdf

class FertilizerModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FertilizerModel, cls).__new__(cls, *args, **kwargs)
            cls._instance.model = cls.load_model()
        return cls._instance

    @staticmethod
    def load_model():
        options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        return tf.keras.models.load_model('./save_model/1', options=options)

    def predict(self, data):
        predictions = self.load_model().predict(data)
        predicted_label = tf.argmax(predictions, axis=1)[0].numpy()

        label_mappings = {
            0: "NPK 10-26-26",
            1: "NPK 14-35-14",
            2: "NPK 17-17-17",
            3: "NPK 20-20-0",
            4: "NPK 28-28-0",
            5: "NPK DAP",
            6: "NPK Urea"
        }
        return label_mappings[predicted_label]
