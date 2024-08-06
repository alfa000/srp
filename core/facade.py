from strategies.random_forest import RandomForestStrategy
from strategies.svm import SVMStrategy
from model import FertilizerModel
from preprocessor import DataPreprocessor
from observer import ModelMonitor, Observer

class FertilizerPredictorFacade:
    def __init__(self, strategy_type):
        self.model = FertilizerModel()
        self.preprocessor = DataPreprocessor()
        self.monitor = ModelMonitor()
        self.monitor.add_observer(Observer())
        self.strategy = self._select_strategy(strategy_type)

    def _select_strategy(self, strategy_type):
        if strategy_type == 'random_forest':
            return RandomForestStrategy()
        elif strategy_type == 'svm':
            return SVMStrategy()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def load_model(self):
        return FertilizerModel()
    
    def predict_fertilizer(self, raw_data):
        processed_data = self.preprocessor.preprocess(raw_data)
        prediction = self.strategy.execute(processed_data)
        self.monitor.notify_observers("Prediction made using strategy: " + self.strategy.__class__.__name__)
        return prediction
