class Observer:
    def update(self, message):
        pass

class TrainingLogger(Observer):
    def update(self, message):
        print(f"Training Log: {message}")

class ModelTrainer:
    def __init__(self):
        self.observers = []
    
    def add_observer(self, observer):
        self.observers.append(observer)
    
    def notify_observers(self, message):
        for observer in self.observers:
            observer.update(message)
