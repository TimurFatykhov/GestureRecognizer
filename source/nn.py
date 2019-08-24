import numpy as np

class GestureRecognizer():
    def __init__(self, class_num):
        self.class_num = class_num

    def predict(self, frame):
        ###
        ### Not implemented yet
        ###

        return np.random.randint(0, self.class_num)[0]