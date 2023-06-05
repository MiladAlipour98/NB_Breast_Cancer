import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./breast.csv')
print('Breast Dataset:\n', df)

def MLE(self, dataset, class_des, get_instance_likelihood):
    """
    Maximum a posteriori classifier.
    Input
        - dataset: the traning dataset
        - class_des: an aggregation of all required params (different params for different destributions),
                     in addition to prob calculations methods.
        - get_instance_likelihood: a function to calculate the likelyhood (in our case either normal_pdf
                     or multi_normal_pdf)
    """
    class_values = np.unique(dataset[:, -1])  # extract all class values
    self.classes = []
    for class_value in class_values:
        self.classes.append(class_des(dataset, class_value, get_instance_likelihood))


def predict(self, x, is_labeled=True):
    if is_labeled:
        x = x[:-1]
    posterior_list = list(map(lambda cls: cls.get_instance_posterior(x), self.classes))
    maximum_posteriori = self.classes[posterior_list.index(max(posterior_list))].class_value
    return maximum_posteriori