import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_accuracy(accuracy):
    epochs = np.arange(1, len(accuracy) + 1)
    sns.lineplot(x=epochs, y=accuracy)
    plt.show()
