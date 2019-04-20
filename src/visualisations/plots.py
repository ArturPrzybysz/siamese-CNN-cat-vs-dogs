import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_dist_var(dd_distance, cd_distance, cc_distance):
    epochs = np.arange(1, len(dd_distance) + 1)
    sns.lineplot(x=dd_distance, y=epochs)
    plt.show()
    sns.lineplot(x=cd_distance, y=epochs)
    plt.show()
    sns.lineplot(x=cc_distance, y=epochs)
    plt.show()
