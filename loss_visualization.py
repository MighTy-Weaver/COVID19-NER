import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mpl.use('Agg')
loss_list = os.listdir("./results/")
for filename in loss_list:
    if ".png" in filename:
        os.remove("./results/{}".format(filename))
    elif "training" in filename and "loss" in filename:
        training_loss = np.load("./results/{}".format(filename), allow_pickle=True)
        validation_file = filename.replace("training", "validation")
        validation_loss = np.load("./results/{}".format(validation_file), allow_pickle=True)
        parameters = filename.replace("training_loss_", "").replace(".npy", "")
        # Do some visualization, set the style and the font size, figure size
        sns.set_style(style="darkgrid")
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (20, 10)

        # Plot the learning curve
        plt.plot(training_loss, 'b-o', label="Training Loss Curve")
        plt.plot(validation_loss, 'r-o', label="Validation Loss Curve")

        # Label the plot
        title_param = parameters.replace("e", "model with ").replace("_BS", " epoch trained and batch size ")
        plt.title("Loss Curve for Every Epoch trained for {}".format(title_param))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("./results/loss_fig_{}.png".format(parameters))
        plt.clf()
    elif "training" in filename and "accuracy" in filename:
        training_accuracy = np.load("./results/{}".format(filename), allow_pickle=True)
        validation_file = filename.replace("training", "validation")
        validation_accuracy = np.load("./results/{}".format(validation_file), allow_pickle=True)
        parameters = filename.replace("training_accuracy_", "").replace(".npy", "")
        # Do some visualization, set the style and the font size, figure size
        sns.set_style(style="darkgrid")
        sns.set(font_scale=1)
        plt.rcParams["figure.figsize"] = (30, 15)

        # Plot the learning curve
        plt.plot(training_accuracy, 'b-o', label="Training Accuracy Curve")
        plt.plot(validation_accuracy, 'r-o', label="Validation Accuracy Curve")

        # Label the plot
        title_param = parameters.replace("e", "model with ").replace("_BS", " epoch trained and batch size ")
        plt.title("Accuracy Curve for Every Epoch trained for {}".format(title_param))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("./results/accuracy_fig_{}.png".format(parameters))
        plt.clf()
