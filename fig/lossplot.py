from sasa_db.crawler import Crawler
import sqlite3
import pickle
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from testing import plot_single_layer
from hyperparameters import *
#%%
name = "square_fix"
with open(f"data/logs/{name}.pickle", "rb") as f:
    hist = pickle.load(f)
for h in hist:
    print(h)
hist['discrete_out_accuracy']
#set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")

#enable latex rendering
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)

def cont_plot(hist):
    fig, ax = plt.subplots()
    N = np.arange(1, EPOCHS+1)
    ax.plot(N, hist['continuous_out_loss'], label=r"training continuous loss", color="k")
    ax.plot(N, hist['val_continuous_out_loss'], label=r"traing continuous loss", color="r", linestyle="--")
    ax.set_xlabel("Epoch", fontsize=16,)
    ax.set_ylabel("Loss", fontsize=16,)
    ax.legend(loc="upper right", fontsize=16)
    fig.savefig(f"data/plots/{name}_cont.pdf")

def dis_plot(hist):
    fig, ax = plt.subplots()
    N = np.arange(1, EPOCHS+1)
    ax.plot(N, hist['discrete_out_accuracy'], label=r"training discrete acc", color="k")
    ax.plot(N, hist['val_discrete_out_accuracy'], label=r"validation discrete acc", color="r", linestyle="--")
    ax.set_xlabel("Epoch", fontsize=16,)
    ax.set_ylabel("Accuracy", fontsize=16,)
    ax.legend(loc="lower right", fontsize=16)
    fig.savefig(f"data/plots/{name}_dis.pdf")

#%%
dis_plot(hist)
cont_plot(hist)

"""
ax.set_xticks(np.arange(2,EPOCHS+1, 2))
ax.plot(N, H.history["discrete_out_loss"],  label=r"total loss", color="k")
ax.plot(N, H.history["discrete_out_accuracy"], label=r"train discrete acc", color="r")
ax.plot(N, H.history["continuous_out_accuracy"],  label=r"train continuous acc", color="b")
ax.plot(N, H.history["val_discrete_out_accuracy"],  label=r"val. discrete acc", color="r", linestyle="--")
ax.plot(N, H.history["val_continuous_out_accuracy"],  label=r"val. continuous acc", color="b", linestyle="--")
ax.plot(N, H.history["loss"],  label=r"total loss", color="k")
ax.plot(N, H.history["val_loss"],  label=r"total loss", color="r")
ax.set_title("Training Loss and Accuracy", fontsize=16,)
"""
