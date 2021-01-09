import matplotlib.pyplot as plt


def density_plot(dataframe, x_lim=None, title=None, x_label=None, y_label=None):
    plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    dataframe.plot(kind="density", title=title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if x_lim:
        ax.set_xlim(*x_lim)


def density_histogram_plot(dataframe, x_lim=None, logy=False, title=None, x_label=None, y_label=None):
    plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    dataframe.plot(kind="hist", bins=40, density=True, logy=logy, title=title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if x_lim:
        ax.set_xlim(*x_lim)


def histogram_plot(dataframe, x_lim=None, title=None, x_label=None, y_label=None):
    plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    dataframe.plot(kind="hist", bins=40, title=title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if x_lim:
        ax.set_xlim(*x_lim)
