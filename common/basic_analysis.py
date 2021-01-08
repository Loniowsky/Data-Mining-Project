import matplotlib.pyplot as plt


def density_plot(dataframe, x_lim=None):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    dataframe.plot.density()
    dataframe.plot.hist(normed=True)
    if x_lim:
        ax.set_xlim(*x_lim)
