import matplotlib.pyplot as plt


class ScatterPlotter:

    def __init__(self, x_data, y_data, colors, x_size=9, y_size=7):
        plt.figure(figsize=(x_size, y_size))
        plt.scatter(x_data, y_data, c=colors, s=3, cmap="jet_r")

    def with_labels(self, x_label, y_label):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        return self

    def with_ticks(self, x_ticks, y_ticks):
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        return self

    def with_color_bar(self):
        plt.colorbar()
        return self

    def plot(self):
        plt.show()
        return self
