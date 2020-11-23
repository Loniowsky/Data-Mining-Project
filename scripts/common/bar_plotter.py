import matplotlib.pyplot as plt


class BarPlotter:

    def __init__(self, x_data, y_data):
        plt.figure()
        plt.bar(x_data, y_data)

    def with_title(self, title):
        plt.title(title)
        return self

    def with_ticks_rotation(self, x_rotation=0, y_rotation=0):
        plt.xticks(rotation=x_rotation)
        plt.yticks(rotation=y_rotation)
        return self

    def plot(self):
        plt.show()
        return self
