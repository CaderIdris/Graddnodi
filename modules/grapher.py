import matplotlib.pyplot as plt
import numpy as np


class Graphs:
    def __init__(self):
        self.graphs = dict()

    def bar_chart(self, data: dict):
        plt.style.use("Settings/style.mplstyle")
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_labels = list(data.keys())
        bar_cords = np.arange(len(bar_labels))
        bar_heights = list(data.values())
        print(bar_cords, bar_labels, bar_heights)
        ax.bar(x=bar_cords, height=bar_heights, label=bar_labels)
        plt.show(fig)

    def grouped_bar_chart(self, data: dict):
        # what's going on?
        plt.style.use("Settings/style.mplstyle")
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_labels = list(data.keys())
        bar_cords = np.arange(len(bar_labels))
        sub_bars_len = len(list(list(data.values())[0].keys()))
        for col_name, col_values in data.items():

            print(
                bar_cords + (index / sub_bars_len),
                sub_cols.values(),
                1 / sub_bars_len,
                sub_name,
            )
            ax.bar(
                x=bar_cords + (index / sub_bars_len),
                height=sub_cols.values(),
                width=1 / sub_bars_len,
                label=sub_name,
            )
        plt.show(fig)


class AQGraphs(Graphs):
    def __init__(self):
        Graphs.__init__(self)
        self.graphs = dict()

    def time_series_comparison_plot(self, x, y, x_name="", y_name=""):
        plt.style.use("Settings/style.mplstyle")
        x_vals = x["Values"]
        y_vals = y["Values"]
        dates = x["Datetime"]
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(dates, x_vals, label=y_name)
        ax.plot(dates, y_vals, label=x_name)
        x_null = x_vals.isnull()
        y_null = y_vals.isnull()
        x_or_y_null = np.logical_or(x_null, y_null)
        first_datetime = dates[x_null.loc[~x_or_y_null].index[0]]
        last_datetime = dates[x_null.loc[~x_or_y_null].index[-1]]
        ax.legend()
        ax.set_xlim(first_datetime, last_datetime)
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Concentration")
