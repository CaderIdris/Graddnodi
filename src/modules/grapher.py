from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Graphs:
    def __init__(self):
        self.graphs = dict()

    def bar_chart(self, data: dict, name: str = ""):
        plt.style.use("Settings/style.mplstyle")
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_labels = list(data.keys())
        bar_cords = np.arange(len(bar_labels))
        bar_heights = list(data.values())
        ax.bar(x=bar_cords, height=bar_heights)
        ax.set_xticks(bar_cords, labels=bar_labels, rotation=45)
        self.graphs[name] = fig

    def save_plots(self, path: str):
        for key, plot in self.graphs.items():
            plot_path = f"{path}/{key}.pgf"
            Path("/".join(Path(plot_path).parts[:-1])).mkdir(parents=True, exist_ok=True)
            plot.savefig(plot_path)
            plt.close(plot)


class AQGraphs(Graphs):
    def __init__(self):
        Graphs.__init__(self)

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
        self.graphs["Time Series"] = fig
