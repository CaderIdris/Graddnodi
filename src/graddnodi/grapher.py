from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class GradGraphs:
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
            Path("/".join(Path(plot_path).parts[:-1])).mkdir(
                parents=True, exist_ok=True
            )
            plot.savefig(plot_path)
            plt.close(plot)
