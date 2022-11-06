import matplotlib.pyplot as plt
import numpy as np

class AQGraphs:
    def __init__(self):
        self.graphs = dict()

    def aq_time_series_comparison_plot(self, x, y, x_name="", y_name=""):
        plt.style.use('Settings/style.mplstyle')
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
