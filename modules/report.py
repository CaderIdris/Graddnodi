import shutil

import numpy as np
import pandas as pd

class Report:
    def __init__(self, title="", subtitle="", author=""):
        self.report_text = list()
        self.report_text.append(r"\documentclass[12pt]{book}")
        self.report_text.append(r"\usepackage{Style}")
        self.report_text.append(r"\usepackage{float}")
        self.report_text.append(r"\usepackage{graphicx}")
        self.report_text.append(r"\usepackage{pgf}")
        self.report_text.append(r"\usepackage{subcaption}")
        self.report_text.append(f"\\title{{{title}\\\\ \\large {subtitle}}}")
        self.report_text.append(f"\\author{{{author}}}")
        self.report_text.append(r"\begin{document}")
        self.report_text.append(r"\maketitle")
        self.report_text.append(r"\tableofcontents")

    def add_part(self, title=""):
        self.report_text.append(f"\\part{{{title}}}")

    def add_chapter(self, title=""):
        self.report_text.append(f"\\chapter{{{title}}}")

    def add_section(self, title=""):
        self.report_text.append(f"\\section{{{title}}}")

    def add_subsection(self, title=""):
        self.report_text.append(f"\\subsection{{{title}}}")

    def add_subsubsection(self, title=""):
        self.report_text.append(f"\\subsubsection{{{title}}}")

    def add_table(self, table, caption, column_split=0, table_split=0):
        table_pages = list()
        num_of_splits = 0
        if column_split:
            num_of_splits = int(np.floor(len(table.columns) / column_split))
        if not num_of_splits:
            table_pages.append([table])
        else:
            page = list()
            for split in range(num_of_splits):
                page.append(table.iloc[:, (split*column_split):((split+1)*column_split)])
                if table_split and len(page) >= table_split:
                    table_pages.append(page.copy())
                    page = list()
            if not (len(table.columns) / column_split).is_integer():
                page.append(table.iloc[:, (num_of_splits*column_split):])
            if page:
                table_pages.append(page.copy())
        for index, t_page in enumerate(table_pages):
            self.report_text.append(r"\begin{table}[H]")
            self.report_text.append(r"\centering")
            for tab in t_page:
                self.report_text.append(tab.style.to_latex())
                self.report_text.append(" ")
            if len(table_pages) > 1:
                self.report_text.append(f"\\caption{{{caption} ({index+1}/{len(table_pages)})}}")
            else:
                self.report_text.append(f"\\caption{{{caption}}}")
            self.report_text.append(r"\end{table}")
            if index < len(table_pages) - 1:
                self.report_text.append(r"\clearpage")

    def add_multiple_pgf(self, images, caption, column_split=0, row_split=0):
        fig_pages = list() 
        num_of_rows = 1
        if column_split:
            num_of_rows = int(np.floor(len(images) / column_split))
        else: 
            column_split = len(images)
        page = list()
        for row in range(num_of_rows):
            page.append(images[row*column_split:(row+1)*column_split])
            if len(page) >= row_split:
                fig_pages.append(page.copy())
                page = list()
        if not (len(images) / column_split).is_integer():
            page.append(images[(num_of_rows+1)*column_split:])
        if page:
            fig_pages.append(page.copy())
        if column_split:
            fig_sizing = str(np.floor(10 / column_split)/ 10)
        else:
            fig_sizing = str(np.floor(10 / len(images))/ 10)
        for index, f_page in enumerate(fig_pages):
            self.report_text.append(r"\begin{figure}[H]")
            self.report_text.append(r"\centering")
            for row in f_page:
                for fig in row:
                    self.report_text.append(f"\\begin{{subfigure}}{{{fig_sizing}\\textwidth}}")
                    self.report_text.append(f"\\resizebox{{\\linewidth}}{{!}}{{\\input{{\"{fig}\"}}}}")
                    self.report_text.append(r"\end{subfigure}%")
                self.report_text.append(" ")
            if len(fig_pages) > 1:
                self.report_text.append(f"\\caption{{{caption} ({index+1}/{len(fig_pages)})}}")
            else:
                self.report_text.append(f"\\caption{{{caption}}}")
            self.report_text.append(r"\end{figure}")
            if index < len(fig_pages) - 1:
                self.report_text.append(r"\clearpage")

    def add_sideways_table(self, table, caption):
        self.report_text.append(r"\begin{sidewaystable}")
        self.report_text.append(table.style.to_latex())
        self.report_text.append(f"\\caption{{{caption}}}")
        self.report_text.append(r"\end{sidewaystable}")

    def add_sideways_pgf(self, fig, caption):
        self.report_text.append(r"\begin{sidewaysfigure}")
        self.report_text.append(r"\centering")
        self.report_text.append(f"\\resizebox{{\\pagewidth}}{{!}}{{\\input{{\"{fig}\"}}}}")
        self.report_text.append(f"\\caption{{{caption}}}")
        self.report_text.append(r"\end{sidewaysfigure}")

    def add_tables_sbs(self, tables, caption):
        self.report_text.append(r"\begin{table}[H]")
        for tab in tables:
            self.report_text.append(r"\begin{subtable}{0.4\linewidth}")
            self.report_text.append(r"\centering")
            self.report_text.append(f"{tab.style.to_latex()}")
            self.report_text.append(r"\end{subtable}%")
        self.report_text.append(f"\\caption{{{caption}}}")
        self.report_text.append(r"\end{table}")

    def add_pgf_figure(self, figure, caption):
        self.report_text.append(r"\begin{figure}[H]")
        self.report_text.append(r"\centering")
        self.report_text.append(f"\\resizebox{{\\linewidth}}{{!}}{{\\input{{\"{figure}\"}}}}")
        self.report_text.append(f"\\caption{{{caption}}}")
        self.report_text.append(r"\end{figure}")

    def add_pgf_figures_sbs(self, figures, caption):
        self.report_text.append(r"\begin{figure}[H]")
        self.report_text.append(r"\centering")
        for fig in figures:
            self.report_text.append(r"\begin{subfigure}{0.5\textwidth}")
            self.report_text.append(f"\\resizebox{{0.9\\linewidth}}{{!}}{{\\input{{\"{fig}\"}}}}")
            self.report_text.append(r"\end{subfigure}%")
        self.report_text.append(f"\\caption{{{caption}}}")
        self.report_text.append(r"\end{figure}")

    def clear_page(self):
        self.report_text.append(r"\clearpage")

    def save_tex(self, path):
        self.report_text.append(r"\end{document}")
        with open(f"{path}/Report.tex", "w+") as tex_file:
            tex_file.write("\n".join(self.report_text))
        shutil.copy("Style.sty", f"{path}/Style.sty")
