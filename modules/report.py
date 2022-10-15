import shutil

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

    def add_table(self, table, caption):
        pass

    def add_sideways_table(self, table, caption):
        self.report_text.append(r"\begin{sidewaystable}")
        self.report_text.append(table.style.to_latex())
        self.report_text.append(f"\\caption{{{caption}}}")
        self.report_text.append(r"\end{sidewaystable}")

    def add_tables_sbs(self, tables, caption):
        self.report_text.append(r"\begin{table}[H]")
        for tab in tables:
            self.report_text.append(r"\begin{subtable}{0.4\linewidth}")
            self.report_text.append(r"\centering")
            self.report_text.append(f"{tab.style.to_latex()}")
            self.report_text.append(r"\end{subtable}%")
        self.report_text.append(f"\\caption{{{caption}}}")
        self.report_text.append(r"\end{table}")

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
