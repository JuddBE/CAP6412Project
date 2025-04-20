import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


class PlotGeneration:
    def __init__(self, directory):
        self.directory = directory
        self.columns_to_plot = {
            "one_person",
            "face_visible",
            "age",
            "gender",
            "race",
        }

    def plot_csv_distribution(self, filename, directory):
        df = pd.read_csv(filename)

        for col in self.columns_to_plot:
            if col in df.columns:
                df[col] = df[col].str.lower()
                no_error_df = df[df[col] != "error"]

                counts = df[col].value_counts()
                no_error_counts = no_error_df[col].value_counts()

                self.create_bar_graph(
                    col, col + " distribution", counts, directory, filename
                )
                col = col + "_noError"
                self.create_bar_graph(
                    col, col + " distribution", no_error_counts, directory, filename
                )

    def create_bar_graph(self, label, title, counts, directory, csv_filename):
        csv_filename = os.path.splitext(os.path.basename(csv_filename))[0]
        output_filename = os.path.join(directory, f"{csv_filename}_{label}.png")

        plt.figure(figsize=(10, 6))
        plt.bar(counts.index, counts.values, color="skyblue")
        plt.xlabel(label)
        plt.ylabel("count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()

    def process_all_csvs(self):
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if file.lower().endswith(".csv"):
                    csv_full_path = os.path.join(root, file)
                    print(f"Processing: {csv_full_path}")
                    self.plot_csv_distribution(csv_full_path, root)


def main():
    directory = sys.argv[1]

    plot_generator = PlotGeneration(directory)
    plot_generator.process_all_csvs()


if __name__ == "__main__":
    main()
