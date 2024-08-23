from numba import jit
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json


def load_excel_data(directory):
    """
    Load all Excel files and their sheets from the specified directory into a list of dataframes.

    Parameters:
    directory (str): The path to the folder containing the Excel files.

    Returns:
    list: A list of pandas dataframes containing the data from each sheet in the Excel files.
    """
    # Initialize an empty list to store dataframes
    datasets = []
    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(directory, filename)

            # Load the Excel file
            excel_file = pd.ExcelFile(file_path)

            # Iterate over each sheet in the Excel file
            i = 0
            for sheet_name in excel_file.sheet_names:
                # if i > 0:
                #     break
                # Load the sheet into a dataframe and append to the datasets list
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                datasets.append(df)
                i += 1

    return datasets


def save_datasets_to_excel(datasets, output_file):
    """
    Save a list of dataframes to a single Excel file with each dataframe in a separate sheet.

    Parameters:
    datasets (list): A list of tuples containing filename, sheet name, and dataframe.
    output_file (str): The path to the output Excel file.
    """
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for i, (df) in enumerate(datasets):
            sheet_name_clean = f"Sheet_{i}"[
                :31
            ]  # Excel sheet names must be <= 31 chars
            df.to_excel(writer, sheet_name=sheet_name_clean, index=False)


def load_json(directory):
    with open(directory, "r") as file:
        data = json.load(file)
    return data


def save_json(data, directory):
    with open(directory, "w") as file:
        json.dump(data, file, indent=4)


def add_to_subplot(axis, x, y, xlabel=None, ylabel=None, color="blue"):
    plt.rcParams.update({"font.size": 14})
    axis.scatter(0, 0, alpha=0)
    axis.plot(
        x,
        y,
        color=color,
        label=ylabel,
    )
    axis.minorticks_on()
    axis.legend()
    axis.set_xlabel(xlabel)
    axis.grid(which="both")


@jit(nopython=True)
def calculate_squared_errors(y1, y2):
    squared_errors = np.zeros(len(y1))
    for i in range(len(y1)):
        squared_errors[i] = (y1[i] - y2[i]) ** 2
    return squared_errors


@jit(nopython=True)
def calculate_sum_root_mean_squared_errors(interpolated_array_, simulated_array_):
    sum_root_mean_squared_errors = 0.0
    for column in range(len(interpolated_array_)):
        y1 = interpolated_array_[column]
        y2 = simulated_array_[column]
        n = len(y1)
        sum_squared_error = 0.0
        for j in range(n):
            sum_squared_error += (y1[j] - y2[j]) ** 2
        root_mean_squared_errors = np.sqrt(sum_squared_error / n)
        sum_root_mean_squared_errors += root_mean_squared_errors
    return sum_root_mean_squared_errors
