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


def add_to_subplot(
    axis, x, y, xlabel=None, ylabel=None, color=plt.get_cmap("tab20")(0), linestyle="-"
):
    axis.scatter(0, 0, alpha=0)
    axis.plot(
        x,
        y,
        color=color,
        label=ylabel,
        linestyle=linestyle,
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
        if column <= 4:
            y1 = interpolated_array_[column]
            y2 = simulated_array_[column]
            n = len(y1)
            sum_squared_error = 0.0
            for j in range(n):
                sum_squared_error += (y1[j] - y2[j]) ** 2
            root_mean_squared_errors = np.sqrt(sum_squared_error / n)
            sum_root_mean_squared_errors += root_mean_squared_errors
    return sum_root_mean_squared_errors


@jit(nopython=True)
def get_rise_index(time_series, setpoint):
    time_series = np.array(time_series)

    if time_series[0] > setpoint:
        time_series = -time_series
        setpoint = -setpoint

    # Cari nilai awal dari time series
    start_value = time_series[0]
    # Hitung 10% dan 90% dari nilai maksimum
    threshold_10 = start_value + 0.1 * np.abs(setpoint - start_value)
    threshold_90 = start_value + 0.9 * np.abs(setpoint - start_value)

    # Cari index ketika time series memiliki nilai terdekat dengan 10% dan 90% threshold
    rise_10_index = len(time_series) - 1
    for i in range(len(time_series)):
        if time_series[i] > threshold_10:
            rise_10_index = i
            break

    rise_90_index = len(time_series) - 1
    for i in range(len(time_series)):
        if time_series[i] > threshold_90:
            rise_90_index = i
            break

    return rise_10_index, rise_90_index


@jit(nopython=True)
def get_settling_index(time_series, setpoint, threshold=0.02):
    time_series = np.array(time_series)

    # Toleransi terhadap steady state
    bound = np.abs(setpoint - time_series[0]) * np.abs(threshold)
    if setpoint == 0:
        bound = threshold

    lower_bound = setpoint - bound
    upper_bound = setpoint + bound

    # Cari waktu ketika time series pertama kali melewati threshold 2%
    settling_index = len(time_series) - 1  # Nilai default jika tidak ditemukan
    for i in range(len(time_series) - 1, 0, -1):
        if time_series[i] < lower_bound or time_series[i] > upper_bound:
            settling_index = i
            break

    return settling_index


@jit(nopython=True)
def get_overshoot_index(time_series, setpoint):
    time_series = np.array(time_series)

    if time_series[0] > setpoint:
        time_series = -time_series
        setpoint = -setpoint

    # Cari index ketika time series pertama kali melewati setpoint
    pass_index = len(time_series)  # Nilai default jika tidak ditemukan
    for i in range(len(time_series)):
        if time_series[i] > setpoint:
            pass_index = i
            break

    # Jika time series tidak pernah melewati setpoint
    if pass_index == len(time_series):
        return pass_index

    # Cari index ketika time series mencapai nilai maksimum
    overshoot_index = np.argmax(time_series[pass_index:]) + pass_index

    return overshoot_index


@jit(nopython=True)
def get_RMSE_settle(settled_time_series, setpoint):
    time_series = np.array(settled_time_series)

    # Hitung RMSE terhadap setpoint
    RMSE = np.sqrt(np.mean((time_series - setpoint) ** 2))

    return RMSE
