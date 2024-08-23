import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the varying PWM signal
period = 2  # Period of the PWM signal in milliseconds
voltage_high = 12  # High voltage level in volts
voltage_low = 0  # Low voltage level in volts
time_resolution = 0.001  # Time resolution in milliseconds
time_duration = 100  # Total duration of the signal

# Generate time array
time = np.arange(0, time_duration, time_resolution)

# Create a ladder function for the duty cycle
duty_cycle_steps = np.linspace(0, 1, num=10)  # Create 10 steps between 0 and 1
step_duration = time_duration / len(duty_cycle_steps)  # Duration of each step

# Create a duty cycle that changes in steps (ladder function)
duty_cycle_variation = np.zeros_like(time)
for i, duty in enumerate(duty_cycle_steps):
    duty_cycle_variation[
        int(i * step_duration / time_resolution) : int(
            (i + 1) * step_duration / time_resolution
        )
    ] = duty

duty_cycle_variation = 0.5
# Generate PWM signal with ladder duty cycle
pwm_signal = np.where(
    (time % period) < duty_cycle_variation * period, voltage_high, voltage_low
)

# Compute moving average of the PWM signal, where the current point is at the end of the window
window_size = int(
    2 / time_resolution
)  # Define the window size for moving average (100 ms window)
cumulative_sum = np.cumsum(np.insert(pwm_signal, 0, 0))  # Compute cumulative sum
moving_average_voltage = (
    cumulative_sum[window_size:] - cumulative_sum[:-window_size]
) / window_size

# Append zeros at the beginning to make the moving average array match the time array length
moving_average_voltage = np.concatenate(
    [np.zeros(window_size - 1), moving_average_voltage]
)

# Plot the PWM signal and the moving average of the voltage
plt.figure(figsize=(10, 5))

# Plot the ladder PWM signal
plt.plot(time, pwm_signal, label="PWM Signal", drawstyle="steps-post")

# Plot the moving average voltage
plt.plot(time, moving_average_voltage, "r--", label="Moving Average Voltage")

# Add labels and legend
plt.title("Ladder PWM Signal and Moving Average Voltage (Window Start Aligned)")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.ylim(-1, voltage_high + 1)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
