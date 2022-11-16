import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import moteus



plot_angles = True
plot_force_sensor = True
plot_imu = False


with open("logs/sinusoidal_multi_test_hinged_prop2022-16-11_20-12-22", "rb") as f:
    data = pickle.load(f)
if plot_force_sensor:
    data_hz = 10000
    df = pd.read_csv("logs/multilog_test/sinusoidal_multitest_hinged_prop_3.csv")
    df = df.ewm(span = 5000).mean()
    num_samples = len(df["Torque Z (N-m)"])
    force_timesteps = np.linspace(0,num_samples/data_hz, num_samples)

    thrust_angle = (180/math.pi)*np.arccos((df["Force Y (N)"])/(np.sqrt(df["Force X (N)"]**2+df["Force Y (N)"]**2+df["Force Z (N)"]**2)))

timesteps = np.asarray([x[2] for x in data])
print("Mean update rate: ", 1/np.diff(timesteps).mean())

velocities = [60*x[0].values[moteus.Register.VELOCITY] for x in data]
torques = [x[0].values[moteus.Register.TORQUE] for x in data]
#positions = [(x[0].values[moteus.Register.POSITION]%1) for x in data]
velocity_setpoints = [60*x[1] for x in data]
currents = [x[0].values[moteus.Register.Q_CURRENT] + x[0].values[moteus.Register.D_CURRENT] for x in data]
temperatures = [x[0].values[moteus.Register.TEMPERATURE] for x in data]
if plot_angles:
    angles = [x[3]*180/math.pi for x in data]

num_plots = 3
if plot_force_sensor:
    num_plots = 5
plt.subplot(num_plots,1,1)
plt.plot(timesteps, velocities, label="Velocity")
plt.plot(timesteps, velocity_setpoints, label="Velocity setpoint")
plt.ylabel("RPM")
#plt.xlim(0,0.6)
plt.legend()
plt.title("MN5006 450kV, hinged propeller, 20Hz, 35% Amplitude")

plt.subplot(num_plots,1,2)
if plot_force_sensor:
    plt.plot(force_timesteps, df["Torque Z (N-m)"])
plt.plot(timesteps, torques, label="Torque Nm")

#plt.plot(timesteps, positions, label="Position")
plt.legend()
#plt.xlim(0,0.6)

plt.subplot(num_plots,1,3)
#plt.plot(timesteps, currents, label="Total current A")
plt.plot(timesteps, temperatures, label="Controller Temperature C")
if plot_angles:
    plt.plot(timesteps, angles, label = "Control angle")
if plot_force_sensor:
    plt.plot(force_timesteps, thrust_angle, label="Force sensor measured angle")

if plot_force_sensor:
    plt.subplot(num_plots,1,4)
    plt.plot(force_timesteps, df["Force X (N)"], label="Force X (N)")
    plt.plot(force_timesteps, df["Force Y (N)"], label="Force Y (N)")
    plt.plot(force_timesteps, df["Force Z (N)"], label="Force Z (N)")
    plt.xlabel("Seconds")
    plt.legend()
    plt.subplot(num_plots,1,5)
    plt.plot(force_timesteps, df["Torque X (N-m)"], label="Torque X (N-m)")
    plt.plot(force_timesteps, df["Torque Y (N-m)"], label="Torque Y (N-m)")
    plt.plot(force_timesteps, df["Torque Z (N-m)"], label="Torque Z (N-m)")
    plt.xlabel("Seconds")
    plt.legend()




plt.show()