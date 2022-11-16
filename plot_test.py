import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import moteus



plot_angles = False
plot_force_sensor = False
plot_imu = False


with open("logs/sinusoidal_test_hinged_prop_2022-17-11_00-07-06", "rb") as f:
    data = pickle.load(f)
if plot_force_sensor:
    data_hz = 1000
    df = pd.read_csv("logs/MN5006/800hz_force/3/sinusoidal_multitest_hinged_prop_7.csv")
    df = df.ewm(span = 10).mean()
    num_samples = len(df["Torque Z (N-m)"])
    force_timesteps = np.linspace(0,num_samples/data_hz, num_samples)

    thrust_angle = 180+(180/math.pi)*np.arctan2(df["Force Y (N)"],df["Force X (N)"])

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
plt.legend()
plt.title("MN5006 450kV, hinged propeller, 40Hz, 20% Amplitude")

plt.subplot(num_plots,1,2)
plt.plot(timesteps, torques, label="Torque Nm")
plt.legend()
#plt.xlim(0,12)

plt.subplot(num_plots,1,3)
#plt.plot(timesteps, currents, label="Total current A")
#plt.plot(timesteps, temperatures, label="Controller Temperature C")
if plot_force_sensor:
    plt.plot(force_timesteps, thrust_angle, label="atan2(forceY,forceX)")
if plot_angles:
    plt.plot(timesteps, angles, label = "Control angle")
plt.legend()
#plt.xlim(0,12)

if plot_force_sensor:
    plt.subplot(num_plots,1,4)
    plt.plot(force_timesteps, df["Force X (N)"], label="Force X (N)")
    plt.plot(force_timesteps, df["Force Y (N)"], label="Force Y (N)")
    plt.plot(force_timesteps, df["Force Z (N)"], label="Force Z (N)")
    plt.legend()
    #plt.xlim(0,12)
    plt.subplot(num_plots,1,5)
    plt.plot(force_timesteps, df["Torque X (N-m)"], label="Torque X (N-m)")
    plt.plot(force_timesteps, df["Torque Y (N-m)"], label="Torque Y (N-m)")
    plt.plot(force_timesteps, df["Torque Z (N-m)"], label="Torque Z (N-m)")
    plt.xlabel("Seconds")
    plt.legend()
    #plt.xlim(0,12)




plt.show()