import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import moteus
import scipy.fftpack
from scipy import signal


plot_angles = True
plot_force_sensor = True
plot_imu = True
plot_fft = True
plot_stft = True


with open("logs/MN5006/200Hz_force/200hz2022-17-11_21-12-05", "rb") as f:
    data = pickle.load(f)
if plot_force_sensor:
    data_hz = 10000
    force_df = pd.read_csv("logs/MN5006/200Hz_force/mini45_200hz_bw3.csv")
    force_df_smooth = force_df.ewm(span = 1500).mean()
    num_samples = len(force_df["Torque Z (N-m)"])
    force_timesteps = np.linspace(-5.7,num_samples/data_hz-5.7, num_samples)

    thrust_phi_angle = 180+(180/math.pi)*np.arctan2(force_df_smooth["Force Y (N)"],force_df_smooth["Force X (N)"])
    thrust_theta_angle = (180/math.pi)*np.arctan2(np.sqrt(force_df_smooth["Force X (N)"]**2+force_df_smooth["Force Y (N)"]**2), force_df_smooth["Force Z (N)"])
    
    if plot_fft:
        #FFT analysis
        force_fft = scipy.fftpack.rfft(force_df["Force X (N)"].to_numpy())/num_samples
        #force_fft = force_fft[len(force_fft)//2:]
        #force_freq_timesteps = np.linspace(0, 0.5*data_hz, num_samples//2)
        force_freq_timesteps = scipy.fftpack.rfftfreq(num_samples, 1/data_hz)

if plot_imu:
    df_imu = pd.read_csv("logs/MN5006/200Hz_force/FA_01_31_05.csv")
    imu_data_hz = 800
    imu_timesteps = df_imu["TimeStartup"]
    num_imu_samples = len(imu_timesteps)
    imu_timesteps = (imu_timesteps - imu_timesteps[0])*10e-10
    if plot_fft:
        #FFT analysis. Real signal will produce symmetric FT along x-axis, plot positve part
        imu_freq_timesteps = np.linspace(0, 0.5*imu_data_hz, num_imu_samples//2)
        imu_fft = scipy.fftpack.rfft(df_imu["GyroX"].to_numpy())/num_imu_samples
        #imu_fft = imu_fft[len(imu_fft)//2:]
        imu_freq_timesteps = scipy.fftpack.rfftfreq(num_imu_samples, 1/imu_data_hz)
    if plot_stft:
        imu_stft_f, imu_stft_t, imu_stft_zxx = scipy.signal.stft(df_imu["GyroX"].to_numpy(), imu_data_hz, nperseg=5000)

timesteps = np.asarray([x[2] for x in data])
print(timesteps)
print("Mean update rate: ", 1/np.diff(timesteps).mean())

velocities = [60*x[0].values[moteus.Register.VELOCITY] for x in data]
torques = [x[0].values[moteus.Register.TORQUE] for x in data]
#positions = [(x[0].values[moteus.Register.POSITION]%1) for x in data]
velocity_setpoints = [60*x[1] for x in data]
currents = [x[0].values[moteus.Register.Q_CURRENT] + x[0].values[moteus.Register.D_CURRENT] for x in data]
temperatures = [x[0].values[moteus.Register.TEMPERATURE] for x in data]
if plot_angles:
    angles = [x[3]*180/math.pi for x in data]

num_plots = 2
if plot_angles:
    num_plots = 3
plt.subplot(num_plots,1,1)
plt.plot(timesteps, velocities, label="Velocity")
plt.plot(timesteps, velocity_setpoints, label="Velocity setpoint")
plt.ylabel("RPM")
plt.legend()
plt.title("MN5006 450kV, hinged propeller, 200Hz torque bandwidth")

plt.subplot(num_plots,1,2)
plt.plot(timesteps, torques, label="Motor Driver Torque Nm")
plt.legend()
#plt.xlim(0,12)
if plot_angles: 
    if plot_force_sensor:
        plt.subplot(num_plots,1,3)
        plt.plot(force_timesteps, thrust_phi_angle, label="atan2(forceY,forceX)")

        plt.plot(timesteps, angles, label = "Control angle")
    plt.legend(prop={'size': 6})
    plt.xlim(0,12)

if plot_force_sensor:
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(force_timesteps, force_df_smooth["Force X (N)"], label="Force X (N)")
    plt.plot(force_timesteps, force_df_smooth["Force Y (N)"], label="Force Y (N)")
    plt.plot(force_timesteps, force_df_smooth["Force Z (N)"], label="Force Z (N)")
    plt.legend(prop={'size': 6})
    #plt.xlim(-1,13)
    plt.subplot(2,1,2)
    plt.plot(force_timesteps, force_df_smooth["Torque X (N-m)"], label="Torque X (N-m)")
    plt.plot(force_timesteps, force_df_smooth["Torque Y (N-m)"], label="Torque Y (N-m)")
    plt.plot(force_timesteps, force_df_smooth["Torque Z (N-m)"], label="Torque Z (N-m)")
    plt.xlabel("Seconds")
    plt.legend(prop={'size': 6})
    #plt.xlim(-1,13)

    plt.figure()
    plt.plot(force_timesteps, thrust_theta_angle)
if plot_imu or plot_force_sensor:
    plt.figure()
    plt.subplot(2,1,1)
    if plot_imu:
        plt.plot(imu_freq_timesteps, np.abs(imu_fft))
    if plot_force_sensor:
        plt.subplot(2,1,2)
        plt.plot(force_freq_timesteps , np.abs(force_fft))
if plot_stft:
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #plt.pcolormesh(imu_stft_t, imu_stft_f, np.abs(imu_stft_zxx))
    #ax.plot_surface(imu_stft_t[None, :], imu_stft_f[:, None], 20*np.log10(np.abs(imu_stft_zxx)), cmap='viridis')
    plt.specgram(df_imu["GyroX"].to_numpy(), NFFT=5000, Fs=imu_data_hz)
plt.show()