import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import moteus
import scipy.fftpack
from scipy import signal
from datetime import datetime
import matplotlib.backends.backend_pdf

plt.rcParams["figure.figsize"] = (10,5)

save_plot = False
plot_angles = True
plot_force_sensor = False
plot_imu = False
plot_fft = False
plot_stft = False

title = "Velocity sweep for 60rpm to 3600 rpm, 20% sinusoidal amplitude"
folder = "logs/sinusoidal_multi_amplitude/40/"
time_str = "2022-22-11_22-05-09"

experiment_length = 10

with open(folder + time_str, "rb") as f:
    data = pickle.load(f)
if plot_force_sensor:
    start_time = datetime.strptime(time_str, "%Y-%d-%m_%H-%M-%S")
    data_hz = 2000
    force_df = pd.read_csv(folder + "multi_amplitude40.csv")
    force_start_time = datetime.strptime(
        force_df.columns[-1][-19:], "%d/%m/%Y %H:%M:%S")
    force_offset = (start_time-force_start_time).total_seconds() + 1.1
    force_df_smooth = force_df.ewm(span = 1500).mean()
    num_samples = len(force_df["Torque Z (N-m)"])
    force_timesteps = np.linspace(-force_offset,num_samples/data_hz-force_offset, num_samples)

    thrust_phi_angle = 180+(180/math.pi)*np.arctan2(force_df_smooth["Force Y (N)"],force_df_smooth["Force X (N)"])
    thrust_theta_angle = (180/math.pi)*np.arctan2(np.sqrt(force_df_smooth["Force X (N)"]**2+force_df_smooth["Force Y (N)"]**2), force_df_smooth["Force Z (N)"])
    
    sum_of_force = force_df["Force X (N)"].to_numpy()+force_df["Force Y (N)"].to_numpy()+force_df["Force Z (N)"].to_numpy()
    sum_of_torque = force_df["Torque X (N-m)"].to_numpy() + force_df["Torque Y (N-m)"].to_numpy()+ force_df["Torque Z (N-m)"].to_numpy()
    if plot_fft:
        #FFT analysis
        force_fft = scipy.fftpack.rfft(sum_of_force)/num_samples
        #force_fft = force_fft[len(force_fft)//2:]
        #force_freq_timesteps = np.linspace(0, 0.5*data_hz, num_samples//2)
        force_freq_timesteps = scipy.fftpack.rfftfreq(num_samples, 1/data_hz)
    if plot_stft:
        force_stft_f, force_stft_t, force_stft_zxx = scipy.signal.stft(force_df["Force X (N)"].to_numpy(), data_hz, nperseg=500)

if plot_imu:
    df_imu = pd.read_csv(folder + "FA_01_31_05.csv")
    imu_data_hz = 800
    imu_timesteps = df_imu["TimeStartup"]
    num_imu_samples = len(imu_timesteps)
    imu_timesteps = (imu_timesteps - imu_timesteps[0])*10e-10
    sum_of_gyro = df_imu["GyroX"].to_numpy()+df_imu["GyroY"].to_numpy()+df_imu["GyroZ"].to_numpy()
    sum_of_imu_accel = df_imu["AccelX"].to_numpy()+df_imu["AccelY"].to_numpy()+df_imu["AccelZ"].to_numpy()
    if plot_fft:
        #FFT analysis. Real signal will produce symmetric FT along x-axis, plot positve part
        imu_freq_timesteps = np.linspace(0, 0.5*imu_data_hz, num_imu_samples//2)
        imu_fft = scipy.fftpack.rfft(sum_of_imu_accel)/num_imu_samples
        imu_freq_timesteps = scipy.fftpack.rfftfreq(num_imu_samples, 1/imu_data_hz)
    if plot_stft:
        imu_stft_f, imu_stft_t, imu_stft_zxx = scipy.signal.stft(df_imu["GyroX"].to_numpy(), imu_data_hz, nperseg=1600)

timesteps = np.asarray([x[2] for x in data])
print("Mean update rate: ", 1/np.diff(timesteps).mean())

velocities = [60*x[0].values[moteus.Register.VELOCITY] for x in data]
torques = [x[0].values[moteus.Register.TORQUE] for x in data]
motor_angle = [(x[0].values[moteus.Register.POSITION]%1)*360 for x in data]
velocity_setpoints = [60*x[1] for x in data]
currents = [x[0].values[moteus.Register.Q_CURRENT] + x[0].values[moteus.Register.D_CURRENT] for x in data]
temperatures = [x[0].values[moteus.Register.TEMPERATURE] for x in data]
if plot_angles:
    angles = [x[3]*180/math.pi for x in data]

num_plots = 2
plt.subplot(num_plots,1,1)
plt.plot(timesteps, velocities, label="Velocity")
plt.plot(timesteps, velocity_setpoints, label="Velocity setpoint")
plt.ylabel("RPM")
plt.legend()
plt.title(title)

plt.subplot(num_plots,1,2)
plt.plot(timesteps, torques, label="Motor Driver Torque Nm")
plt.ylabel("Torque [Nm]")
plt.xlabel("Seconds")
plt.legend()
plt.xlim(0,experiment_length)
if plot_force_sensor:
    plt.figure()
    plt.subplot(2,1,1)
    plt.title(title)
    plt.plot(force_timesteps, force_df_smooth["Force X (N)"], label="Force X (N)")
    plt.plot(force_timesteps, force_df_smooth["Force Y (N)"], label="Force Y (N)")
    plt.plot(force_timesteps, force_df_smooth["Force Z (N)"], label="Force Z (N)")
    plt.ylabel("Force [N]")
    plt.xlabel("Seconds")
    plt.legend()
    plt.xlim(0,experiment_length)
    plt.subplot(2,1,2)
    plt.plot(force_timesteps, force_df_smooth["Torque X (N-m)"], label="Torque X (N-m)")
    plt.plot(force_timesteps, force_df_smooth["Torque Y (N-m)"], label="Torque Y (N-m)")
    plt.plot(force_timesteps, force_df_smooth["Torque Z (N-m)"], label="Torque Z (N-m)")
    plt.ylabel("Torque [Nm]")
    plt.xlabel("Seconds")
    plt.legend()
    plt.xlim(0,experiment_length)


    plt.figure()
    plt.subplot(2,1,1)
    plt.title(title)
    plt.ylabel("Angle [deg]")

    plt.plot(force_timesteps, thrust_theta_angle, label="Theta")
    plt.legend()
    plt.xlim(0,experiment_length)
    plt.ylim(0,35)
    plt.subplot(2,1,2)
    plt.plot(force_timesteps, thrust_phi_angle, label="Phi")
    plt.plot(timesteps, angles, label = "Control angle")
    plt.ylim(30,180)
    plt.legend()
    plt.xlabel("Seconds")
    plt.xlim(0,experiment_length)
if plot_imu or plot_force_sensor:
    plt.figure()
    plt.subplot(2,1,1)
    plt.title("FFT")
    if plot_imu:
        plt.plot(imu_freq_timesteps, np.abs(imu_fft), label="FFT of sum of IMU x,y,z accelerations")
        plt.ylabel("")
        plt.legend()
    if plot_force_sensor:
        plt.subplot(2,1,2)
        plt.plot(force_freq_timesteps , np.abs(force_fft), label="FFT of sum of x,y,z forces")
        plt.ylabel("")
    plt.xlabel("Frequency")
    plt.legend()
if plot_stft:
    fig = plt.figure()
    plt.title("Spectogram of sum of imu x,y,z accelerations")
    #ax = fig.gca(projection='3d')
    #plt.pcolormesh(imu_stft_t, imu_stft_f, np.abs(imu_stft_zxx))
    #ax.plot_surface(imu_stft_t[None, :], imu_stft_f[:, None], 20*np.log10(np.abs(imu_stft_zxx)), cmap='viridis')
    _,_,_,cax = plt.specgram(sum_of_imu_accel, NFFT=800, Fs=imu_data_hz)
    fig.colorbar(cax).set_label('Intensity [dB]')
    plt.xlabel("Seconds")
    plt.ylabel("Frequency")
    fig = plt.figure()
    plt.title("Spectogram of sum of x,y,z forces")
    #ax = fig.gca(projection='3d')
    #ax.plot_surface(force_stft_t[None, :], force_stft_f[:, None], np.abs(force_stft_zxx), cmap='viridis')
    _,_,_,cax = plt.specgram(sum_of_force[int(data_hz*force_offset):int(force_offset*data_hz+experiment_length*data_hz)], NFFT=500, Fs=data_hz)
    fig.colorbar(cax).set_label('Intensity [dB]')
    plt.xlabel("Seconds")
    plt.ylabel("Frequency")

## Scatter plot of velocity tracking
start_time = 13
end_time = 15
base_velocity = 60*60
amplitude = 0.2*60
angle = 0

motor_pos = np.linspace(0,2*np.pi, 20)
vel_setpoint = base_velocity + 60*amplitude*np.cos(motor_pos+angle)

plt.figure()
plt.title("Velocities vs hub angle psi")
start_index = (np.abs(timesteps - start_time)).argmin()
end_index = idx = (np.abs(timesteps - end_time)).argmin()
#plt.plot(motor_pos*360-180, vel_setpoint)

plt.scatter(motor_angle[start_index:end_index], velocities[start_index:end_index], label="Velocity")
plt.scatter(motor_angle[start_index:end_index], velocity_setpoints[start_index:end_index], label="Velocity setpoint")
plt.xlabel("Hub angle [deg]")
plt.ylabel("RPM")
plt.legend()
if save_plot:
    pdf = matplotlib.backends.backend_pdf.PdfPages(folder + "output.pdf")
    for i in plt.get_fignums():
        plt.figure(i).savefig(folder + str(i) + '.png')
        pdf.savefig(plt.figure(i))
    pdf.close()
plt.show()

