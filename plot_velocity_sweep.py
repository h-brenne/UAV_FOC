import pickle
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import moteus
import scipy.fftpack
from scipy import signal
from datetime import datetime
import matplotlib.backends.backend_pdf

from get_input import get_amplitudes

plt.rcParams["figure.figsize"] = (10,7)
params = {'text.usetex' : True,
                    'font.size' : 14,
                    }
plt.rcParams.update(params)
plt.rc('font', family='serif')

save_pgf = False
if save_pgf:
    matplotlib.use('pgf')


old_index = True
speed_radians = True
save_plot = True
plot_angles = True
plot_force_sensor = True
plot_imu = True
plot_fft = True
plot_stft = True

title = "Average speed $\Omega$ increase from 15Hz to 60Hz, $A = 0.2 \\times \Omega$"
folder = "logs/velocity_sweep/0.2A/"
time_str = "2022-22-11_22-54-35"

experiment_length = 8
experiment_start = 2
sub_experiment_length = 2

with open(folder + time_str, "rb") as f:
    data = pickle.load(f)
if plot_force_sensor:
    start_time = datetime.strptime(time_str, "%Y-%d-%m_%H-%M-%S")
    data_hz = 2000
    force_df = pd.read_csv(folder + "velocity_sweep0.2A.csv")
    force_df["Force X (N)"] = -force_df["Force X (N)"]
    force_df["Force Y (N)"] = -force_df["Force Y (N)"]
    force_start_time = datetime.strptime(
        force_df.columns[-1][-19:], "%d/%m/%Y %H:%M:%S")
    force_offset = (start_time-force_start_time).total_seconds() + 0.5
    force_df_smooth = force_df.ewm(span = 500).mean()
    num_samples = len(force_df["Torque Z (N-m)"])
    force_timesteps = np.linspace(-force_offset,num_samples/data_hz-force_offset, num_samples)

    thrust_phi_angle = 180-(180/math.pi)*np.arctan2(force_df_smooth["Force Y (N)"],force_df_smooth["Force X (N)"])
    thrust_theta_angle = (180/math.pi)*np.arctan2(np.sqrt(force_df_smooth["Force X (N)"]**2+force_df_smooth["Force Y (N)"]**2), force_df_smooth["Force Z (N)"])
    
    sum_of_force = force_df["Force X (N)"].to_numpy()+force_df["Force Y (N)"].to_numpy()+force_df["Force Z (N)"].to_numpy()
    sum_of_torque = force_df["Torque X (N-m)"].to_numpy() + force_df["Torque Y (N-m)"].to_numpy()+ force_df["Torque Z (N-m)"].to_numpy()
    force_magnitude = np.sqrt(force_df_smooth["Force X (N)"].to_numpy()**2 + force_df_smooth["Force Y (N)"].to_numpy()**2 + force_df_smooth["Force Z (N)"].to_numpy()**2)
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

velocities = [2*math.pi*x[0].values[moteus.Register.VELOCITY] for x in data]
torques = [x[0].values[moteus.Register.TORQUE] for x in data]
currents = [x[0].values[moteus.Register.Q_CURRENT] + x[0].values[moteus.Register.D_CURRENT] for x in data]
temperatures = [x[0].values[moteus.Register.TEMPERATURE] for x in data]
motor_angle = [(x[0].values[moteus.Register.POSITION]%1)*360 for x in data]



if old_index:
    # data[0] is shifted one index to the right
    timesteps = np.asarray([x[2] for x in data])
    timesteps = timesteps[:-1]
    velocity_setpoints = [2*math.pi*x[1] for x in data]
    velocity_setpoints.pop(0)
    velocities.pop()
    torques.pop()
    currents.pop()
    temperatures.pop()
    motor_angle.pop()
    
else:
    timesteps = np.asarray([x[3] for x in data])
    velocity_setpoints = [2*math.pi*x[1] + 60*x[2] for x in data]
if plot_angles:
    angles = [x[3]*180/math.pi for x in data]
    if old_index:
        angles.pop(0)
print("Mean update rate: ", 1/np.diff(timesteps).mean())
accelerations = np.gradient(velocities)

###For amplitude sweeps
average_vel_setpoint = np.linspace(10*2*np.pi,60*2*math.pi, len(timesteps))

timesteps = timesteps -2
force_timesteps = force_timesteps - 2

fig = plt.figure(figsize=(10, 15))
plt.subplot(5,1,1)
plt.title(title)
plt.plot(timesteps, average_vel_setpoint, label="Average speed $\Omega$", linewidth="3")
#plt.xlabel("seconds")
plt.ylabel("rad/s")
plt.xlim(experiment_start,experiment_length)
plt.xticks(np.arange(0, experiment_length+sub_experiment_length, sub_experiment_length))
plt.yticks(np.arange(50, 450, 100))
plt.legend()
plt.grid()

plt.subplot(5,1,2)
#plt.title(title)
plt.plot(force_timesteps, force_magnitude, label="Thrust magnitude $T$", linewidth = "3")
plt.plot(force_timesteps, force_df_smooth["Force X (N)"], label="Force X (N)")
plt.plot(force_timesteps, force_df_smooth["Force Y (N)"], label="Force Y (N)")
plt.plot(force_timesteps, force_df_smooth["Force Z (N)"], label="Force Z (N)")
plt.ylabel("Force [N]")
plt.xticks(np.arange(0, experiment_length+sub_experiment_length, sub_experiment_length))
#plt.xlabel("Seconds")
plt.legend()
plt.grid()
plt.xlim(0,experiment_length)
plt.subplot(5,1,3)
plt.plot(force_timesteps, force_df_smooth["Torque X (N-m)"], label="Torque X (N-m)")
plt.plot(force_timesteps, force_df_smooth["Torque Y (N-m)"], label="Torque Y (N-m)")
plt.plot(force_timesteps, force_df_smooth["Torque Z (N-m)"], label="Torque Z (N-m)")
plt.ylabel("Torque [Nm]")
#plt.xlabel("Seconds")
plt.legend()
plt.xlim(0,experiment_length)
plt.xticks(np.arange(0, experiment_length+sub_experiment_length, sub_experiment_length))
plt.grid()

###### Plot angles from force sensor 
plt.subplot(5,1,4)
#plt.title(title)
plt.ylabel("Angle [deg]")

plt.plot(force_timesteps, thrust_theta_angle, label="Thrust vector elevation $\\beta_c$")
plt.legend()
plt.xlim(0,experiment_length)
plt.ylim(0,25)
plt.yticks(np.arange(0, 30, 5))
plt.xticks(np.arange(0, experiment_length+sub_experiment_length, sub_experiment_length))
plt.grid()
plt.subplot(5,1,5)
amplitude_start_index = 0
plt.plot(force_timesteps[amplitude_start_index:], thrust_phi_angle[amplitude_start_index:], label="Thrust vector azimuth $\psi_c$")
plt.plot(timesteps, angles, label = "$\psi_c^{ref}$")
plt.ylim(60,120)
plt.yticks(np.arange(30, 120, 30))
plt.legend()
plt.ylabel("Angle [deg]")
plt.xlabel("seconds")
plt.xlim(0,experiment_length)
plt.xticks(np.arange(0, experiment_length+sub_experiment_length, sub_experiment_length))
plt.grid()

plt.tight_layout()
if save_plot:
    pdf = matplotlib.backends.backend_pdf.PdfPages(folder + "output.pdf")
    for i in plt.get_fignums():
        if save_pgf:
            plt.figure(i).savefig(folder + str(i) + '.pdf')
        else:
            plt.figure(i).savefig(folder + str(i) + '.pdf')
            #pdf.savefig(plt.figure(i))
            #pdf.close()
if not save_pgf:
    plt.show()

