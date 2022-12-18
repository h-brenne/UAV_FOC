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
import matplotlib.animation as animation

from get_input import get_amplitudes

plt.rcParams["figure.figsize"] = (10,7)
params = {'text.usetex' : True,
                    'font.size' : 22,
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

title = "$\\tilde \Omega$ amplitude steps, $\Omega = 60 (2 \pi)$"

folder = "logs/sinusoidal_multi_amplitude/20/"
time_str = "2022-22-11_22-11-49"

#folder = "logs/sinusoidal_multi_amplitude/40/"
#time_str = "2022-22-11_22-05-09"

#folder = "logs/sinusoidal_multi_amplitude/60/"
#time_str = "2022-22-11_22-13-48"
#folder = "logs/sinusoidal_torque/1/"
#time_str = "2022-22-11_19-57-08"

experiment_length = 24
sub_experiment_length = 4

with open(folder + time_str, "rb") as f:
    data = pickle.load(f)
if plot_force_sensor:
    start_time = datetime.strptime(time_str, "%Y-%d-%m_%H-%M-%S")
    data_hz = 2000
    force_df = pd.read_csv(folder + "multi_amplitude20.csv")
    force_df["Force X (N)"] = -force_df["Force X (N)"]
    force_df["Force Y (N)"] = -force_df["Force Y (N)"]
    force_start_time = datetime.strptime(
        force_df.columns[-1][-19:], "%d/%m/%Y %H:%M:%S")
    force_offset = (start_time-force_start_time).total_seconds() + 1.1
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
    imu_timesteps = (imu_timesteps - imu_timesteps[0])*10e-10 - 9.3
    num_imu_samples = len(imu_timesteps)
    
    start_time = 16.5
    end_time = 19.5
    start_index = (np.abs(imu_timesteps - start_time)).argmin()
    end_index = (np.abs(imu_timesteps - end_time)).argmin()
    num_imu_samples_section = len(imu_timesteps[start_index:end_index])

    sum_of_gyro = df_imu["GyroX"].to_numpy()+df_imu["GyroY"].to_numpy()+df_imu["GyroZ"].to_numpy()
    sum_of_imu_accel = df_imu["AccelX"].to_numpy()+df_imu["AccelY"].to_numpy()+df_imu["AccelZ"].to_numpy()
    sum_of_imu_accel_section = df_imu["AccelX"][start_index:end_index].to_numpy()+df_imu["AccelY"][start_index:end_index].to_numpy()+df_imu["AccelZ"][start_index:end_index].to_numpy()
    if plot_fft:
        #FFT analysis. Real signal will produce symmetric FT along x-axis, plot positve part
        imu_fft = scipy.fftpack.rfft(sum_of_imu_accel_section)/num_imu_samples_section
        imu_freq_timesteps = scipy.fftpack.rfftfreq(num_imu_samples_section, 1/imu_data_hz)
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
    angles = [180+x[3]*180/math.pi for x in data]
    if old_index:
        angles.pop(0)
print("Mean update rate: ", 1/np.diff(timesteps).mean())
accelerations = np.gradient(velocities)

###For amplitude sweeps
amplitudes = get_amplitudes(timesteps)

#setup figure
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1)
fig.patch.set_alpha(0.)
#data_per_frame = 80
#data_per_frame = 16
data_per_frame = 1

start_index_force = (np.abs(force_timesteps)).argmin()
start_index_velocity_slow = (np.abs(timesteps)-20).argmin()
def anim_func(n):
    ax.clear()
    plt.plot(force_timesteps[start_index_force:start_index_force + n*data_per_frame], force_magnitude[start_index_force:start_index_force +n*data_per_frame], label="Thrust magnitude (N)")
    #ax.set_ylim([0,2.5])
    ax.set_ylabel("Force [N]")
    ax.get_xaxis().set_visible(False)
    plt.legend(loc='lower right')
    plt.grid()

def anim_velocity(n):
    ax.clear()
    plt.plot(timesteps[0:n*data_per_frame], velocities[0:n*data_per_frame], label="Velocity")
    plt.plot(timesteps[0:n*data_per_frame], velocity_setpoints[0:n*data_per_frame], label="Velocity setpoint")
    ax.set_ylabel("rad/s")
    ax.set_ylim([85,160])
    ax.set_xlim(-0.5+n*data_per_frame/416,0.5+n*data_per_frame/422)
    #ax.get_xaxis().set_visible(False)
    plt.legend()
    plt.grid()
def anim_angle(n):
    ax.clear()
    plt.plot(force_timesteps[start_index_force:start_index_force + n*data_per_frame], 
        thrust_theta_angle[start_index_force:start_index_force +n*data_per_frame], label="Thrust vector elevation angle")
    ax.set_ylabel("degrees")
    #ax.set_xlim(n/500,100*n/500)
    ax.set_ylim([0,25])
    #ax.get_xaxis().set_visible(False)
    plt.legend(loc='lower left')
    plt.grid()
def anim_velocity_slow(n):
    ax.clear()
    plt.plot(timesteps, velocities, label="Velocity")
    plt.plot(timesteps, velocity_setpoints, label="Velocity setpoint")
    plt.vlines(20 + n/800, 0, 1000, color='red')
    ax.set_ylabel("rad/s")
    ax.set_ylim([85,160])
    ax.set_xlim(20-0.05+n/800,20+0.05+n/800)
    ax.get_xaxis().set_visible(False)
    plt.legend(loc='lower left')
    plt.grid()
#ani = animation.FuncAnimation(fig, anim_velocity_slow, frames=len(timesteps[start_index_velocity_slow:])//data_per_frame)
ani = animation.FuncAnimation(fig, anim_velocity_slow, frames=200)
#ani = animation.FuncAnimation(fig, anim_angle, frames=len(force_timesteps)//data_per_frame)

#plt.tight_layout()
#plt.show()
writervideo = animation.FFMpegWriter(fps=25)
ani.save('velocity_slow.mp4', writer=writervideo)
#ani.save(
#    "test_anim.mp4",
#    fps = 25,
#    #codec="png",
#    savefig_kwargs={"transparent": True, "facecolor": "none"},
#)