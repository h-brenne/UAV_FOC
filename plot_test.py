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
                    'font.size' : 20,
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

#folder = "logs/sinusoidal_multi_amplitude/20/"
#time_str = "2022-22-11_22-11-49"

#folder = "logs/sinusoidal_multi_amplitude/40/"
#time_str = "2022-22-11_22-05-09"

folder = "logs/sinusoidal_multi_amplitude/60/"
time_str = "2022-22-11_22-13-48"
#folder = "logs/sinusoidal_torque/1/"
#time_str = "2022-22-11_19-57-08"

experiment_length = 24
sub_experiment_length = 4

with open(folder + time_str, "rb") as f:
    data = pickle.load(f)
if plot_force_sensor:
    start_time = datetime.strptime(time_str, "%Y-%d-%m_%H-%M-%S")
    data_hz = 2000
    force_df = pd.read_csv(folder + "multi_amplitude60.csv")
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

num_plots = 2
plt.subplot(num_plots,1,1)
plt.plot(timesteps, velocities, label="Velocity")
plt.plot(timesteps, velocity_setpoints, label="Velocity setpoint")
plt.ylabel("rad/s")
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
    #plt.title(title)
    plt.plot(force_timesteps, force_magnitude, label="Thrust magnitude")
    plt.plot(force_timesteps, force_df_smooth["Force X (N)"], label="Force X (N)")
    plt.plot(force_timesteps, force_df_smooth["Force Y (N)"], label="Force Y (N)")
    plt.plot(force_timesteps, force_df_smooth["Force Z (N)"], label="Force Z (N)")
    plt.ylabel("Force [N]")
    #plt.xlabel("Seconds")
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
    plt.xticks(np.arange(0, experiment_length+sub_experiment_length, sub_experiment_length))
    
    ###### Plot angles from force sensor 
    plt.figure()
    plt.subplot(2,1,1)
    #plt.title(title)
    plt.ylabel("Angle [deg]")

    plt.plot(force_timesteps, thrust_theta_angle, label="Theta")
    plt.legend()
    plt.xlim(0,experiment_length)
    plt.ylim(0,25)
    plt.subplot(2,1,2)
    plt.plot(force_timesteps, thrust_phi_angle, label="Phi")
    plt.plot(timesteps, angles, label = "Control angle")
    plt.ylim(30,180)
    plt.legend()
    plt.xlabel("Seconds")
    plt.xlim(0,experiment_length)
    plt.xticks(np.arange(0, experiment_length+sub_experiment_length, sub_experiment_length))
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
    plt.ylabel("Frequency [Hz]")
    
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    
    plt.title("$\\tilde \Omega$ amplitude steps as percentage of $\Omega$", y=1.05)
    #ax = fig.gca(projection='3d')
    #ax.plot_surface(force_stft_t[None, :], force_stft_f[:, None], np.abs(force_stft_zxx), cmap='viridis')
    _,_,_,cax = ax1.specgram(sum_of_force[int(data_hz*force_offset):int(force_offset*data_hz+experiment_length*data_hz)], NFFT=500, Fs=data_hz)
    fig.colorbar(cax).set_label('Intensity [dB]')
    plt.xticks(np.arange(0, experiment_length+sub_experiment_length, sub_experiment_length))
    plt.xlabel("Seconds")
    plt.ylabel("Frequency")
    plt.yticks(np.arange(0,720,60))
    plt.ylim(0,720)
    cell_text = [['0\%','5\%','10\%','15\%','20\%','25\%']]
    a_table = plt.table(cellText=cell_text,
                      #rowLabels=["Amplitude"],
                      loc='top',
                      cellLoc='center')
    a_table.scale(1,2)
    #ax2 = ax1.twiny()
    #ax2.plot(amplitudes, np.ones(len(amplitudes)))
    #ax2.cla()
    #plt.xlim(0,24)
    #plt.xticks([0,4,8,12,16,20], [0,5,10,15,20,25])
    plt.grid()
    
    #plt.xticks(np.arange(0, 25, 5))

## Scatter plot of velocity tracking
start_time = 17
end_time = 18
base_velocity = 60*60
amplitude = 0.2*60
angle = 0

motor_pos = np.linspace(0,2*np.pi, 20)
vel_setpoint = base_velocity + 60*amplitude*np.cos(motor_pos+angle)

fig,ax = plt.subplots(figsize=(8, 8))

#plt.title("Velocity tracking vs hub angle $\psi$")
start_index = (np.abs(timesteps - start_time)).argmin()
end_index = (np.abs(timesteps - end_time)).argmin()
#plt.plot(motor_pos*360-180, vel_setpoint)

ax.scatter(motor_angle[start_index:end_index], velocities[start_index:end_index], label="motor speed $\omega$")
ax.scatter(motor_angle[start_index:end_index], velocity_setpoints[start_index:end_index], label="speed reference $\omega_{ref}$")
ax.tick_params(axis ='y', labelcolor = 'C0') 
plt.xlabel("$\psi$ [deg]")
plt.ylabel("rad/s", color="C0")
#plt.ylim(95,160)
#plt.ylim(190,320)
plt.ylim(260,500)
plt.grid()
plt.legend(loc='upper left')
ax2 = ax.twinx()
ax2.scatter(motor_angle[start_index:end_index], accelerations[start_index:end_index], label="acceleration $\dot\omega$", color="C2")
ax2.tick_params(axis ='y', labelcolor = 'C2')
plt.ylabel("$rad/s^2$", color = "C2")
#plt.ylim(-10,10)
#plt.ylim(-40,40)
plt.ylim(-100,100)
plt.xticks(np.arange(0, 370, 45))
plt.legend(loc='lower right')


plt.figure(figsize=(8, 8))
plt.plot((timesteps-17.0)*1000, velocities, label="motor speed $\omega$", linewidth="3")
plt.plot((timesteps-17.0)*1000, velocity_setpoints, label="speed reference $\omega_{ref}$", linewidth="3")
plt.ylabel("rad/s")
plt.xlabel("milliseconds")
#plt.xlim(0,100)
#plt.xlim(0,50)
plt.xlim(0,32)
#plt.ylim(95,160)
#plt.ylim(190,320)
plt.ylim(260,500)

plt.legend(loc='upper left')
#plt.title(title)


#### Plot sequence
fig,ax = plt.subplots(figsize=(10, 5))
matplotlib.rcParams.update({'font.size': 14})
#plt.title("Velocity tracking vs hub angle $\psi$")
start_index = (np.abs(timesteps - start_time)).argmin()
end_index = (np.abs(timesteps - end_time)).argmin()
#plt.plot(motor_pos*360-180, vel_setpoint)

ax.scatter(motor_angle[start_index:end_index], velocities[start_index:end_index], label="motor speed $\omega$")
ax.scatter(motor_angle[start_index:end_index], velocity_setpoints[start_index:end_index], label="speed reference $\omega_{ref}$")
ax.tick_params(axis ='y', labelcolor = 'C0') 
plt.xlabel("$\psi$ [deg]")
plt.ylabel("rad/s", color="C0")
#plt.ylim(95,160)
#plt.ylim(190,320)
plt.ylim(250,500)
#plt.grid()
plt.legend(loc='lower left')
plt.xticks(np.arange(20, 380, 90))
ax2 = ax.twinx()
ax2.scatter(motor_angle[start_index:end_index], accelerations[start_index:end_index], label="acceleration $\dot\omega$", color="C2")
ax2.tick_params(axis ='y', labelcolor = 'C2')
plt.ylabel("$rad/s^2$", color = "C2")
#plt.ylim(-10,10)
#plt.ylim(-40,40)
plt.ylim(-130,100)
#plt.xticks(np.arange(0, 370, 45))
ax.grid(axis='x',color='r', linestyle='dotted', linewidth=3)
ax2.grid(axis="y")

ax.text(0.06, 0.97, '4', fontsize=22,transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='square', facecolor='wheat', alpha=0.5))
ax.text(0.29, 0.97, '1', fontsize=22, transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='square', facecolor='wheat', alpha=0.5))
ax.text(0.52, 0.97, '2', fontsize=22, transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='square', facecolor='wheat', alpha=0.5))
ax.text(0.74, 0.97, '3', fontsize=22, transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='square', facecolor='wheat', alpha=0.5))
plt.legend(loc='lower right')


plt.tight_layout()
if save_plot:
    pdf = matplotlib.backends.backend_pdf.PdfPages(folder + "output.pdf")
    for i in plt.get_fignums():
        if save_pgf:
            plt.figure(i).savefig(folder + str(i) + '.pdf')
        else:
            plt.figure(i).savefig(folder + str(i) + '.pdf', bbox_inches = "tight")
            #pdf.savefig(plt.figure(i))
            #pdf.close()
if not save_pgf:
    plt.show()

