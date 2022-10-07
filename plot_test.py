import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import moteus


with open("test4", "rb") as f:
    data = pickle.load(f)

timesteps = np.asarray([x[2] for x in data])
print("Mean update rate: ", 1/np.diff(timesteps).mean())

velocities = [60*x[0].values[moteus.Register.VELOCITY] for x in data]
torques = [x[0].values[moteus.Register.TORQUE] for x in data]
positions = [(x[0].values[moteus.Register.POSITION]%1) for x in data]
velocity_setpoints = [60*x[1] for x in data]

plt.subplot(2,1,1)
plt.plot(timesteps, velocities, label="Velocity")
plt.plot(timesteps, velocity_setpoints, label="Velocity setpoint")
plt.ylabel("RPM")
#plt.xlim(0,0.4)
plt.legend()
plt.title("MN5006 450kV, no propeller, 20Hz 97.5% amplitude ")

plt.subplot(2,1,2)
plt.plot(timesteps, torques, label="Torque Nm")
plt.plot(timesteps, positions, label="Position")
#plt.xlim(0,0.4)

plt.legend()


plt.show()