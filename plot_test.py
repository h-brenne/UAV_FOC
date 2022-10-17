import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import moteus


with open("normal_prop_30rpm", "rb") as f:
    data = pickle.load(f)


timesteps = np.asarray([x[2] for x in data])
print("Mean update rate: ", 1/np.diff(timesteps).mean())

velocities = [60*x[0].values[moteus.Register.VELOCITY] for x in data]
torques = [x[0].values[moteus.Register.TORQUE] for x in data]
positions = [(x[0].values[moteus.Register.POSITION]%1) for x in data]
velocity_setpoints = [60*x[1] for x in data]
currents = [x[0].values[moteus.Register.Q_CURRENT] + x[0].values[moteus.Register.D_CURRENT] for x in data]
temperatures = [x[0].values[moteus.Register.TEMPERATURE] for x in data]

plt.subplot(3,1,1)
plt.plot(timesteps, velocities, label="Velocity")
plt.plot(timesteps, velocity_setpoints, label="Velocity setpoint")
plt.ylabel("RPM")
#plt.xlim(0,0.4)
plt.legend()
plt.title("MN5006 450kV, no propeller, 20Hz 97.5% amplitude ")

plt.subplot(3,1,2)
plt.plot(timesteps, torques, label="Torque Nm")
plt.plot(timesteps, positions, label="Position")
plt.legend()
#plt.xlim(0,0.4)

plt.subplot(3,1,3)
plt.plot(timesteps, currents, label="Total current A")
plt.plot(timesteps, temperatures, label="Temperature C")

plt.legend()


plt.show()