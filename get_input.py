import numpy as np

def get_amplitudes(timesteps):
    amplitudes = np.zeros(timesteps.shape)
    start_index = 0
    for i in range(6): 
        end_time = 4*(i+1)
        end_index = (np.abs(timesteps - end_time)).argmin()
        amplitudes[start_index:end_index] = 0.05*i
        start_index = end_index
    return amplitudes