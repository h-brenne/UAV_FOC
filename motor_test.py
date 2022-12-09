import logging

import asyncio
import math
import time
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import moteus

class Motor_tester:
    def __init__(self, mode):
        qr = moteus.QueryResolution()
        # Query current
        qr.q_current = moteus.F32
        qr.d_current = moteus.F32
        self.controller = moteus.Controller(query_resolution=qr)
        self.stream = moteus.Stream(self.controller)
        self.motor_action = {
            'position': math.nan,
            'velocity': 0,
            'maximum_torque': math.nan,
            'stop_position': math.nan,
            'accel_limit': math.nan,
            'feedforward_torque': None,
            'sinusoidal': 0
        }
        self.motor_position = 0
        self.motor_velocity = 0
        if mode == "velocity" or mode == "torque" or mode == "voltage": 
            self.mode = mode
        else:
            raise Exception("Invalid mode")

        #Custom PI
        self.Kp = 0.017
        self.Ki = 0.00005
        self.vel_cumulative_error = 0
        
        
    async def init_driver(self):
        # In case the controller had faulted previously, at the start of
        # this script we send the stop command in order to clear it.
        await self.controller.set_stop()
        if self.mode == "velocity":
            await self.stream.command(b'conf set servo.voltage_mode_control 0.00')
            await self.stream.command(b'conf set servo.pid_position.kp 0.02')
            await self.stream.command(b'conf set servo.pid_position.ki 0.00')
            await self.stream.command(b'conf set servo.pid_position.kd 0.05')
            await self.stream.command(b'conf set servo.max_position_slip 20.0')
        if self.mode == "torque":
            await self.stream.command(b'conf set servo.voltage_mode_control 0.00')
            await self.stream.command(b'conf set servo.pid_position.kp 0.00')
            await self.stream.command(b'conf set servo.pid_position.ki 0.00')
            await self.stream.command(b'conf set servo.pid_position.kd 0.00')
        if self.mode == "voltage":
            await self.stream.command(b'conf set servo.voltage_mode_control 1.00')
            await self.stream.command(b'conf set servo.pid_position.kp 0.02')
            await self.stream.command(b'conf set servo.pid_position.ki 0.00')
            await self.stream.command(b'conf set servo.pid_position.kd 0.05')
            await self.stream.command(b'conf set servo.max_position_slip 20.0')

    async def command_motor(self):
        if self.mode == "velocity":
            velocity = self.motor_action["velocity"] + self.motor_action["sinusoidal"]
            ff = 0
        if self.mode == "torque":
            velcocity_error = self.motor_action["velocity"] - self.motor_velocity
            self.cumulative_error = self.cumulative_error + velcocity_error
            ff = self.Kp*velcocity_error +self.Ki*self.cumulative_error + self.motor_action["sinusoidal"]
            velocity = 0
        if self.mode == "voltage":
            velocity = self.motor_action["velocity"]
            ff = self.motor_action["sinusoidal"]
        motor_telemetry = await self.controller.set_position(
            position=self.motor_action["position"],
            velocity=velocity,
            feedforward_torque=ff,
            maximum_torque=self.motor_action["maximum_torque"],
            stop_position=self.motor_action["stop_position"],
            accel_limit=self.motor_action["accel_limit"],
            query=True)
        self.motor_position = (motor_telemetry.values[moteus.Register.POSITION] % 1)*2*math.pi 
        self.motor_velocity = motor_telemetry.values[moteus.Register.VELOCITY]
        return motor_telemetry
    async def stop_motor_driver(self):
        await self.controller.set_stop()

    async def soft_start(self):
        data = []
        self.motor_action["accel_limit"] = 10
        self.motor_action["velocity"] = 10
        t0 = time.time()
        t = 0
        motor_telem = await self.command_motor()
        while t<1:
            t = time.time() - t0
            data.append([motor_telem, motor.motor_action["velocity"], t])
            motor_telem = await self.command_motor()
        self.motor_action["accel_limit"] = math.nan
        return data

async def constant_velocity_test(motor):
    timestr = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    test_name = "logs/constant_velocity_hinged_prop_" + timestr
    data = []
    await motor.init_driver()
    await motor.soft_start()

    motor.motor_action["sinusoidal"] = 0
    motor.motor_action["velocity"] = 20
    t0 = time.time()
    t = 0
    motor_telem = await motor.command_motor()
    while t<3:
        t = time.time() - t0
        
        data.append([motor_telem, motor.motor_action["velocity"], t])
        motor_telem = await motor.command_motor()

    with open(test_name, 'wb') as f:
        pickle.dump(data, f)

async def constant_velocities_test(motor):
    timestr = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    test_name = "logs/constant_velocities/200Hz_bw" + timestr
    data = []
    await motor.init_driver()
    await motor.soft_start()
    motor.motor_action["sinusoidal"] = 0

    t0 = time.time()
    t = 0
    motor_telem = await motor.command_motor()
    while t<10:
        t = time.time() - t0
        if t > 2:
            motor.motor_action["velocity"] = 20
        if t > 4:
            motor.motor_action["velocity"] = 40
        if t > 6:
            motor.motor_action["velocity"] = 60
        if t > 8:
            motor.motor_action["velocity"] = 80
       
        data.append([motor_telem, motor.motor_action["velocity"], motor.motor_action["sinusoidal"], t])
        motor_telem = await motor.command_motor()
    with open(test_name, 'wb') as f:
        pickle.dump(data, f)
async def debug(motor):
    await motor.init_driver()
    while(True):
        await motor.command_motor()
        print(motor.motor_position)
        print(motor.motor_velocity)
async def sinusoidal_test(motor):
    timestr = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    test_name = "logs/sinusoidal_test_hinged_prop_" + timestr
    await motor.init_driver()
    await motor.soft_start()
    
    data = []

    base_frequency = 25
    amplitude = 0.3*base_frequency
    angle = 0

    t0 = time.time()
    t = 0
    motor_telem = await motor.command_motor()
    while t<4:
        t = time.time() - t0
        print("Motor angle: ", motor.motor_position*180/math.pi)
        print("Amplitude: ", math.cos(motor.motor_position+angle))
        print("")
        sine_wave = amplitude*math.cos(motor.motor_position+angle)
        motor.motor_action["velocity"] = base_frequency
        motor.motor_action["sinusoidal"] = sine_wave
    
        data.append([motor_telem, motor.motor_action["velocity"], motor.motor_action["sinusoidal"], t, angle, motor.motor_position])
        motor_telem = await motor.command_motor()
    with open(test_name, 'wb') as f:
        pickle.dump(data, f)

async def sinusoidal_multi_test_amplitude(motor):
    timestr = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    test_name = "logs/sinusoidal_multi_amplitude/1/" + timestr
    await motor.init_driver()
    await motor.soft_start()
    
    data = []

    base_frequency = 40
    amplitude = 0
    angle = math.pi/2

    t0 = time.time()
    t = 0
    motor_telem = await motor.command_motor()
    while t<24:
        t = time.time() - t0
        if t > 4:
            amplitude = 0.05*base_frequency
        if t > 8:
            amplitude = 0.1*base_frequency
        if t > 12:
            amplitude = 0.15*base_frequency
        if t > 16:
            amplitude = 0.2*base_frequency
        if t > 20:
            amplitude = 0.25*base_frequency

        sine_wave = amplitude*math.cos(motor.motor_position+angle)
        motor.motor_action["velocity"] = base_frequency
        motor.motor_action["sinusoidal"] = sine_wave
 
        data.append([motor_telem, motor.motor_action["velocity"], motor.motor_action["sinusoidal"], t, angle])
        motor_telem = await motor.command_motor()
    with open(test_name, 'wb') as f:
        pickle.dump(data, f)

async def sinusoidal_multi_test_angle(motor):
    timestr = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    test_name = "logs/sinusoidal_multi_angle/" + timestr
    await motor.init_driver()
    await motor.soft_start()
    
    data = []

    base_frequency = 30
    amplitude = 0
    angle = 0

    t0 = time.time()
    t = 0
    motor_telem = await motor.command_motor()
    while t<12:
        t = time.time() - t0
        if t > 2:
            amplitude = 0.1*base_frequency
            angle = 0
        if t > 4:
            angle = math.pi/2
        if t > 6:
            angle = math.pi
        if t > 8:
            angle = 3*math.pi/2
        if t > 10:
            amplitude = 0
            angle = 0

        sine_wave = amplitude*math.cos(motor.motor_position+angle)
        motor.motor_action["velocity"] = base_frequency
        motor.motor_action["sinusoidal"] = sine_wave
 
        data.append([motor_telem, motor.motor_action["velocity"], motor.motor_action["sinusoidal"], t, angle])
        motor_telem = await motor.command_motor()
    with open(test_name, 'wb') as f:
        pickle.dump(data, f)

async def sinusoidal_test_angle_sweep(motor):
    timestr = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    test_name = "logs/angle_sweep/" + timestr
    await motor.init_driver()
    await motor.soft_start()
    
    data = []

    spin_start = 1 #Spins/s
    spin_end = 7
    base_frequency = 35
    amplitude = 0.25*base_frequency
    angle = 0

    t0 = time.time()
    t = 0
    motor_telem = await motor.command_motor()
    while t<5:
        t = time.time() - t0
        spin_velocity = spin_start + (spin_end-spin_start)*(t/5)
        angle = (t/5)*2*math.pi*spin_velocity
        sine_wave = amplitude*math.cos(motor.motor_position+angle)
        motor.motor_action["velocity"] = base_frequency
        motor.motor_action["sinusoidal"] = sine_wave
 
        data.append([motor_telem, motor.motor_action["velocity"], motor.motor_action["sinusoidal"], t, angle])
        motor_telem = await motor.command_motor()
    with open(test_name, 'wb') as f:
        pickle.dump(data, f)
async def sinusoidal_test_velocity_sweep(motor):
    timestr = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    test_name = "logs/velocity_sweep/" + timestr
    await motor.init_driver()
    await motor.soft_start()
    
    data = []

    start_frequency = 10
    end_frequency = 60
    angle = math.pi/2
    
    t0 = time.time()
    t = 0
    end_time = 10
    motor_telem = await motor.command_motor()
    while t<end_time:
        t = time.time() - t0
        frequency = start_frequency + (end_frequency-start_frequency)*(t/end_time)
        amplitude = 0.2*frequency
        sine_wave = amplitude*math.cos(motor.motor_position+angle)
        motor.motor_action["velocity"] = frequency
        motor.motor_action["sinusoidal"] = sine_wave
 
        data.append([motor_telem, motor.motor_action["velocity"], motor.motor_action["sinusoidal"], t, angle])
        motor_telem = await motor.command_motor()
    with open(test_name, 'wb') as f:
        pickle.dump(data, f)
async def sinusoidal_torque_test(motor):
    timestr = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    test_name = "logs/sinusoidal_torque/" + timestr
    await motor.init_driver()
    #start_data = await motor.soft_start()
    #velocity = start_data[-1][0].values[moteus.Register.VELOCITY]
    motor_telem = None
    velocity = 0
    data = []

    base_frequency = 30
    amplitude = 0.05 #Nm
    angle = math.pi/2

    #Simple PI
    Kp = 0.017
    Ki = 0.00005
    cumulative_error = 0

    t0 = time.time()
    t = 0
    motor_telem = await motor.command_motor()
    while t<2:
        t = time.time() - t0

        setpoint_velocity = t*10
        if not motor_telem:
            velocity = 0
        else:
            velocity = motor_telem.values[moteus.Register.VELOCITY]
        velcocity_error = setpoint_velocity - velocity
        cumulative_error = cumulative_error + velcocity_error
        ff_torque = Kp*velcocity_error +Ki*cumulative_error
        motor.motor_action["feedforward_torque"] = ff_torque
        motor_telem = await motor.command_motor()
    t0 = time.time()
    t = 0
    while t<4:
        t = time.time() - t0

        sine_wave = amplitude*math.cos(motor.motor_position+angle)
        #motor.motor_action["velocity"] = base_frequency
        if data:
            velocity = data[-1][0].values[moteus.Register.VELOCITY]

        velcocity_error = base_frequency - velocity
        cumulative_error = cumulative_error + velcocity_error

        
        ff_torque = Kp*velcocity_error +Ki*cumulative_error + sine_wave
        
        motor.motor_action["feedforward_torque"] = ff_torque
        motor_telem = await motor.command_motor()
 
        data.append([motor_telem, base_frequency, t, angle])
    with open(test_name, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    motor = Motor_tester("velocity")
   
    
    task = loop.create_task(sinusoidal_test(motor))
    try: 
        loop.run_until_complete(task)
    finally:
        task.cancel()
        loop.run_until_complete(motor.stop_motor_driver())