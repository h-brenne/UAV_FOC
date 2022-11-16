import logging

import asyncio
import math
import time
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import moteus

class Motor_tester:
    def __init__(self):
        qr = moteus.QueryResolution()
        # Query current
        qr.q_current = moteus.F32
        qr.d_current = moteus.F32
        self.controller = moteus.Controller(query_resolution=qr)
        self.stream = moteus.Stream(self.controller)
        self.motor_action = {
            'position': math.nan,
            'velocity': 0,
            'maximum_torque': 3,
            'stop_position': math.nan,
            'accel_limit': 100
        }
        self.motor_position = 0
        
    async def init_driver(self):
        # In case the controller had faulted previously, at the start of
        # this script we send the stop command in order to clear it.
        await self.controller.set_stop()
    async def command_motor(self):
        motor_telemetry = await self.controller.set_position(
            position=self.motor_action["position"],
            velocity=self.motor_action["velocity"],
            maximum_torque=self.motor_action["maximum_torque"],
            stop_position=self.motor_action["stop_position"],
            accel_limit=self.motor_action["accel_limit"],
            query=True)
        self.motor_position = (motor_telemetry.values[moteus.Register.POSITION])*2*math.pi 
        return motor_telemetry
    async def stop_motor_driver(self):
        await self.controller.set_stop()
        #time.sleep(2)
        await self.stream.command(b'd stop')

    async def soft_start(self):
        data = []
        self.motor_action["accel_limit"] = 10
        self.motor_action["velocity"] = 10
        t0 = time.monotonic()
        t = 0
        while t<1:
            t = time.monotonic() - t0
            motor_telem = await self.command_motor()
            data.append([motor_telem, motor.motor_action["velocity"], t])
        self.motor_action["accel_limit"] = math.nan
        return data

async def constant_velocity_test(motor):
    timestr = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    test_name = "logs/constant_velocity_hinged_prop_" + timestr
    data = []
    await motor.init_driver()
    await motor.soft_start()
    
    motor.motor_action["velocity"] = 10
    t0 = time.monotonic()
    t = 0
    while t<3:
        t = time.monotonic() - t0
        motor_telem = await motor.command_motor()
        data.append([motor_telem, motor.motor_action["velocity"], t])

    with open(test_name, 'wb') as f:
        pickle.dump(data, f)

async def constant_velocities_tests(motor):
    timestr = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    test_name = "logs/soft_start_test_" + timestr
    data = []
    await motor.init_driver()
    await motor.soft_start()
    
    t0 = time.monotonic()
    t = 0
    while t<10:
        t = time.monotonic() - t0
        if t > 2:
            motor.motor_action["velocity"] = 30
        if t > 4:
            motor.motor_action["velocity"] = 50
        if t > 6:
            motor.motor_action["velocity"] = 60
        if t > 8:
            motor.motor_action["velocity"] = 80
        motor_telem = await motor.command_motor()
        data.append([motor_telem, motor.motor_action["velocity"], t])

    with open(test_name, 'wb') as f:
        pickle.dump(data, f)

async def sinusoidal_test(motor):
    timestr = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    test_name = "logs/sinusoidal_test_hinged_prop_" + timestr
    await motor.init_driver()
    await motor.soft_start()
    
    data = []

    base_frequency = 20
    amplitude = 4
    angle = 0

    t0 = time.monotonic()
    t = 0
    while t<4:
        t = time.monotonic() - t0

        sine_wave = amplitude*math.cos(motor.motor_position+angle)
        motor.motor_action["velocity"] = base_frequency + sine_wave
        motor_telem = await motor.command_motor()
 
        data.append([motor_telem, motor.motor_action["velocity"], t, angle])
    with open(test_name, 'wb') as f:
        pickle.dump(data, f)
async def sinusoidal_multi_test(motor):
    timestr = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    test_name = "logs/MN5006/800hz_force/2/sinusoudal_multi_test" + timestr
    await motor.init_driver()
    await motor.soft_start()
    
    data = []

    base_frequency = 20
    amplitude = 0
    angle = 0

    t0 = time.monotonic()
    t = 0
    while t<12:
        t = time.monotonic() - t0
        if t > 2:
            amplitude = 0.2*base_frequency
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
        motor.motor_action["velocity"] = base_frequency + sine_wave
        motor_telem = await motor.command_motor()
 
        data.append([motor_telem, motor.motor_action["velocity"], t, angle])
    with open(test_name, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    motor = Motor_tester()

    task = loop.create_task(sinusoidal_test(motor))
    try: 
        loop.run_until_complete(task)
    finally:
        task.cancel()
        loop.run_until_complete(motor.stop_motor_driver())
    