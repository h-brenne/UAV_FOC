import logging

import asyncio
import math
import time
import pickle
import matplotlib.pyplot as plt

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
            'maximum_torque': 1,
            'stop_position': math.nan,
            'accel_limit': math.nan
        }

        
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
        return motor_telemetry
    async def stop_motor_driver(self):
        await self.controller.set_stop()
        #time.sleep(2)
        await self.stream.command(b'd stop')
   
async def main(motor):
    test_name = "test5"
    await motor.init_driver()
    
    data = []

    motor_position = 0
    base_frequency = 20
    amplitude = 10
    t0 = time.time()
    t = 0
    while t<1:
        t = time.time() - t0

        sine_wave = amplitude*math.sin(motor_position)
        motor.motor_action["velocity"] = base_frequency + sine_wave
        
        motor_telem = await motor.command_motor()
        
        # No way to get non-accumulated rotor position?
        motor_position = (motor_telem.values[moteus.Register.POSITION])*2*math.pi 
        
        data.append([motor_telem, motor.motor_action["velocity"], t])


        #await asyncio.sleep(0.001) #Try 1000Hz

    with open(test_name, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    motor = Motor_tester()

    task = loop.create_task(main(motor))
    try: 
        loop.run_until_complete(task)
    finally:
        task.cancel()
        loop.run_forever
        loop.run_until_complete(motor.stop_motor_driver())
    