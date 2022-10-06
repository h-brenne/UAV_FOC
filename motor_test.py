import logging

import asyncio
import math
import moteus

async def main():
    c = moteus.Controller()

    # In case the controller had faulted previously, at the start of
    # this script we send the stop command in order to clear it.
    await c.set_stop()

    motor_action = {
            'position': math.nan,
            'velocity': 0,
            'maximum_torque': 0,
            'stop_position': math.nan
        }
    
    while True:

        motor_telem = await c.set_position(
            position=motor_action["position"],
            velocity=motor_action["velocity"],
            maximum_torque=motor_action["maximum_torque"],
            stop_position=motor_action["stop_position"],
            query=True)