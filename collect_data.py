import numpy as np
import scipy.io
from OAgent_sim import OAgentSim

oAgent = OAgentSim()
dict_launch = {}

pi = 3.141592653589793
RAD = 180 / pi

for launch_time in range(int(1e3)):
    oAgent.modify()
    print("========", launch_time + 1, "========")
    step = []
    runtime = np.array([])
    done = False
    while done is False:
        done = oAgent.step(action=0, h=0.01)
        v, theta, r, q, x, y, t = oAgent.collect()
        runtime = np.append(runtime, t)
        step.append([v / 315., theta / -0.6, x / -9.3e3, y / 1.3e4])  # r / 1.6e4, -q
        # step.append([(v - 200) / 400, theta, r / 30000, q])
    time = t * np.ones([runtime.shape[0]]) - runtime
    dict_launch['DNN_input{0}'.format(launch_time)] = step
    dict_launch['DNN_output{0}'.format(launch_time)] = time

flight_data = {'flight_data': dict_launch}
scipy.io.savemat('./flight_data.mat', flight_data)
