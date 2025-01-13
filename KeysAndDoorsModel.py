import timeit

import numpy as np
from psyneulink import *
import KeysAndDoorsWrapper as kad

# Runtime Switches:
RENDER = False
PNL_COMPILE = False
RUN = True

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

# temp
obs_len = 7

# *********************************************************************************************************************
# **************************************  MECHANISMS AND COMPOSITION  *************************************************
# *********************************************************************************************************************

# Perceptual Mechanism
player_obs = ProcessingMechanism(input_shapes=obs_len, function=GaussianDistort, name="PLAYER OBS")
# Value and Reward Mechanisms (not yet used;  for future use)
values = TransferMechanism(input_shapes=3, name="AGENT VALUES")
reward = TransferMechanism(name="REWARD")

# Create Composition
agent_comp = Composition(name='KEYS AND DOORS COMPOSITION')
agent_comp.add_node(player_obs)

# CONTROL MECHANISM GOES HERE

# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

# Edits will have to be made once the control mechanism is created

num_trials = 4

def main():
    env = kad.KeysAndDoorsEnv("""
                              .t..
                              ....
                              ###D
                              s...
                              """)
    reward = 0
    done = False

    def my_print():
        print(ocm.net_outcome)

    print("Running simulation...")
    steps = 0
    start_time = timeit.default_timer()
    for _ in range(num_trials):
        observation = env.reset()
        while True:
            if PNL_COMPILE:
                BIN_EXECUTE = 'LLVM'
            else:
                BIN_EXECUTE = 'Python'
            dx, dy, open, pickup = agent_comp.run(inputs={player_obs:[observation]
                                                    },
                                         call_after_trial=my_print,
                                         bin_execute=BIN_EXECUTE
                                         )
            observation, reward, done, _ = env.step(dx, dy, open, pickup)
            if RENDER:
                env.render()
            print('OCM ControlSignals:')
            print('\n\tOutcome: {}\n\tPlayer OBS: {}'.
                  format(ocm._objective_mechanism.value,
                         ocm.control_signals[0].value))
            for sample, value in zip(ocm.saved_samples, ocm.saved_values):
                print('\n\t\tSample: {} Value: {}'.format(sample, value))
            print('\n\tOutcome: {}\n\tPlayer OBS: {}'.
                  format(ocm._objective_mechanism.value,
                         ocm.control_signals[0].value))
            if done:
                break
    stop_time = timeit.default_timer()
    print(f'{steps / (stop_time - start_time):.1f} steps/second, {steps} total steps in '
          f'{stop_time - start_time:.2f} seconds')

if RUN:
    if __name__ == "__main__":
        main()
