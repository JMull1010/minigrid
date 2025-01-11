class KeysAndDoorsEnv:
    def __init__(
        self,
        coherence=.95,
        discount_rate=.95,
        step_cost=-1,
        target_reward=100,
        grid=str
    ):

        import KeysAndDoors as KAD

        self.env = KeysAndDoors(
        grid=grid,
        coherence=coherence,
        discount_rate=discount_rate,
        step_cost=step_cost,
        target_reward=target_reward
        )

        self.current_state = None
        self.current_obs = None
    
    def reset(self):
        state_dist = self.env.initial_state_dist()
