from gym.envs.registration import register

register(
    id='asset_allocation-v0',
    entry_point='gym_asset_allocation.envs:AssetAllocationEnv',
)