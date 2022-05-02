import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='PandaFrite-v0',
    entry_point='gym_panda_frite.envs:PandaFriteEnv',
    kwargs={'database': None, 'distance_threshold': None, 'gui': None, 'E': None}
)

register(
    id='PandaFrite-v1',
    entry_point='gym_panda_frite.envs:PandaFriteEnvROS',
    kwargs={'database': None, 'json_decoder' : None, 'env_pybullet': None, 'gui': None}
)
