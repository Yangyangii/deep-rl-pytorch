# coding: utf-8

class Hyperparams:
    
    ENV = 'PongDeterministic-v4'
    MODEL = 'ActorCritic'
    MODE = 'squeeze'
    NO_KEYS = 2
    GAME = 'pong'
    
    HEIGHT = 84
    WIDTH = 84
    
    # Hyperparameters for optimization
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.02
    EPS_DECAY = 20000
    TARGET_UPDATE = 1000
    NO_REPLAY = 30000
    NO_SEQ = 5
    
    no_episodes = 10000
    stop_score = 18.0
