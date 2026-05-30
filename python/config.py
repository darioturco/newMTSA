"""
Default hyperparameters for each agent and the shared training loop.

To customise a run, edit the relevant dict here before launching main.py.
All keys must match the corresponding agent constructor parameter names exactly.
"""

DQN = dict(
    use_lstm=False,
    use_transformer=False,
    lstm_hidden=64,
    seq_len=16,
    min_episodes=20,
    episode_capacity=1000,
    hidden_size=128,
    lr=1e-3,
    gamma=0.99,
    buffer_size=10_000,
    batch_size=64,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay="auto",
    epsilon_decay_episodes=800,
    target_update_freq=200,
    tau=1.0,
    min_replay_size=500,
    weight_decay=0.0,
    double_dqn=False,
    d_model=64,
    nhead=4,
    num_transformer_layers=2,
    save_frequency=1,
    gradient_steps=1,
    priority_mode='none', # 'none' | 'td_error' | 'reward'
    priority_alpha=0.6,
    priority_beta=0.4,
)

PPO = dict(
    use_lstm=False,
    use_transformer=False,
    lstm_hidden=64,
    hidden_size=128,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    ppo_epochs=4,
    mini_batch=64,
    d_model=64,
    nhead=4,
    num_transformer_layers=2,
    save_frequency=1,
)

SAC = dict(
    use_lstm=False,
    use_transformer=False,
    lstm_hidden=64,
    seq_len=16,
    min_episodes=50,
    episode_capacity=1000,
    hidden_size=128,
    lr_q=3e-4,
    lr_actor=3e-4,
    lr_alpha=3e-4,
    gamma=0.99,
    tau=0.005,
    buffer_size=10_000,
    batch_size=64,
    target_entropy=1.0,
    auto_alpha=True,
    init_alpha=0.2,
    min_replay_size=500,
    target_update_freq=1,
    d_model=64,
    nhead=4,
    num_transformer_layers=2,
    save_frequency=1,
)

TRAIN = dict(
    max_episodes=5000,
    max_steps=1_000_000,
    patience=500,
)
