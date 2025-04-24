# Mixed pendulum

Hyperparameters include:
- n_controls
- n_episodes
- n_runs
- lr
- tau (for gumbel)

Description of the files:
- `mixed.py` - compares the gumbel and categorical methods for the mixed pendulum environment.
- `mixed_scat.py` - compares the cat and cat++ method
- `mixed_final.py` - compares all the methods

To run the code, use the following command:
```bash
python mixed_final.py
```

> Code to produce plots of reward function vs episodes is commented out. can be uncommented to visualize the results.