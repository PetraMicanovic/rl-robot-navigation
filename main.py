"""
Entry point for robot navigation RL training and experiments.

Set MODE to control what runs:

  "curriculum"  — trains the curriculum model only (use this first).
  "experiments" — runs all four experiments using already-trained models.
  "all"         — trains the curriculum model, then runs all experiments.

Typical workflow:
  1. Set MODE = "curriculum" and run to train the model.
  2. Set MODE = "experiments" and run to evaluate and visualise results.
"""
from training.train import train_curriculum
from experiments.experiment_e1 import run_e1
from experiments.experiment_e2 import run_e2
from experiments.experiment_e3 import run_e3
from experiments.experiment_e4 import run_e4
from visualization.visualize import run_visualization

# Configure what to run
# "curriculum" — only train the curriculum model
# "experiments" — only run experiments (models must already exist)
# "all" — train curriculum, then run all experiments
MODE = "all"

if __name__ == "__main__":

    # Curriculum training 
    # Trains the agent through progressively harder stages and saves the
    # final model as models/ppo_robot_nav_curriculum_shaping.zip
    if MODE in ("curriculum", "all"):
        train_curriculum(use_reward_shaping=True)

    # Experiments 
    # train_models = True -> each experiment trains its own model from scratch
    # train_models = False -> experiments evaluate the curriculum model directly
    if MODE in ("experiments", "all"):
        train_models = MODE == "experiments"

        run_e1(train_models=train_models)  # varying obstacle count
        run_e2(train_models=train_models)  # varying obstacle speed
        run_e3(train_models=train_models)  # generalization to unseen configs
        run_e4(train_models=train_models)  # reward shaping vs no shaping

        # Generate plots and summary tables from saved results
        run_visualization()