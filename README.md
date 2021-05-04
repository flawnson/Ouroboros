# Ouroboros
Introverted NNs, inspired by meta learning techniques, and NN architectures including NN Quines, Introspective NNs, Hypernetworks, and self-referential weight matrices.

# Paper
We have an annotated bibliography in progress ([Link to LaTeX document on Overleaf](https://www.overleaf.com/read/mkyktsvwsjqh)) as part of our literature review efforts.

# <!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

## About The Project
### Built With
Several specialty libraries are used. A list of all packages and their version can be found in the config/envs directory.
### Code Style
Docstrings, typehints, and comments brought to you in [Google Style](https://google.github.io/styleguide/pyguide.html).
### Directory Structure
Directory structure mimics conventional PyTorch projects. A full dated summary can be found in directory_tree.txt (use ```Get-ChildItem | tree /F > foo.txt``` in PowerShell to create your own!)

## Getting Started
### Prerequisites
You will need conda and Python version 3.6 or above.

### Installation
Assuming you're in base dir of this project and are using a Linux based system:
First you'll want to create a new conda (or pip) env with Python 3.7
```shell
conda create -n env_name python=3.7 anaconda
source activate env_name
```

Before cloning into this repository:
```shell
git clone https://github.com/flawnson/Generic_GNN.git
OR
pip install git+https://github.com/flawnson/Generic_GNN.git
```

Then you can run setup.py
```shell
python setup.py
```

### Environment Setup
Install depedencies in the requirements.txt.
```shell
pip install -r configs/envs/requirements_cpu.txt
```

Then you'll need to create an empty directory for model outputs (including saved models).
```shell
cd Generic_GNN && mkdir outputs
```

Finally you can run a demo version of the pipeline (default configs in configs directory).
```shell
python -c path/to/config/files/file.json -s path/to/schema/files/file.json
```

You can see the logged results using TensorBoard (to be setup soon).
```shell
tensorboard --logdir=logs/GAT_tuning/tune_model
```

### Container Setup
Docker containers for running the project are on the roadmap!

## Usage
The core component of usage is the training pipeline. The tuning pipeline, parallelization pipeline, and all others inherit from the Trainer superclass.

Run:
```shell
python main.py -c "path_to_config_file.json"
```

### Demo
Demo is a simple training run. It takes a configuration file and runs the Trainer once.

### Tuning
Tuning is a pair of consecutive runs. The first run executes the Tuner (a wrapper of the Trainer pipeline meant to find and return the best parameters it can find) once and the second run executes the Trainer once.

### Parallelizing
Parallelizing allows you to execute and run several Demo and/or Tuning pipelines in tandem. It uses mutliprocessing to find and use as many cores you define in the confiuration file (yet to be implemented).

### Logging and Checkpointing
Logging is controlled by the config files.
1. Console logs - Runs are logged to console with logzero (mostly consists of info and exception logs) and saved as a `.txt` file in the `saves/logs` directory.
2. Config logs - A copy of the config is saved as a `.json` for each run in the `saves/logs` directory.
3. Tensorboard logs - Saved in the `runs` directory, used to visualize training.

Model checkpointing is performed with PyTorch's `save` function.
1. Model checkpoints are saved at each interval as specified in the `run_config` (saved in the `saves/checkpoints` directory).
2. The model file itself is copied into the checkpoint directory, where it can be used with the saved `.json` config (saved in the `saves/checkpoints` directory).

## Roadmap

## Contributing
Currently we have a couple branches for making changes and contributions.
New branches can be created if the process of achieving the desired outcome risks breaking the existing copy of the codebase on main.

## License

## Contact
The core contributors are reachable by email, Twitter, and most other means.

## Acknowledgements
Thank you to Kevin for being a reliable partner, and a close friend.
