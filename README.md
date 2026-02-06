# MORetro

MORetro is a Multi-Objective Retrosynthesis planning tool.

## Installation

Create the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate moretro
```

## Usage

The main entry point for running the search is the `moretro.moretro_star` module.

```bash
python -m moretro.moretro_star --dataset <path_to_dataset> --output_dir <output_directory> --config_file <config_filename>
```

**Arguments:**
- `--dataset`: Path to the input file containing target SMILES (CSV format).
- `--output_dir`: Directory where results and logs will be saved. Default is `output`.
- `--config_file`: Name of the gin config file to use (located in `moretro/configs/`). Default is `search_config.gin`.

## Configuration

The main configuration file is located at `moretro/configs/search_config.gin`. You can modify various parameters to customize the search behavior.

### Single-Step Predictor
You can change the single-step retrosynthesis predictor by modifying the `retro_model` path and the `OneStepModel.model_type`.

In `moretro/configs/search_config.gin`:
```python
# Checkpoint files
retro_model = "models/model_retro.pt"  # Path to your model checkpoint

# ...

# Retro model config
retro_prediction.OneStepModel.model_type = "st" # "st" for single-step, etc.
retro_prediction.OneStepModel.checkpoint_path = %retro_model
```

Make sure `OneStepModel.model_type` matches the type of model you are using (e.g., `"st"` for the standard model).

### Sampling Strategy
You can choose between Bayesian Optimization (BO) or a simple Queue-based strategy for weight sampling during the multi-objective search.

Modify `MOGraph.weight_update_strategy` in `moretro/configs/search_config.gin`:

```python
# Weight Selection Parameters
# ...
MOGraph.weight_update_strategy = "bo"  # Options: "bo" or "queue"
```

- `"bo"`: Uses Bayesian Optimization to select weight vectors.
- `"queue"`: Uses a standard queue for weight processing.
