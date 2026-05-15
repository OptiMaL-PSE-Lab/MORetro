# MORetro*

[![CI](https://github.com/fredhastedt/MORetro/actions/workflows/test.yml/badge.svg)](https://github.com/fredhastedt/MORetro/actions/workflows/test.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**MORetro\*** is a multi-objective retrosynthesis planning tool. Given a target molecule, it searches for synthesis routes that simultaneously optimise multiple objectives — such as sustainability, scalability and toxicity — and returns the full Pareto front of trade-off solutions.

---

## Installation

### Option A: Clone + uv (recommended for Linux)

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/fredhastedt/MORetro.git
cd MORetro
uv sync --extra cpu   # CPU
uv sync --extra gpu   # GPU (CUDA 11.8)
```

To use `--visualize` (see [CLI](#command-line-arguments)), also install the Graphviz system package:

```bash
sudo apt-get install graphviz   # Linux / WSL2
brew install graphviz           # macOS
```

### Option B: Docker 

Requires [Docker](https://docs.docker.com/get-docker/). For GPU runs, `nvidia-smi` must be available on the host machine.

```bash
git clone https://github.com/fredhastedt/MORetro.git
cd MORetro
docker compose build          # CPU
docker compose --profile gpu build  # GPU
```

---

## Download models

Regardless of installation method, download the models once before running:

```bash
python moretro/preprocess/download_figshare.py
```

Files are saved into `./models/`.

One can also manually download from [Figshare](https://figshare.com/articles/software/Data_and_Models_for_MORetro_/30395614).

---

## Running a search

### Prepare input

Create a CSV/txt file with one or several SMILES string per row (no header). See [`data/example.txt`](data/example.txt).

### Run with uv 

Switch out [`data/example.txt`](data/example.txt) for your own file. 

Set `device = "cpu"` or `device = "cuda"` in `configs/search_config.gin`, then:

```bash
python -m moretro.moretro_star \
  --dataset data/example.txt \
  --config_file search_config.gin \
```

### Run with Docker (CPU)

Switch out [`data/example.txt`](data/example.txt) for your own file.

Set `device = "cpu"` in `configs/search_config.gin`, then:

```bash
docker compose run moretro --dataset /app/data/example.txt --config_file search_config.gin
```

### Run with Docker (GPU)

Switch out [`data/example.txt`](data/example.txt) for your own file.

Set `device = "cuda"` in `configs/search_config.gin`, then:

```bash
docker compose --profile gpu run moretro-gpu \
  --dataset /app/data/example.txt \
  --config_file search_config.gin \
```

---

## Command line arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--dataset` | yes | — | Path to CSV file with one SMILES per row |
| `--config_file` | no | `search_config.gin` | Gin config filename (looked up in `configs/`) |
| `--output_dir` | no | `my_run` | Output directory name; results are written to `output/<name>/` |
| `--visualize` | no | off | Save synthesis route images (PNG/PDF) |
| `--save_json` | no | off | Save all routes as structured JSON files |

---

## Output

All results are written under `output/<output_dir>/<target>/`.

| Path | When | Contents |
|---|---|---|
| `output/my_run/<target>/solution_costs.pkl` | always | All solution cost vectors |
| `output/my_run/<target>/pareto/` | `--visualize` | Pareto-optimal route images |
| `output/my_run/<target>/dominated/` | `--visualize` | Dominated route images |
| `output/my_run/<target>/pareto/` | `--save_json` | Pareto-optimal routes as JSON |
| `output/my_run/<target>/dominated/` | `--save_json` | Dominated routes as JSON |
| `output/my_run/<target>/solution_summary.json` | `--save_json` | Summary with route counts and cost ranges |
| `logs/my_run.log` | always | Run log |

---

## Search configuration

The search is controlled by `configs/search_config.gin`. Key parameters:

| Macro | Default | Description |
|---|---|---|
| `device` | `"cuda"` | Hardware for ML models (`"cpu"` or `"cuda"`) |
| `single_step_model` | `"template"` | Retrosynthesis model: `"template"`, `"pdvn"`, or `"g2e"` |
| `objective_functions` | `["sustainability_cost", "scaleup_cost", "toxicity_cost", "convergence_cost"]` | Objectives to optimise |
| `pareto_objectives` | `3` | Number of objectives used for Pareto filtering |
| `iteration_budget` | `300` | Max MORetro iterations per target |
| `time_budget` | `0` | Wall-clock limit in seconds (`0` = no limit) |
| `max_pareto_solutions` | `150` | Stop early when this many Pareto solutions are found |
| `sampling_strategy` | `"bo"` | Weight update strategy: `"bo"` (Bayesian optimisation) or `"queue"` |

See [`configs/README.md`](configs/README.md) for the full parameter reference including search, weight sampling, and BO selector settings.

---

## Adding custom objectives

Each objective requires two implementations that share the same string key:

**1. Cost function** — `moretro/inference/calculate_costs.py`

Called once per predicted reaction. Return a float in `[0, 1]` (0 = best).

```python
def my_cost(prediction: dict) -> float:
    # available keys: rxn_smiles, reactants, template, score, reagents, temperature
    return 0.5

COST_MAPPING = {
    ...,
    "my_cost": my_cost,
}
```

If the function needs a model loaded from disk, add an `elif` branch to `cost_loader()` in the same file (follows the same pattern as `toxicity_cost` or `scaleup_cost`).

**2. Heuristic function** — `moretro/inference/heuristic_functions.py`

Called on a molecule SMILES to estimate the cost before the reaction is expanded. Return a float in `[0, 1]`.

```python
def my_heuristic(smiles: str) -> float:
    return 0.3

COST_MAPPING = {
    ...,
    "my_cost": my_heuristic,
}
```

Similarly, add an `elif` to `heuristic_loader()` if a model needs initialising.

**3. Register in the config** — `configs/search_config.gin`

Add the key to `objective_functions`. Everything else wires up automatically.

```
objective_functions = ["sustainability_cost", "scaleup_cost", "my_cost"]
```

> Note: when using `"pdvn"` or `"g2e"` as the single-step model, replace `"convergence_cost"` with `"policy_cost"` — a heuristic for this is already provided.

---

## Development

```bash
uv sync --dev --extra cpu   # or --extra gpu for CUDA
uv run pytest               # run tests
uv run ruff check .         # lint
uv run ruff format .        # format
```
