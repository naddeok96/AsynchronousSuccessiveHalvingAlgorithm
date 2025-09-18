# Asynchronous Successive Halving Algorithm (ASHA)

## Algorithm Primer
Asynchronous Successive Halving Algorithm (ASHA) is an early-stopping strategy for hyperparameter optimisation. It evaluates many candidate configurations with a small resource budget (for example, epochs or training steps), ranks their fitness, and successively allocates more resources to the most promising runs while pruning under-performers. The asynchronous variant removes global synchronisation barriers: workers report results as soon as they finish, and the scheduler immediately promotes the best available configurations, keeping all compute resources busy.

## Implementation Highlights
- `HyperbandASHA` in `asha_sweep.py` implements bracket generation, promotion logic, and job launching. It maintains run metadata in CSV ledgers so the controller can recover state between invocations.
- Jobs are executed by calling the provided evaluator script (`california_housing_train.py` in the example). The scheduler now supports CPU-only environments: if CUDA is unavailable it runs configurations sequentially without raising GPU errors. Optional dependencies (`names`, `tqdm`) fall back to lightweight stubs so you can experiment without additional installs.
- Each evaluation script must append its fitness results via `append_fitness_score` and mark itself as running with `modify_current_runs`, enabling the controller to coordinate promotions and avoid oversubscribing workers.

## California Housing Example
The `example/` directory contains a minimal end-to-end sweep that trains a small multilayer perceptron on the California Housing regression task.

### Prerequisites
- Python 3.8+
- PyTorch and scikit-learn
- Virtual environment populated from `requirements.txt` (for example, `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`)
- Optional for richer output: `pip install tqdm names`

### Running the sweep
```bash
cd AsynchronousSuccessiveHalvingAlgorithm
python example/run_california_housing_sweep.py
```
The script caches the dataset under `example/outputs/data/` and writes all sweep artefacts to `example/outputs/`. Re-running the command automatically overwrites the previous outputs for a clean slate.

If you're launching long sweeps, start a `tmux` session (for example, `tmux new -s asha`) so the run continues even if your SSH connection drops.

## Extending Beyond The Example
To adapt ASHA to a new project:
1. **Create an evaluation script** modelled after `example/california_housing_train.py`. It should:
   - Read the sampled hyperparameters from `<run_dir>/config_params.yaml`.
   - Train for the requested resource increment (`--resources_to_add`).
   - Compute a scalar fitness score (larger is better) and call `append_fitness_score(save_path, config_name, fitness_score)`.
   - Use `modify_current_runs` with `add=True/False` to signal worker availability.
2. **Define your search space** in YAML, following `example/california_housing_sweep_cfg.yaml`.
3. **Instantiate `HyperbandASHA`** with paths to your evaluator script, search-space YAML, and a dedicated output directory. Set `max_resource`, `reduction_factor`, and worker configuration to match your problem.
4. **Custom fitness metrics**: inside your evaluator, compute any objective you like (accuracy, negative loss, reward, etc.). Ensure higher values represent better performance, or negate the metric if necessary. You can also log additional metrics by passing `additional_metrics_to_track` when creating `HyperbandASHA` and writing to the corresponding CSV via `append_metric_value`.

With these pieces in place you can plug ASHA into existing training loops, reuse the ledger utilities, and customise the promotion strategy without rewriting the core scheduler.
