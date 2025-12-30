"""
Optimize search speed and node reduction parameters using Optuna.
This script focuses on tuning only the 'search' parameters in skaks_opt.
"""
import optuna
import yaml
import time
from pathlib import Path
from skaks_opt.params import DEFAULT_PARAMS, default_param_space, apply_param_updates
from skaks_opt.selfplay import evaluate_candidate

# Only tune search-related parameters
def search_param_space():
    return [spec for spec in default_param_space() if spec.name.startswith("search.")]

def objective(trial):
    # Suggest values for each search parameter
    params = DEFAULT_PARAMS["search"].copy()
    for spec in search_param_space():
        if spec.is_float:
            val = trial.suggest_float(spec.name, spec.low, spec.high, step=spec.step)
        else:
            val = trial.suggest_int(spec.name, int(spec.low), int(spec.high), step=int(spec.step or 1))
        key = spec.name.split(".", 1)[1]
        params[key] = val
    # Dummy baseline (could load from file)
    baseline = DEFAULT_PARAMS["search"].copy()
    # Evaluate speed and node reduction (replace with real call)
    start = time.time()
    # Here you would call your engine with these params and measure speed/nodes
    # For now, just simulate
    time.sleep(0.1)
    elapsed = time.time() - start
    # Lower is better (minimize time)
    return elapsed

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print("Best params:", study.best_params)

if __name__ == "__main__":
    main()
