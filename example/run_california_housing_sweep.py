# Imports
#---------------------------------------------------------------------------------------#

import sys
from pathlib import Path

# Ensure the ASHA module is importable when running this script directly
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asha_sweep import HyperbandASHA

#---------------------------------------------------------------------------------------#

# Entry Point
#---------------------------------------------------------------------------------------#

if __name__ == "__main__":
    print("Launching Hyperband-ASHA sweep for the California Housing example...")

    hyperband = HyperbandASHA(
        venv_path=str(PROJECT_ROOT.parent / "envs/california_housing"),
        evaluate_script=str(CURRENT_DIR / "california_housing_train.py"),
        config_path=str(CURRENT_DIR / "california_housing_sweep_cfg.yaml"),
        save_path=str(CURRENT_DIR / "outputs"),
        max_resource=75,
        reduction_factor=4,
        gpu_workers=[1,2,4,5,6,7],
        num_runs_per_gpu=5,
        time_between_runs=3,
    )
    hyperband.run()

#---------------------------------------------------------------------------------------#
