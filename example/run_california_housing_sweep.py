# Imports
#---------------------------------------------------------------------------------------#

from asha_sweep import HyperbandASHA

#---------------------------------------------------------------------------------------#

# Entry Point
#---------------------------------------------------------------------------------------#

if __name__ == "__main__":

    hyperband = HyperbandASHA(
        venv_path='../../envs/california_housing',
        evaluate_script='california_housing_train.py',
        config_path="california_housing_sweep_cfg.yaml", 
        save_path="example/outputs/",
        max_resource=100, 
        reduction_factor=4, 
        gpu_workers=[4,5,6]
    )
    hyperband.run()

#---------------------------------------------------------------------------------------#