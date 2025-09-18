## Imports
#---------------------------------------------------------------------------------------#

# Standard Library Imports
import os
import csv
import math
import time
import torch
import random
import shutil
import subprocess
from abc import ABC

# Third-Party Imports
import yaml
import names
from tqdm import tqdm

#---------------------------------------------------------------------------------------#

## Functions
#---------------------------------------------------------------------------------------#

def get_fitness_scores(save_path, config_name):
    """
    Retrieve fitness scores from a CSV file for a given configuration name.

    Parameters:
        save_path (str): The directory where the fitness_scores.csv file is located.
        config_name (str): The column name in the CSV file representing the configuration of interest.

    Returns:
        list: A list of fitness scores as floats. Returns an empty list if the file or column is missing.
    """
    # Construct the full path to the fitness_scores.csv file.
    fitness_scores_path = os.path.join(save_path, 'fitness_scores.csv')
    scores = []  # Initialize an empty list to store scores.
    max_retries = 10
    wait_time = 30  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            # Open the fitness scores file and read its contents.
            with open(fitness_scores_path, 'r') as f:
                reader = csv.DictReader(f)  # Use DictReader to access rows as dictionaries.

                # Extract scores for the given config_name from the rows.
                for row in reader:
                    score = row.get(config_name)
                    if score:  # Skip rows where the score is missing or None.
                        scores.append(score)

            # Convert scores from strings to floats.
            scores = [float(x) for x in scores]
            return scores  # Successfully retrieved scores, exit the function.

        except (FileNotFoundError, ValueError) as e:
            if attempt > max_retries:
                print("Error accessing fitness scores...")
                exit()

def append_metric_value(save_path, config_name, metric_name, metric_value):
    """
    Append a metric value to a specified configuration column in a metric CSV file.
    If the file or column does not exist, they will be created.

    Parameters:
        save_path (str): Directory where the metric CSV file is located or should be created.
        config_name (str): Column name representing the configuration to which the metric will be appended.
        metric_name (str): Base name of the CSV file (without extension) to store this metric (e.g., 'fitness_scores').
        metric_value (Any): The metric value to append (e.g., float, int, str).

    Returns:
        None
    """
    # Construct the CSV filename and full path
    csv_filename = f"{metric_name}.csv"
    file_path = os.path.join(save_path, csv_filename)

    # If file doesn't exist, create it with header and first row
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[config_name])
            writer.writeheader()
            writer.writerow({config_name: metric_value})
        return

    # Read existing data
    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    # Add the column if it doesn't exist
    if config_name not in fieldnames:
        fieldnames.append(config_name)
        for row in rows:
            row[config_name] = ''

    # Find first empty cell in the specified column
    for row in rows:
        if not row.get(config_name):
            row[config_name] = metric_value
            break
    else:
        # No empty cell found: append a new row
        new_row = {name: '' for name in fieldnames}
        new_row[config_name] = metric_value
        rows.append(new_row)

    # Write back to the CSV file
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
def get_current_runs(save_path):
    """
    Reads the `current_runs.csv` file in the specified save path and retrieves 
    a dictionary of current runs.

    Each key in the dictionary represents a configuration name, and the corresponding 
    value represents the associated GPU number.

    Parameters:
        save_path (str): The directory where the `current_runs.csv` file is located.

    Returns:
        dict: A dictionary with configuration names as keys and GPU numbers as values.
              Returns an empty dictionary if the file does not exist or is empty.
    """
    # Construct the full path to the current_runs.csv file.
    current_runs_path = os.path.join(save_path, 'current_runs.csv')
    runs_dict = {}  # Initialize an empty dictionary to store the runs.

    # Check if the current_runs.csv file exists.
    if not os.path.exists(current_runs_path):
        return runs_dict

    # Open the file and read its content.
    with open(current_runs_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Ensure the row has at least two elements (config name and GPU number).
            if row and len(row) >= 2:
                config_name, gpu_number = row[0], row[1]
                # Add the config name and GPU number to the dictionary.
                runs_dict[config_name] = gpu_number

    return runs_dict

def modify_current_runs(save_path, config_name, gpu_number=None, add=True):
    """
    Modify entries in the `current_runs.csv` file by adding, updating, or removing a config name.

    Parameters:
        save_path (str): The directory where the `current_runs.csv` file is located.
        config_name (str): The configuration name to modify.
        gpu_number (str, optional): The GPU number to associate with the config name when adding/updating.
                                    Required if `add=True`.
        add (bool): If True, add or update the config name with the specified GPU number.
                    If False, remove the config name from the file.

    Returns:
        None
    """
    # Construct the full path to the `current_runs.csv` file.
    current_runs_path = os.path.join(save_path, 'current_runs.csv')

    # Ensure the CSV file exists by creating it if it doesn't.
    if not os.path.exists(current_runs_path):
        with open(current_runs_path, 'w', newline='') as f:
            pass  # Create an empty file.

    # Load the current dictionary of runs.
    runs_dict = get_current_runs(save_path)

    if add:
        # Ensure a GPU number is provided when adding/updating a config name.
        if gpu_number is None:
            raise ValueError("gpu_number must be specified when adding or updating a config.")
        # Add or update the config name with the new GPU number.
        runs_dict[config_name] = gpu_number
    else:
        # Remove the config name from the dictionary if it exists.
        if config_name in runs_dict:
            del runs_dict[config_name]

    # Write the updated dictionary back to the CSV file.
    with open(current_runs_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write each config name and its associated GPU number as rows.
        for name, gpu in runs_dict.items():
            writer.writerow([name, gpu])

#---------------------------------------------------------------------------------------#


## Classes
#---------------------------------------------------------------------------------------#
 
class HyperbandASHA(ABC):

    def __init__(
            self, 
            venv_path,
            evaluate_script,
            config_path, 
            save_path,
            max_resource=81, 
            reduction_factor=4, 
            gpu_workers=[0,1,2,3,4,5],
            num_runs_per_gpu = 1,
            time_between_runs=10,
            additional_metrics_to_track=[]
        ):
        """
        Initialize the Hyperband-ASHA scheduler.

        Args:
            venv_path (str): Path to the virtual environment used for evaluation.
            evaluate_script (str): Path to the script for evaluating configurations.
            config_path (str): Path to the YAML file defining the hyperparameter search space.
            save_path (str): Path to save results.
            max_resource (int): Maximum resource allocated per configuration (default: 81).
            reduction_factor (int): Reduction factor (\u03b7) (default: 4).
            gpu_workers (list): List of GPU indices available for runs (default: [0, 1, 2, 3, 4, 5]).
            num_runs_per_gpu (int): Number of runs allowed on a single gpu
            time_between_runs (int): Time to wait before starting another run, in seconds (default: 10).
            additional_metrics_to_track (list, optional): list of additional metrics to track during evaluation besides fitness_scores.
        """
        self.venv_path = venv_path  # Path to the virtual environment
        self.evaluate_script = evaluate_script  # Path to the evaluation script
        self.config_path = config_path  # Path to the hyperparameter search space YAML file
        self.save_path = save_path  # Directory for saving results
        self.time_between_runs = time_between_runs  # Delay between starting new runs
        self.max_resource = max_resource  # Maximum resources per configuration
        self.reduction_factor = reduction_factor  # Reduction factor for Hyperband-ASHA
        self.gpu_workers = gpu_workers  # List of GPU indices for parallel runs
        self.num_runs_per_gpu = num_runs_per_gpu # Number of runs allowed on a gpu
        self.config_names = set()  # Set of unique configuration names
        self.config_tuples = set()  # Set of unique configuration tuples
        self.additional_metrics_to_track = additional_metrics_to_track # Additional metrics to track

        # Check if the save path exists
        if os.path.exists(self.save_path):
            response = input(f"The path '{self.save_path}' already exists. Would you like to overwrite it? (yes/no): ").strip().lower()
            if response == 'yes':
                shutil.rmtree(self.save_path)  # Remove the existing directory
            elif response == 'no':
                print("Operation aborted. Exiting...")
                exit(0)
            else:
                print("Invalid response. Please type 'yes' or 'no'. Exiting...")
                exit(1)

        # Initialize GPUs, search space, and directories
        self.reserve_gpus()
        self.search_space = self.load_search_space(config_path)
        self.initialize_brackets()
        self.create_save_path()

    def reserve_gpus(self):
        """
        Reserves GPUs by creating and holding a placeholder tensor on each GPU.

        This method initializes a dictionary `holders` to keep track of placeholder tensors 
        on each GPU specified in `self.gpu_workers`. It creates a tensor with a value of 42 
        and moves it to the corresponding GPU using the `cuda` device string.

        Example:
            If `self.gpu_workers` contains [0, 1, 2], this method will create a tensor 
            on each GPU (cuda:0, cuda:1, cuda:2) and store them in `self.holders`.

        Attributes:
            holders (dict): A dictionary mapping GPU indices to their corresponding 
                            placeholder tensors.
        """
        self.holders = {}
        for gpu in self.gpu_workers:
            self.holders[gpu] = torch.tensor(42).to(f"cuda:{gpu}")

    def create_save_path(self):
        """
        Creates or verifies the directory specified by `self.save_path`.

        If the directory already exists, the user is prompted to decide whether to overwrite it.
        - If the user chooses "yes", the existing directory is deleted and recreated.
        - If the user chooses "no", the program exits.
        - If the user provides an invalid response, the program also exits with an error message.

        If the directory does not exist, it is created.

        Raises:
            SystemExit: Exits the program if the user chooses not to overwrite or provides an invalid response.
        """
        
        
        os.makedirs(self.save_path)
        print(f"Path created at: {self.save_path}")

        # Create the metrics .csv files with headers from config_names
        metrics_to_track = ['fitness_scores'] + self.additional_metrics_to_track
        for metric in metrics_to_track:
            metric_path = os.path.join(self.save_path, f"{metric}.csv")
            with open(metric_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.config_names)

        # Create an empty current_runs.csv file
        current_runs_path = os.path.join(self.save_path, 'current_runs.csv')
        with open(current_runs_path, 'w') as f:
            pass  # Create an empty file

    def load_search_space(self, config_path):
        """
        Load the search space from a YAML configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            dict: Parsed search space.
        """
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def sample_configuration(self):
        """
        Generate a random configuration from the search space.

        Returns:
            dict: A single hyperparameter configuration.
        """
        config = {}
        for param, settings in self.search_space.items():
            if settings['type'] == 'categorical':
                config[param] = random.choice(settings['values'])
        return config
    
    def get_configurations(self, num_configurations):
        """
        Randomly sample unique configurations from the search space.

        Args:
            num_configurations (int): Number of configurations to sample.

        Returns:
            dict: A dictionary of unique configurations with names as keys.
        """
        configurations = {}
        attempts = 0
        max_attempts = 10 * num_configurations  # Limit retries to prevent infinite loops

        with tqdm(total=num_configurations, desc="Sampling Configurations") as pbar:
            while len(configurations) < num_configurations and attempts < max_attempts:
                config = self.sample_configuration()

                # Convert lists in the config dictionary to tuples
                immutable_config = {
                    key: tuple(value) if isinstance(value, list) else value
                    for key, value in config.items()
                }

                # Create a sorted tuple of the items
                config_tuple = tuple(sorted(immutable_config.items()))

                if config_tuple not in self.config_tuples:
                    self.config_tuples.add(config_tuple)
                    # Generate a unique name for the configuration
                    while True:
                        first_name = names.get_first_name()
                        middle_name = names.get_first_name()
                        last_name = names.get_last_name()
                        name = f"{first_name}_{middle_name}_{last_name}"

                        if name not in self.config_names:
                            self.config_names.add(name)
                            break
                    configurations[name] = {"params": config}
                    pbar.update(1)  # Update the progress bar
                attempts += 1

        if len(configurations) < num_configurations:
            raise ValueError(
                f"Not enough unique configurations available. "
                f"Increase the search space or reduce the number of configurations."
            )

        return configurations

    def initialize_brackets(self):
        """
        Initialize the brackets for the Hyperband algorithm based on the reduction factor.

        The brackets represent different levels of aggressiveness (degree of exploration) 
        for allocating resources to configurations. Brackets are determined by scaling the 
        total number of configurations and resource allocation geometrically.

        Process:
        - Compute the maximum number of brackets based on the maximum resource and reduction factor.
        - Exclude the two least aggressive brackets for better focus on high exploration.
        - For each remaining bracket:
            - Calculate the number of configurations and the minimum resource per configuration.
            - Allocate resources for each rung geometrically, scaling with the reduction factor.
        - Store the brackets with their respective configurations and rungs.

        Attributes set:
        - `self.brackets`: A dictionary where each key is a bracket index `s`, and the value 
        contains:
            - 'configs': A list of configurations for the bracket.
            - 'rungs': A dictionary mapping rung indices to allocated resources.
        """
        max_bracket = math.floor(math.log(self.max_resource, self.reduction_factor))
        # Exclude the two least aggressive brackets (0 and 1) and list from highest to lowest
        brackets_list = list(range(max_bracket + 1))[:-2]

        print(f"Brackets to generate: {brackets_list}")

        self.brackets = {}
        for s in brackets_list:
            # Total number of configurations in the bracket
            num_configs = math.ceil((self.reduction_factor ** (max_bracket - s)) / (s + 1))

            print(f"Begin generating {num_configs} configs for bracket {s}....")

            # Minimum resource per configuration in the first rung
            min_resource = max(1, self.max_resource // (self.reduction_factor ** (max_bracket - s)))

            # Initialize rungs with properly scaled resources
            rungs = {}
            current_resource = min_resource
            for i in range(max_bracket - s + 1):
                rungs[i] = current_resource
                current_resource = min(current_resource * self.reduction_factor, self.max_resource)

            # Add the bracket with its configurations and rungs
            try:
                self.brackets[s] = {
                    'configs': self.get_configurations(num_configs),
                    'rungs': rungs
                }
            except ValueError as e:
                raise ValueError(f"Error in bracket {s}: {e}")
            
        for bracket_num  in self.brackets.keys():
            print(f"Bracket {bracket_num} ({len(self.brackets[bracket_num]['configs'])}): {self.brackets[bracket_num]['rungs']}")

    def top_k_div_n_indices(self, fitness_scores_list):
        """
        Selects the top `k / n` indices from a list of fitness scores.

        This method computes the number of top indices to return by dividing the total number 
        of fitness scores (`k`) by the reduction factor (`self.reduction_factor`). The indices 
        are determined based on the highest fitness scores.

        Parameters:
            fitness_scores_list (list): A list of fitness scores, where each score is associated with an index.

        Returns:
            list: A list of indices corresponding to the top `k / n` scores, where `k` is the length 
            of `fitness_scores_list` and `n` is `self.reduction_factor`. If `k` is less than 
            `self.reduction_factor`, an empty list is returned.

        Notes:
            - If the length of `fitness_scores_list` is less than `self.reduction_factor`, the method 
            returns an empty list.
            - The scores are sorted in descending order, and the indices of the top scores are returned.
        """
        k = len(fitness_scores_list)
        if k < self.reduction_factor:
            slice_size = 0  # Return none if length of available scores is less than the reduction factor (explicit case)
        else:
            slice_size = k // self.reduction_factor

        sorted_indices = sorted(range(len(fitness_scores_list)), key=lambda i: fitness_scores_list[i], reverse=True)
        top_indices = sorted_indices[:slice_size]  # The top K/n indices
        return top_indices

    def get_job(self, bracket):
        """
        Selects the next configuration to execute based on the current state of the bracket and rungs.

        This method implements a hyperparameter optimization strategy that promotes configurations 
        across rungs in a bracket based on their fitness scores. It ensures that configurations are 
        promoted and selected in a manner consistent with the optimization process.

        Parameters:
            bracket (dict): A dictionary representing the state of a bracket. It should include:
                - "rungs" (dict): Keys are rung indices, and values contain rung-specific metadata.
                - "configs" (dict): Keys are configuration names, and values include details 
                about each configuration's progress.

        Returns:
            tuple:
                - str: The name of the configuration selected for execution.
                - int: The rung index of the selected configuration.

        Behavior:
            - The method first checks for promotable configurations in higher rungs based on fitness scores.
            - If no promotable configuration is found, it checks for new configurations in the current rung.
            - If no configurations are available, the method waits for 30 seconds and retries.

        Notes:
            - A configuration is promotable if it has enough fitness scores to meet the requirements of the current rung.
            - The method avoids selecting configurations that are already running.
            - Configurations are selected based on their fitness scores, ensuring the top-performing ones are prioritized.
            - If no valid configuration is found, the method waits before retrying.

        Raises:
            SystemExit: If an unexpected error occurs, the method may lead to indefinite waiting.

        Example:
            bracket = {
                "rungs": {1: {}, 2: {}, 3: {}},
                "configs": {"config1": {...}, "config2": {...}}
            }
            selected_config, rung_index = self.get_job(bracket)
        """
        while True:  # Loop to retry if no valid configuration is found
            # Get the list of currently running configurations
            current_runs = get_current_runs(self.save_path)

            # Check all rungs in reverse order except the first one as it is default
            for k in list(reversed(bracket["rungs"].keys()))[:-1]: 
                # Get config names that are either on this rung or have already been promoted to the next rungs
                rung_config_names = [
                    config_name for config_name in bracket["configs"].keys()
                    if len(get_fitness_scores(self.save_path, config_name)) >= k
                ]

                # If configs on this rung or higher exist
                if len(rung_config_names) > 0:
                    # Get list of fitness scores from the rung before the current rung being evaluated
                    fitness_scores = [
                        get_fitness_scores(self.save_path, config_name)[k-1]
                        for config_name in bracket["configs"].keys()
                        if len(get_fitness_scores(self.save_path, config_name)) >= k
                    ]

                    # Determine the top k configs
                    top_k_in_rung_indices = self.top_k_div_n_indices(fitness_scores)
                    top_k_rung_config_names = [rung_config_names[i] for i in top_k_in_rung_indices]

                    # Check if any of the top k configs are promotable and not already running
                    best_promotable_config_name = None
                    best_promotable_config_fitness_score = -1e6
                    for candidate_config_name in top_k_rung_config_names:
                        candidate_is_promotable = (
                            len(get_fitness_scores(self.save_path, candidate_config_name)) == k
                        )
                        if candidate_is_promotable and candidate_config_name not in current_runs:
                            candidate_fitness_score = get_fitness_scores(self.save_path, candidate_config_name)[k-1]

                            if candidate_fitness_score > best_promotable_config_fitness_score:
                                best_promotable_config_fitness_score = candidate_fitness_score
                                best_promotable_config_name = candidate_config_name

                    if best_promotable_config_name is not None:
                        return best_promotable_config_name, k

            # If no promotable configuration is found, check for new configurations on the current rung
            for k in list(bracket["rungs"].keys()): 
                remaining_rung_config_names = [
                    config_name for config_name in bracket["configs"].keys()
                    if len(get_fitness_scores(self.save_path, config_name)) == k
                    and config_name not in current_runs
                ]
                if len(remaining_rung_config_names) > 0:
                    return random.choice(remaining_rung_config_names), k

            # If no valid configuration is found, wait for 30 seconds and try again
            time.sleep(30)
    
    def get_available_gpus(self):
        """
        Retrieves a list of available GPU indices.

        This method checks the GPUs currently in use and returns the indices of GPUs 
        that are not being overutilized based on the allowed number of runs per GPU.

        Returns:
            list: A list of integers representing the indices of available GPUs.

        Notes:
            - The method fetches the list of GPUs currently in use by querying the 
            current runs using `get_current_runs(self.save_path)`.
            - The method compares the list of all GPU workers (`self.gpu_workers`) 
            against the list of GPUs currently in use and excludes the busy ones
            based on `self.num_runs_per_gpu`.

        Example:
            If `self.gpu_workers = [0, 1, 2, 3]`, `self.num_runs_per_gpu = 2`, and
            GPUs `[1, 1, 3]` are in use, the method will return `[0, 2, 3]` since
            GPU 3 has only 1 run.

        Dependencies:
            - Assumes `get_current_runs(self.save_path)` returns a dictionary where 
            the values represent the indices of GPUs currently in use.
        """
        # Fetch currently used GPUs
        used_gpus = [int(x) for x in get_current_runs(self.save_path).values()]
        
        # Count the number of runs per GPU
        gpu_usage_count = {gpu: used_gpus.count(gpu) for gpu in self.gpu_workers}
        
        # Find GPUs that have not exceeded the allowed number of runs
        available_gpus = [
            gpu for gpu, count in gpu_usage_count.items() 
            if count < self.num_runs_per_gpu
        ]
        
        return available_gpus

    def start_job(self, bracket_num, config_name, config_params, resources_to_add, gpu_number):
        """
        Initializes and starts a job for a specific configuration in a given bracket.

        This method sets up the directory structure for the job, saves the configuration 
        parameters if they do not already exist, and then starts the evaluation process 
        for the specified configuration.

        Parameters:
            bracket_num (int): The index of the bracket to which the job belongs.
            config_name (str): The unique name of the configuration to be executed.
            config_params (dict): The parameters for the configuration, stored as a dictionary.
            resources_to_add (dict): Additional resources to include during the evaluation process.
            gpu_number (int): The index of the GPU to be used for this job.

        Behavior:
            - Ensures the necessary directory structure exists for the job.
            - Saves the configuration parameters in a `config_params.yaml` file if it does not already exist.
            - Creates a subdirectory for storing trained models if it does not already exist.
            - Invokes the `self.evaluate` method to begin the evaluation process with the specified parameters.

        Notes:
            - The method assumes that `self.evaluate` is a separate method responsible for executing 
            the evaluation process using the provided `run_dir`, `resources_to_add`, and `gpu_number`.
            - Directory paths include:
            - `runs/<bracket_num>/<config_name>`: Base directory for the job.
            - `config_params.yaml`: File for storing configuration parameters.
            - `trained_models`: Subdirectory for storing trained models.

        """
        # Define paths
        run_dir = os.path.join(self.save_path, "runs", f"bracket_{bracket_num}", config_name)
        config_yaml_path = os.path.join(run_dir, "config_params.yaml")
        trained_models_dir = os.path.join(run_dir, "trained_models")

        # Check if the run directory exists
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # Save configuration parameters if config_params.yaml does not exist
        if not os.path.exists(config_yaml_path):
            with open(config_yaml_path, 'w') as yaml_file:
                yaml.dump(config_params, yaml_file)

        # Check if trained_models subdirectory exists
        if not os.path.exists(trained_models_dir):
            os.makedirs(trained_models_dir)

        self.evaluate(
            run_dir, 
            resources_to_add, 
            gpu_number
        )

    def evaluate(self, run_dir, resources_to_add, gpu_number):
        """
        Evaluate a given configuration by running a Python script using a specified virtual environment.

        This method constructs and executes a command to call a Python script (`self.evaluate_script`) 
        located in a virtual environment specified by `self.venv_path`. The script is executed with 
        parameters for the run directory, number of epochs, and GPU index.

        Args:
            run_dir (str): Path to the directory containing the configuration files for the run.
            resources_to_add (int): Number of additional epochs or resources to allocate.
            gpu_number (int): GPU index to use for the evaluation process.

        Raises:
            subprocess.CalledProcessError: If the script fails to execute successfully.
        """
        # Ensure the terminal_outputs subdirectory exists
        terminal_outputs_dir = os.path.join(run_dir, "terminal_outputs")
        os.makedirs(terminal_outputs_dir, exist_ok=True)

        # Find the next available terminal output filename
        file_index = 0
        while os.path.exists(os.path.join(terminal_outputs_dir, f"terminal_outputs_run_{file_index}.txt")):
            file_index += 1

        output_file = os.path.join(terminal_outputs_dir, f"terminal_outputs_run_{file_index}.txt")

        # Define the Python executable from the virtual environment
        python_executable = f"{self.venv_path}/bin/python3"

        # Construct the command to call the script
        command = [
            python_executable,
            self.evaluate_script,
            "--run_dir", run_dir,
            "--resources_to_add", str(resources_to_add),
            "--gpu", str(gpu_number),
        ]

        # Run the command and capture stdout/stderr
        try:
            with open(output_file, "w") as f:
                subprocess.Popen(command, stdout=f, stderr=f)
        except Exception as e:
            print(f"Error occurred while running the script: {e}")

    def run(self):
        """
        Executes the Hyperband optimization process across all defined brackets.

        This method iterates through each bracket, managing the allocation of resources 
        to configurations and tracking progress across rungs. Configurations are evaluated 
        based on their fitness scores, and the best configuration is determined once the 
        stopping criteria are met.

        Workflow:
            1. Display the total number of brackets and resources allocated per rung.
            2. Initialize progress bars to track the percentage of configurations evaluated 
            in each rung.
            3. Allocate available GPUs to configurations and start jobs for evaluation.
            4. Update progress bars after each evaluation.
            5. Check stopping criteria to determine when to end the process for a bracket.
            6. Identify and display the best configuration once all brackets are complete.

        Attributes Used:
            - self.brackets (dict): The dictionary containing bracket information, including rungs, 
            configurations, and their parameters.
            - self.save_path (str): Path where fitness scores and other job information are saved.
            - self.time_between_runs (int): Time to wait between job submissions.

        Methods Invoked:
            - self.get_available_gpus(): Retrieves a list of available GPU indices.
            - self.get_job(bracket): Fetches the next configuration and rung to evaluate.
            - self.start_job(bracket_num, config_name, config_params, resources_to_add, gpu_number): 
            Starts the evaluation of a configuration.
            - get_fitness_scores(save_path, config_name): Retrieves fitness scores for a given configuration.
            - get_current_runs(save_path): Returns the list of currently running jobs.
            - hyperband.get_best_configuration(): Determines the best configuration based on fitness scores.

        Stopping Criteria:
            - The process ends for a bracket once at least one configuration has been fully evaluated 
            (all rungs completed).

        Notes:
            - Progress bars track the percentage of configurations evaluated in each rung.
            - Configurations are selected dynamically based on availability of resources and fitness scores.
            - The method ensures the best configuration is determined only after all runs are complete.

        """
        
        print(f"Number of brackets: {len(self.brackets.keys())}")
        for bracket_num in self.brackets.keys():
            print("\n" + "-" * 150)
            print(f"Bracket {bracket_num}")
            print("Resources per Rung: ", self.brackets[bracket_num]['rungs'])
            print("-" * 150)

            # Initialize rung progress bars
            configs_in_bracket = self.brackets[bracket_num]["configs"]
            total_rungs = len(self.brackets[bracket_num]['rungs']) + 1
            total_configs = len(configs_in_bracket)

            # Persistent progress bars for each rung
            progress_bars = [
                tqdm(total=total_configs, desc=f"Rung {rung_index} Percentage of Total Configurations", position=rung_index, leave=True)
                for rung_index in range(total_rungs)
            ]

            done = False
            while not done:
                
                #Check which GPUs are available, if none then wait
                available_gpus = self.get_available_gpus()
                if len(available_gpus) > 0:
                    gpu_number = available_gpus[0]
                else:
                    time.sleep(1)
                    continue

                # Fetch the next available config
                config_name, config_rung = self.get_job(self.brackets[bracket_num])
                config_params = self.brackets[bracket_num]['configs'][config_name]['params']

                resources = self.brackets[bracket_num]['rungs'][config_rung]
                prev_resources = self.brackets[bracket_num]['rungs'][config_rung - 1] if config_rung > 0 else 0
                resources_to_add = resources - prev_resources

                # Evaluate the fitness score with additional resources
                self.start_job(bracket_num, config_name, config_params, resources_to_add, gpu_number)
                time.sleep(self.time_between_runs) # Give some time for run to start
              
                # Update progress for each rung
                for rung_index in range(total_rungs):
                    configs_in_current_rung = sum(
                        len(get_fitness_scores(self.save_path, config_name)) == rung_index
                        for config_name in configs_in_bracket.keys()
                    )
                    progress_bars[rung_index].n = configs_in_current_rung
                    progress_bars[rung_index].refresh()

                # Evaluate stopping criteria
                num_configs_fully_evaluated = sum(
                    len(get_fitness_scores(self.save_path, config_name)) == len(self.brackets[bracket_num]['rungs'])
                    for config_name in configs_in_bracket.keys()
                )
                if num_configs_fully_evaluated >= 1:
                    done = True

            # Close all progress bars
            for pbar in progress_bars:
                pbar.close()

            # Dont check best config on last bracket till all runs are complete
            if bracket_num == list(self.brackets.keys())[-1]: 
                while len(get_current_runs(self.save_path)) > 0:
                    time.sleep(10)

            # Determine current best configuration
            best_config_name, best_config, best_fitness_score, best_config_bracket = self.get_best_configuration()
            print("\nStopping criteria met: At least one configuration has been fully evaluated.\n")
            print(f"Current Best Configuration is {best_config_name} in Bracket {best_config_bracket} with Fitness Score of {best_fitness_score}")
            for name, value in best_config.items():
                print(f"{name}: {value}")
            print("\n" + "-" * 150)

    def get_best_configuration(self):
        """
        Retrieve the best configuration based on the highest fitness score from 'fitness_scores.csv',
        and use `self.brackets` to get additional information about the configuration.

        Returns:
            tuple: (best_config_name, best_config, best_fitness_score, best_config_bracket), where:
                - best_config_name (str): The name of the configuration with the highest fitness score.
                - best_config (dict): The configuration details from `self.brackets`.
                - best_fitness_score (float): The highest fitness score.
                - best_config_bracket (str): The bracket associated with the best configuration.
        """
        fitness_scores_path = os.path.join(self.save_path, 'fitness_scores.csv')

        best_config_name = None
        best_config = None
        best_fitness_score = float('-inf')
        best_config_bracket = None

        with open(fitness_scores_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for config_name, value in row.items():
                    if config_name == "Bracket":  # Skip any "Bracket" column if it exists
                        continue

                    try:
                        fitness_score = float(value) if value else float('-inf')
                    except ValueError:
                        fitness_score = float('-inf')

                    if fitness_score > best_fitness_score:
                        best_config_name = config_name
                        best_fitness_score = fitness_score

        # Once the best_config_name is determined, search `self.brackets` for details
        if best_config_name:
            for bracket_num, bracket_data in self.brackets.items():
                if best_config_name in bracket_data['configs']:
                    best_config_bracket = bracket_num
                    best_config = bracket_data['configs'][best_config_name]['params']
                    break

        return best_config_name, best_config, best_fitness_score, best_config_bracket

#---------------------------------------------------------------------------------------#

