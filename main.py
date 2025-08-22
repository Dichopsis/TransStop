import subprocess
import sys
import os

def run_script(script_path: str):
    """
    Executes a given Python script using the same Python interpreter
    that is running this main script. This ensures environment consistency.

    Args:
        script_path (str): The relative path to the Python script to execute.

    Raises:
        SystemExit: If the script is not found or fails to execute.
    """
    
    if not os.path.exists(script_path):
        print(f"ERROR: The script '{script_path}' was not found.")
        sys.exit(1)

    python_executable = sys.executable
    
    print(f"\n{'='*60}")
    print(f"RUNNING SCRIPT: {script_path}")
    print(f"{'='*60}")

    try:
        subprocess.run([python_executable, script_path], check=True, text=True)
        print(f"\n--- SCRIPT '{script_path}' COMPLETED SUCCESSFULLY. ---")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Script '{script_path}' failed with exit code {e.returncode}.")
        print("The pipeline will be stopped.")
        sys.exit(1) # Stop the entire pipeline on failure
    except KeyboardInterrupt:
        print("\nPipeline execution was manually interrupted by the user.")
        sys.exit(1)

if __name__ == "__main__":
    # Define the sequence of scripts to be executed.
    # If you place your scripts in a subdirectory (e.g., 'scripts/'),
    # adjust the paths accordingly (e.g., "scripts/hyperparam_and_final_training.py").
    pipeline_steps = [
        "scripts/download_rds_files.py",
        "scripts/filter.py",
        "scripts/data_preparation.py",
        "scripts/download_models.py",
        "scripts/systematic_evaluation.py",
        "scripts/hyperparam_and_final_training.py",
        "scripts/reconstruct_and_save_map.py",
        "scripts/analysis_and_reporting.py",
        "scripts/genome_wide_inference.py",
        "scripts/genome_prediction_analysis.py",
        "scripts/combine_figures.py"
    ]

    print("STARTING THE FULL ANALYSIS PIPELINE...")

    for step in pipeline_steps:
        run_script(step)

    print(f"\n{'='*60}")
    print(" FULL PIPELINE COMPLETED SUCCESSFULLY! ")
    print(f"All results are available in the ./results/ directory.")
    print(f"{'='*60}")