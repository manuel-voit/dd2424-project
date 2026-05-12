import os
import shutil
import subprocess


def migrate_to_dagshub():
    LOCAL_MLRUNS_DIR = "file:./mlruns"
    EXPERIMENT_NAME = "<experiment_name>"
    
    # Your DagsHub repository details
    DAGSHUB_REPO_OWNER = "manuel.voit"
    DAGSHUB_REPO_NAME = "dd2424-project"
    DAGSHUB_TOKEN = "<your_dagshub_access_token>"
    

    DAGSHUB_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
    TEMP_DIR = "./mlflow_temp_export_dir"

    print(f"--- Starting Migration for Experiment: '{EXPERIMENT_NAME}' ---")

    # --- Step 1: Export from Local ---
    print(f"\n[1/3] Exporting from local directory: {LOCAL_MLRUNS_DIR}")
    
    # Set environment to point to local folder
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = LOCAL_MLRUNS_DIR
    
    # Execute the export using the CLI command
    try:
        subprocess.run([
            "export-experiment", 
            "--experiment", EXPERIMENT_NAME, 
            "--output-dir", TEMP_DIR
        ], env=env, check=True)
        print("Local export complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error during export: {e}")
        return

    # --- Step 2: Import to DagsHub ---
    print(f"\n[2/3] Importing to DagsHub tracking server: {DAGSHUB_TRACKING_URI}")
    
    # Switch environment variables to DagsHub
    env["MLFLOW_TRACKING_URI"] = DAGSHUB_TRACKING_URI
    env["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER
    env["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    # Execute the import using the CLI command
    try:
        subprocess.run([
            "import-experiment", 
            "--experiment-name", EXPERIMENT_NAME, 
            "--input-dir", TEMP_DIR
        ], env=env, check=True)
        print("DagsHub import complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error during import: {e}")
        return

    # --- Step 3: Cleanup ---
    print("\n[3/3] Cleaning up temporary files...")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f"Removed temporary directory: {TEMP_DIR}")

    print("\n--- Migration Successfully Completed! ---")
    print(f"You can view your runs at: https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/experiments")

if __name__ == "__main__":
    migrate_to_dagshub()
