$ConfigDir = ".\configs\active"
$CompletedDir = ".\configs\completed"

# Check if active config directory exists
if (-not (Test-Path -Path $ConfigDir)) {
    Write-Error "Error: Directory $ConfigDir does not exist."
    exit 1
}

# Create completed directory if it doesn't exist
if (-not (Test-Path -Path $CompletedDir)) {
    New-Item -ItemType Directory -Path $CompletedDir -Force | Out-Null
}

# Get all .yaml files
$ConfigFiles = Get-ChildItem -Path $ConfigDir -Filter "*.yaml"

# Check if we actually found any files
if ($ConfigFiles.Count -eq 0 -or $ConfigFiles -eq $null) {
    Write-Host "No .yaml files found in $ConfigDir"
    exit 0
}

# Iterate over each config file
foreach ($file in $ConfigFiles) {
    $config_path = $file.FullName
    $config_name = $file.Name
    
    Write-Host ""
    Write-Host "Running with config: $config_name"
    Write-Host "================================================"
    Write-Host ""
    
    # Execute the python script
    python train.py --config "$config_path"
    
    # Check the exit code of the external executable
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Training with $config_name failed with exit code $LASTEXITCODE." -ForegroundColor Yellow
    } else {
        Write-Host "Successfully completed: $config_name" -ForegroundColor Green
        Move-Item -Path $config_path -Destination $CompletedDir -Force
    }
}
