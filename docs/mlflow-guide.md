# MLflow Tracking Guide

## Quick Start

### Start MLflow UI

```bash
./start_mlflow.sh
```

Then open: http://localhost:5000

### Stop MLflow UI

```bash
./stop_mlflow.sh
```

## Manual Commands

### Start MLflow UI (Foreground)
```bash
mlflow ui
```

### Start MLflow UI (Background)
```bash
nohup mlflow ui --host 0.0.0.0 --port 5000 > mlflow_ui.log 2>&1 &
```

### Check if Running
```bash
pgrep -f "mlflow ui"
# Or check the port
lsof -i :5000
```

### Stop MLflow UI
```bash
pkill -f "mlflow ui"
```

## Access URLs

- **Local machine:** http://localhost:5000
- **Remote access:** http://YOUR_SERVER_IP:5000
- **Custom port:** Change `5000` to your port

## MLflow UI Features

### 1. Experiments View
- See all experiments and runs
- Compare metrics across runs
- Filter and sort by metrics

### 2. Run Details
Click on any run to see:
- **Parameters:** batch_size, lr, model_name, etc.
- **Metrics:** train_loss, val_accuracy, val_macro_f1, etc.
- **Artifacts:** model checkpoints, plots, logs

### 3. Compare Runs
- Select multiple runs (checkbox)
- Click "Compare" button
- View side-by-side metrics and parameters

### 4. Visualizations
- Metric plots over time
- Parallel coordinates plot
- Scatter plot matrix

## Useful MLflow Commands

### List Experiments
```bash
mlflow experiments list
```

### Search Runs
```bash
mlflow runs list --experiment-id 274365810452024205
```

### Delete Runs
```bash
mlflow runs delete --run-id <run_id>
```

### Clean Up Old Runs
```bash
# Delete all runs older than 30 days
mlflow gc --backend-store-uri ./mlruns
```

## Project Structure

```
mlruns/
├── 274365810452024205/          # Experiment ID
│   ├── <run_id_1>/             # Run folder
│   │   ├── meta.yaml           # Run metadata
│   │   ├── metrics/            # Metric files
│   │   ├── params/             # Parameter files
│   │   ├── tags/               # Tag files
│   │   └── artifacts/          # Artifacts (models, plots)
│   ├── <run_id_2>/
│   └── meta.yaml               # Experiment metadata
└── .trash/                      # Deleted runs
```

## Tips

### 1. Filter Runs in UI
Use the search bar:
```
metrics.val_macro_f1 > 0.8
params.batch_size = "64"
```

### 2. Sort by Best Metric
Click column headers to sort (e.g., "val_macro_f1")

### 3. Download Run Data
- Click on run
- Go to "Artifacts" tab
- Download checkpoints or logs

### 4. Track Custom Metrics
In your code:
```python
import mlflow

mlflow.log_metric("custom_metric", value, step=epoch)
mlflow.log_param("custom_param", value)
mlflow.log_artifact("path/to/file")
```

### 5. Set Run Name
```python
with mlflow.start_run(run_name="my_experiment_v1"):
    # Training code
    pass
```

## Current Experiment

- **Experiment Name:** cat_breeds_classification
- **Experiment ID:** 274365810452024205
- **Tracked Metrics:**
  - train_loss, train_accuracy
  - val_loss, val_accuracy, val_balanced_accuracy
  - val_macro_f1, val_precision_macro, val_recall_macro
  - val_f1_weighted, val_precision_weighted, val_recall_weighted
  - val_top_3_accuracy, val_top_5_accuracy

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000
# Kill it
kill -9 <PID>
# Or use different port
mlflow ui --port 5001
```

### Can't Access Remote MLflow UI
Make sure to:
1. Use `--host 0.0.0.0` when starting
2. Open firewall port (if needed)
3. Use server's IP address in browser

### Logs Not Showing
Check the log file:
```bash
cat mlflow_ui.log
```

## Advanced: Remote Tracking Server

For team collaboration, set up a remote tracking server:

```bash
# On server
mlflow server \
  --backend-store-uri sqlite:///mlruns.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

In training code:
```python
import mlflow
mlflow.set_tracking_uri("http://server_ip:5000")
```
