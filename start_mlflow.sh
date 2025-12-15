#!/bin/bash
# Start MLflow UI in background

echo "Starting MLflow UI..."

# Kill existing MLflow UI if running
pkill -f "mlflow ui"

# Start MLflow UI in background
nohup mlflow ui --host 0.0.0.0 --port 5000 > mlflow_ui.log 2>&1 &

echo "MLflow UI started on http://localhost:5000"
echo "Logs: mlflow_ui.log"
echo ""
echo "To stop: pkill -f 'mlflow ui'"
