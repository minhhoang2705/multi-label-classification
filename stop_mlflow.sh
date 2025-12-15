#!/bin/bash
# Stop MLflow UI

echo "Stopping MLflow UI..."
pkill -f "mlflow ui"
echo "MLflow UI stopped"
