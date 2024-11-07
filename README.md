# Anushka-ML 🤖

A Telegram bot with ML capabilities, powered by FastAPI backend. Train models, make predictions, and manage ML models directly through Telegram.

## Features 🌟

- Train ML models (Random Forest, SVM) through Telegram
- Make predictions using trained models
- Support for Excel (.xlsx) and CSV (.csv) files
- Interactive model management
- Real-time training and prediction status updates
- Maximum active models limit (configurable)
- Rate limiting and security features

## Quick Start 🚀

1. Clone and setup:
```bash
git clone https://github.com/yourusername/anushka-ml.git
cd anushka-ml
```

2. Set your settings in `config/bot_config.yaml` and `config/api_config.yaml`

3. Run with Docker:
```bash
docker-compose -f docker-compose.prod.yml up --build --force-recreate
```

Or run locally (need to install python requirements.txt):
### Start API server and Telegram bot:
```bash
./start.sh
```

## Check API Server Example 📊

To check the API server, use `api_client_example.py` script:

Change API base URL in the script to your API server URL if needed.
```python
base_url = "http://localhost:8000/api/v1"
```

Run the script:
```bash
python -m examples.api_client_example
```
This will train a model and make predictions using the API with the self explanatory output console logs.

## Telegram Bot Test Datasets 📊

Example datasets in `examples/` folder:
- `dataset.xlsx` - Training data (features + label column)
- `prediction_data.xlsx` - Prediction data (features only)

Generate new test data:
```bash
python -m examples.create_dataset
```