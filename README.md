# Walmart Capstone Project

**Description:** Time-series style regression on Walmart weekly sales. The sample `main.py` aggregates weekly sales and trains a simple Linear Regression model using lag features.

**Files**
- `main.py` — expects `data/walmart_sales.csv` with at least `date` and `weekly_sales` columns.
- `requirements.txt` — Python dependencies.
- `data/` — place `walmart_sales.csv` here before running.

**Run**
```bash
pip install -r requirements.txt
python main.py
```
