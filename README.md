# Interactive Logistics Demand Analytics Dashboard

An interactive logistics analytics dashboard built with Python and Streamlit to
explore delivery demand patterns, order composition, operational activity, and
relationships between logistics variables using a real-world dataset.

🔗 Live site: https://interactivelogisticsdemanddashboard.streamlit.app/

## Features

- **Dashboard Overview** — KPI summary, demand distribution, weekday demand
  trends, urgent vs. non-urgent order comparison, order type breakdown,
  operational summary table, and a correlation heatmap.
- **Logistics Demand Profiles** — K-Means clustering (K=3) on all operational
  order categories to group days into Low / Normal / High demand profiles,
  with a PCA visualization and per-profile operational characteristics.
- Dark / Light theme toggle with a muted, card-based UI.
- Sidebar filters for Week of Month and Day of Week that apply across both tabs.

## Dataset

This project uses the **Daily Demand Forecasting Orders** dataset (UCI Machine
Learning Repository), which contains daily order volumes for a Brazilian
logistics company across several operational categories.

## Project Structure

```
.
├── dashboard/
│   └── app.py              # Streamlit application
├── data/
│   └── Daily_Demand_Forecasting_Orders.csv
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure the dataset is present at `data/Daily_Demand_Forecasting_Orders.csv`.

5. Run the dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

## Tech Stack

- [Streamlit](https://streamlit.io/) — dashboard framework
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data processing
- [Plotly](https://plotly.com/python/) — interactive charts
- [scikit-learn](https://scikit-learn.org/) — K-Means clustering, PCA, feature scaling
