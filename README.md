# Financial Literacy & Allowance Expenditure Analysis

**Streamlit Web Application** for Mini Project – Module V: Statistical Techniques (Bivariate Curve Fitting)

## Features

- **Survey Data**: Realistic N ≥ 35 student allowance vs savings dataset
- **Normal Equations**: Full summation tables (∑X, ∑Y, ∑X², ∑X³, ∑X⁴, ∑XY, ∑X²Y)
- **Curve Fitting**: Linear (Y = a + bX) and Quadratic (Y = a + bX + cX²) via `np.linalg.solve`
- **Model Comparison**: R² and Standard Error of Estimate (Sᵧ|ₓ) with automatic best-model selection
- **Interactive Predictor**: Input allowance → predicted savings + banking product recommendations
- **Industry Insights**: Banking & fintech applications of the model

## How to Run

```bash
cd financial_literacy_app
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Evaluation Criteria Covered

1. **Mathematical Accuracy** – Normal equations, least squares for linear & quadratic, error metrics
2. **Code Structure & Web App** – Matrix solving, visual curve comparison, model selection, interactive Streamlit GUI
3. **Industry Relevance** – Personal finance modeling, deposit prediction, consumer behaviour analytics
4. **Presentation** – Clear model selection defense (R² vs Sᵧ|ₓ), curve interpretation, live predictions

## Tech Stack

- Streamlit (interactive GUI)
- NumPy (matrix algebra / least squares)
- Pandas (data handling)
- Matplotlib + Seaborn (visualizations)
