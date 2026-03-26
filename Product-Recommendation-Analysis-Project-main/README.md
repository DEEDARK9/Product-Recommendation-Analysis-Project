# Product Recommendation & Analysis Project

A data-driven project that combines **SQL analytics**, **exploratory data analysis**, **machine learning-based product recommendations**, and an **interactive Gradio dashboard** to deliver actionable business insights for an e-commerce superstore.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Process](#project-process)
- [Directory Structure](#directory-structure)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Running the Dashboard](#running-the-dashboard)
- [Dashboard Features](#dashboard-features)
- [Recommendation Model](#recommendation-model)
- [SQL Analysis](#sql-analysis)
- [Key Insights](#key-insights)
- [Recommendations](#recommendations)
- [Conclusion](#conclusion)

---

## Problem Statement

The challenge was to design a recommendation system that accurately predicts products a customer might want to purchase, helping businesses enhance customer satisfaction and profitability. Additionally, the aim was to identify market trends and customer behavior to optimize sales strategies and profitability in key segments.

---

## Project Process

1. **Data Cleaning** — Cleaned and preprocessed 10K customer records using **Excel** to ensure consistency and accuracy in the dataset.
2. **Data Warehouse Architecture** — Built a **SQL**-based data warehouse to organize and store customer and sales data efficiently.
3. **Trend Analysis** — Used **SQL queries** to analyze customer behavior and sales trends.
4. **Dashboard Creation** — Created a **Tableau** dashboard to visualize Quarterly Sales Forecasting and state-wise sales/profit distribution.
5. **Model Building** — Implemented **Python NLP models** to create Collaborative and Popularity-Based Recommendation Systems for personalized item suggestions.
6. **Model Deployment** — Utilized **Gradio** to deploy the Collaborative-Based Recommendation System with an interactive analysis dashboard.
7. **Analysis and Reporting** — Extracted insights from the data to provide actionable recommendations for improving business strategy and customer engagement.

---

## Directory Structure

```
Product-Recommendation-Analysis-Project/
│
├── data/
│   ├── Superstore-Data.csv                 # Sales and order data (9,994 records)
│   ├── Superstore-Dataset-Reviews.csv      # Customer ratings, reviews, and summaries
│   └── Features_Target_Description.txt     # Column descriptions
│
├── notebooks/
│   ├── Product-Recommendation-Project.ipynb  # Full pipeline: EDA + Model Building
│   └── Products-Analysis.sql                 # SQL queries for business analysis
│
├── app.py                # Gradio dashboard application
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Dataset

| File | Records | Description |
|------|---------|-------------|
| `Superstore-Data.csv` | 9,994 | Order details — sales, profit, quantity, discounts, shipping, region, segment, category |
| `Superstore-Dataset-Reviews.csv` | 9,994 | Enriched dataset with customer ratings (1–5), review text, and summary for each order |

**Key Columns:** `Order ID`, `Customer ID`, `Product Name`, `Category`, `Sub-Category`, `Sales`, `Profit`, `Quantity`, `Discount`, `Region`, `Segment`, `Ship Mode`, `Rate`, `Review`, `Summary`

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Data Cleaning | Excel |
| Data Warehouse | SQL |
| Data Analysis | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Tableau |
| ML Model | Scikit-learn (Cosine Similarity) |
| Dashboard & Deployment | Gradio |

---

## Setup & Installation

**Prerequisites:** Python 3.8+

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DEEDARK9/Product-Recommendation-Analysis-Project.git
   cd Product-Recommendation-Analysis-Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Dashboard

```bash
python app.py
```

The app launches at **http://localhost:7860**. Open this URL in your browser.

---

## Dashboard Features

### Tab 1 — Sales Dashboard

| Feature | Description |
|---------|-------------|
| **KPI Cards** | Total Sales, Total Profit, Total Quantity, Profit-to-Sales Ratio |
| **Quarterly Sales Trend** | Line chart showing sales over time by quarter |
| **Sales by Region** | Bar chart comparing average sales across 4 regions |
| **Sub-Category Analysis** | Horizontal bar chart of sales and profit by sub-category |
| **Segment Distribution** | Pie chart of customer segment composition |
| **Ship Mode Breakdown** | Grouped bar chart of sales and profit by shipping method |
| **Interactive Filters** | Filter all charts by Category and Segment |

### Tab 2 — Product Recommendations

- Select any product from a dropdown of 1,850 products.
- Click **Recommend** to get the top 5 similar products.
- Results include: Average Retail Price, Discount, Profit, Sales, and Rating.
- Uses collaborative filtering (cosine similarity on customer–product rating matrix).

---

## Recommendation Model

The recommendation engine uses **item-based collaborative filtering**:

1. **Pivot Table** — Rows = Product Names, Columns = Customer IDs, Values = Average Ratings.
2. **Cosine Similarity** — Computes pairwise similarity between all products based on their rating vectors.
3. **Top-N Retrieval** — For a selected product, returns the most similar products ranked by similarity score.

This approach recommends products that share similar rating patterns across customers, regardless of category or price.

---

## SQL Analysis

The `Products-Analysis.sql` file contains business queries covering:

- Customer segmentation and profitability
- Regional sales performance
- Product category trends
- Quarterly sales forecasting
- Market share analysis

---

## Key Insights

1. **Consumer Segment** is the most profitable customer segment, generating the highest revenue and profit.
2. **California** emerged as the largest and most profitable market.
3. **North Dakota** represents an underperforming market requiring strategic attention.
4. **Q4 Sales** show a forecasted increase, presenting growth opportunities for targeted marketing.

---

## Recommendations

1. **Capitalize on Q4 Growth** — Allocate additional marketing resources during the anticipated Q4 sales increase.
2. **Focus on Consumers** — Develop targeted campaigns and personalized promotions for the Consumer segment.
3. **Invest in California** — Continue focused marketing to maintain and grow the top-performing market.
4. **Improve Underperformers** — Conduct market research in underperforming regions like North Dakota to tailor strategies.

---

## Assumptions & Caveats

- The data is assumed to accurately reflect customer preferences and purchasing patterns.
- Sales forecasts are based on historical trends and may be affected by external factors.
- Recommendation model effectiveness may vary as customer preferences shift or for products with limited rating data.

---

## Conclusion

This project demonstrated the importance of data-driven decision-making through the implementation of a SQL-based Superstore Data Warehouse. Insights from 10,000 customer records helped identify key growth areas, profitable segments, and underperforming markets. The integration of Python-based recommendation models further enhances customer experience by offering personalized product suggestions, while the interactive Gradio dashboard provides a professional interface for real-time analysis and exploration.
