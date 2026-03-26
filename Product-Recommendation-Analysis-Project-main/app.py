"""
Product Recommendation & Analysis Dashboard
=============================================
A Gradio-based interactive dashboard for Superstore data analysis
and product recommendations using collaborative filtering.

Run:  python app.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1.  DATA LOADING
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

ss_data = pd.read_csv(os.path.join(DATA_DIR, "Superstore-Data.csv"), encoding="latin-1")
if "Unnamed: 0" in ss_data.columns:
    ss_data = ss_data.drop(columns=["Unnamed: 0"])

reviews = pd.read_csv(os.path.join(DATA_DIR, "Superstore-Dataset-Reviews.csv"), encoding="latin-1")
reviews["Retail Price"] = reviews["Sales"] / reviews["Quantity"]

# Parse dates for time-series charts
ss_data["Order Date"] = pd.to_datetime(ss_data["Order Date"], format="mixed", dayfirst=False)
ss_data["Quarter"] = ss_data["Order Date"].dt.to_period("Q").astype(str)

# ──────────────────────────────────────────────
# 2.  RECOMMENDATION MODEL (Collaborative)
# ──────────────────────────────────────────────
pt = reviews.pivot_table(
    index="Product Name",
    columns="Customer ID",
    values="Rate",
    aggfunc="mean",
).fillna(0)

similarity_matrix = cosine_similarity(pt.values)
similarity_df = pd.DataFrame(similarity_matrix, index=pt.index, columns=pt.index)

product_info = reviews.groupby("Product Name").agg(
    Retail_Price=("Retail Price", "mean"),
    Discount=("Discount", "mean"),
    Profit=("Profit", "mean"),
    Sales=("Sales", "mean"),
    Avg_Rating=("Rate", "mean"),
    Num_Rating=("Rate", "count"),
).reset_index()


def recommend_products(product_name, top_n=5):
    """Return top-N similar products based on collaborative filtering."""
    if product_name not in similarity_df.index:
        return pd.DataFrame({"Message": ["Product not found in the dataset."]})

    sim_scores = similarity_df[product_name].sort_values(ascending=False)
    sim_scores = sim_scores.iloc[1 : top_n + 1]
    result_names = sim_scores.index.tolist()

    results = product_info[product_info["Product Name"].isin(result_names)].copy()
    results = results.set_index("Product Name").loc[result_names].reset_index()
    results = results.rename(columns={
        "Product Name": "Product",
        "Retail_Price": "Avg Retail Price ($)",
        "Discount": "Avg Discount",
        "Profit": "Avg Profit ($)",
        "Sales": "Avg Sales ($)",
        "Avg_Rating": "Avg Rating",
        "Num_Rating": "# Ratings",
    })
    for col in ["Avg Retail Price ($)", "Avg Discount", "Avg Profit ($)", "Avg Sales ($)", "Avg Rating"]:
        results[col] = results[col].round(2)
    return results


# ──────────────────────────────────────────────
# 3.  CHART HELPERS  — Professional light theme
# ──────────────────────────────────────────────
# Professional muted chart colors
CHART_BLUE     = "#4472C4"
CHART_ORANGE   = "#ED7D31"
CHART_GREEN    = "#70AD47"
CHART_GRAY     = "#A5A5A5"
CHART_GOLD     = "#FFC000"
CHART_TEAL     = "#5B9BD5"
CHART_RED      = "#C44E52"

REGION_COLORS  = [CHART_BLUE, CHART_ORANGE, CHART_TEAL, CHART_GREEN]
SEGMENT_COLORS = [CHART_BLUE, CHART_ORANGE, CHART_GREEN]
SHIP_COLORS    = [CHART_BLUE, CHART_ORANGE, CHART_GREEN, CHART_GOLD]

# Matplotlib global defaults for a clean look
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "600",
})

FIG_BG   = "#FFFFFF"
AX_BG    = "#FAFAFA"
GRID_CLR = "#E0E0E0"
TEXT_CLR = "#333333"
LABEL_CLR = "#555555"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    """Apply professional light styling to a matplotlib axes."""
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color=TEXT_CLR, fontsize=13, fontweight="600", pad=14, loc="left")
    ax.set_xlabel(xlabel, color=LABEL_CLR, fontsize=9)
    ax.set_ylabel(ylabel, color=LABEL_CLR, fontsize=9)
    ax.tick_params(colors=LABEL_CLR, labelsize=8.5)
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#CCCCCC")
        ax.spines[spine].set_linewidth(0.6)


def _format_large(n):
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:,.2f}"


def build_dashboard(category_filter, segment_filter):
    """Generate all dashboard charts & KPIs based on filters."""
    df = ss_data.copy()

    if category_filter and category_filter != "All":
        df = df[df["Category"] == category_filter]
    if segment_filter and segment_filter != "All":
        df = df[df["Segment"] == segment_filter]

    if df.empty:
        empty_fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data for selected filters", ha="center", va="center",
                color=LABEL_CLR, fontsize=13)
        ax.set_facecolor(AX_BG)
        empty_fig.patch.set_facecolor(FIG_BG)
        return ("0", "0", "0", "0%", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig)

    # ── KPIs ──
    total_sales  = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    total_qty    = df["Quantity"].sum()
    profit_ratio = (total_profit / total_sales * 100) if total_sales else 0

    kpi_sales  = _format_large(total_sales)
    kpi_profit = _format_large(total_profit)
    kpi_qty    = _format_large(total_qty)
    kpi_ratio  = f"{profit_ratio:.1f}%"

    # ── Chart 1: Quarterly Sales Trend ──
    quarterly = df.groupby("Quarter")["Sales"].sum().reset_index().sort_values("Quarter")

    fig1, ax1 = plt.subplots(figsize=(9, 3.5))
    fig1.patch.set_facecolor(FIG_BG)
    ax1.fill_between(range(len(quarterly)), quarterly["Sales"], alpha=0.10, color=CHART_BLUE)
    ax1.plot(range(len(quarterly)), quarterly["Sales"], marker="o", markersize=5,
             color=CHART_BLUE, linewidth=2.2, markerfacecolor="white", markeredgewidth=1.8)
    ax1.set_xticks(range(len(quarterly)))
    ax1.set_xticklabels(quarterly["Quarter"], rotation=45, ha="right", fontsize=7.5)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _format_large(x)))
    _style_ax(ax1, "Quarterly Sales Trend", "Quarter", "Sales ($)")
    fig1.tight_layout()

    # ── Chart 2: Average Sales by Region ──
    region_sales = df.groupby("Region")["Sales"].mean().sort_values(ascending=False).reset_index()

    fig2, ax2 = plt.subplots(figsize=(6, 3.8))
    fig2.patch.set_facecolor(FIG_BG)
    bars2 = ax2.bar(region_sales["Region"], region_sales["Sales"],
                    color=REGION_COLORS[: len(region_sales)], edgecolor="white", linewidth=0.8, width=0.55)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 2, f"${h:,.0f}",
                 ha="center", va="bottom", color=TEXT_CLR, fontsize=8, fontweight="500")
    _style_ax(ax2, "Average Sales by Region", "", "Avg Sales ($)")
    fig2.tight_layout()

    # ── Chart 3: Sub-Category Sales & Profit ──
    subcat = df.groupby("Sub-Category").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
    subcat = subcat.sort_values("Sales", ascending=True)

    fig3, ax3 = plt.subplots(figsize=(7, 4.8))
    fig3.patch.set_facecolor(FIG_BG)
    y_pos = np.arange(len(subcat))
    ax3.barh(y_pos - 0.2, subcat["Sales"], color=CHART_BLUE, height=0.38, label="Sales", edgecolor="white", linewidth=0.5)
    ax3.barh(y_pos + 0.2, subcat["Profit"], color=CHART_GREEN, height=0.38, label="Profit", edgecolor="white", linewidth=0.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(subcat["Sub-Category"], fontsize=8.5)
    ax3.legend(fontsize=8.5, frameon=True, facecolor="white", edgecolor="#ddd",
               loc="lower right")
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _format_large(x)))
    ax3.grid(axis="x", color=GRID_CLR, linewidth=0.6, linestyle="--", alpha=0.7)
    ax3.grid(axis="y", visible=False)
    _style_ax(ax3, "Sales & Profit by Sub-Category", "Amount ($)", "")
    fig3.tight_layout()

    # ── Chart 4: Segment Distribution ──
    seg = df.groupby("Segment")["Sales"].sum().reset_index()

    fig4, ax4 = plt.subplots(figsize=(5, 4))
    fig4.patch.set_facecolor(FIG_BG)
    wedges, texts, autotexts = ax4.pie(
        seg["Sales"], labels=seg["Segment"], autopct="%1.1f%%",
        colors=SEGMENT_COLORS[: len(seg)],
        textprops={"color": TEXT_CLR, "fontsize": 9.5},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        startangle=140,
        pctdistance=0.55,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("600")
    ax4.set_title("Customer Segment Distribution", color=TEXT_CLR, fontsize=13,
                   fontweight="600", pad=14, loc="left")
    fig4.tight_layout()

    # ── Chart 5: Ship Mode — Sales & Profit ──
    ship = df.groupby("Ship Mode").agg({"Sales": "sum", "Profit": "sum"}).reset_index()

    fig5, ax5 = plt.subplots(figsize=(6, 3.8))
    fig5.patch.set_facecolor(FIG_BG)
    x_pos = np.arange(len(ship))
    ax5.bar(x_pos - 0.2, ship["Sales"], color=CHART_BLUE, width=0.38, label="Sales", edgecolor="white", linewidth=0.5)
    ax5.bar(x_pos + 0.2, ship["Profit"], color=CHART_ORANGE, width=0.38, label="Profit", edgecolor="white", linewidth=0.5)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(ship["Ship Mode"], fontsize=8.5)
    ax5.legend(fontsize=8.5, frameon=True, facecolor="white", edgecolor="#ddd")
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _format_large(x)))
    _style_ax(ax5, "Sales & Profit by Ship Mode", "", "Amount ($)")
    fig5.tight_layout()

    return (kpi_sales, kpi_profit, kpi_qty, kpi_ratio, fig1, fig2, fig3, fig4, fig5)


# ──────────────────────────────────────────────
# 4.  GRADIO UI  — Professional light theme
# ──────────────────────────────────────────────
CATEGORIES    = ["All"] + sorted(ss_data["Category"].unique().tolist())
SEGMENTS      = ["All"] + sorted(ss_data["Segment"].unique().tolist())
PRODUCT_NAMES = sorted(pt.index.tolist())

custom_css = """
/* ── KPI cards ── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 22px 16px;
    text-align: center;
    min-height: 100px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}
.kpi-label {
    font-size: 0.78rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
}
.kpi-icon {
    font-size: 1.3rem;
    margin-bottom: 2px;
}

/* ── General polish ── */
.gradio-container { max-width: 1280px !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Product Recommendation & Analysis Dashboard",
    theme=gr.themes.Soft(
        primary_hue="gray",
        secondary_hue="blue",
        neutral_hue="gray",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#f3f4f6",
        block_background_fill="#ffffff",
        block_border_color="#e5e7eb",
        block_border_width="1px",
        block_label_text_color="#374151",
        block_title_text_color="#1f2937",
        input_background_fill="#f9fafb",
        input_border_color="#d1d5db",
        button_primary_background_fill="#374151",
        button_primary_background_fill_hover="#1f2937",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#f3f4f6",
        button_secondary_text_color="#374151",
        button_secondary_border_color="#d1d5db",
    ),
    css=custom_css,
) as demo:

    # Header
    gr.Markdown(
        """
        <div style="text-align:center; padding: 16px 0 6px 0;">
            <h1 style="color:#1f2937; font-size:1.85rem; font-weight:700; margin-bottom:4px;">
                Product Recommendation &amp; Analysis Dashboard
            </h1>
            <p style="color:#6b7280; font-size:0.9rem; margin:0; font-weight:400;">
                Superstore Sales Insights &amp; Collaborative Filtering Recommendations
            </p>
        </div>
        """
    )

    with gr.Tabs():
        # ───── TAB 1: SALES DASHBOARD ─────
        with gr.Tab("Sales Dashboard"):
            with gr.Row():
                cat_dd = gr.Dropdown(choices=CATEGORIES, value="All", label="Category",
                                     interactive=True, scale=1)
                seg_dd = gr.Dropdown(choices=SEGMENTS, value="All", label="Segment",
                                     interactive=True, scale=1)
                refresh_btn = gr.Button("Refresh Dashboard", variant="primary", scale=1)

            # KPI row
            with gr.Row(equal_height=True):
                kpi_sales  = gr.HTML('<div class="kpi-card"><div class="kpi-icon">💰</div><div class="kpi-value">—</div><div class="kpi-label">Total Sales</div></div>')
                kpi_profit = gr.HTML('<div class="kpi-card"><div class="kpi-icon">📈</div><div class="kpi-value">—</div><div class="kpi-label">Total Profit</div></div>')
                kpi_qty    = gr.HTML('<div class="kpi-card"><div class="kpi-icon">📦</div><div class="kpi-value">—</div><div class="kpi-label">Total Quantity</div></div>')
                kpi_ratio  = gr.HTML('<div class="kpi-card"><div class="kpi-icon">📊</div><div class="kpi-value">—</div><div class="kpi-label">Profit / Sales Ratio</div></div>')

            with gr.Row():
                chart_quarterly = gr.Plot(label="Quarterly Sales Trend")

            with gr.Row():
                chart_region  = gr.Plot(label="Average Sales by Region")
                chart_segment = gr.Plot(label="Customer Segment Distribution")

            with gr.Row():
                chart_subcat = gr.Plot(label="Sales & Profit by Sub-Category")

            with gr.Row():
                chart_ship = gr.Plot(label="Sales & Profit by Ship Mode")

            def _on_refresh(cat, seg):
                ks, kp, kq, kr, f1, f2, f3, f4, f5 = build_dashboard(cat, seg)
                return (
                    f'<div class="kpi-card"><div class="kpi-icon">💰</div><div class="kpi-value">${ks}</div><div class="kpi-label">Total Sales</div></div>',
                    f'<div class="kpi-card"><div class="kpi-icon">📈</div><div class="kpi-value">${kp}</div><div class="kpi-label">Total Profit</div></div>',
                    f'<div class="kpi-card"><div class="kpi-icon">📦</div><div class="kpi-value">{kq}</div><div class="kpi-label">Total Quantity</div></div>',
                    f'<div class="kpi-card"><div class="kpi-icon">📊</div><div class="kpi-value">{kr}</div><div class="kpi-label">Profit / Sales Ratio</div></div>',
                    f1, f2, f3, f4, f5,
                )

            refresh_btn.click(
                fn=_on_refresh,
                inputs=[cat_dd, seg_dd],
                outputs=[kpi_sales, kpi_profit, kpi_qty, kpi_ratio,
                         chart_quarterly, chart_region, chart_subcat, chart_segment, chart_ship],
            )

            demo.load(
                fn=_on_refresh,
                inputs=[cat_dd, seg_dd],
                outputs=[kpi_sales, kpi_profit, kpi_qty, kpi_ratio,
                         chart_quarterly, chart_region, chart_subcat, chart_segment, chart_ship],
            )

        # ───── TAB 2: RECOMMENDATIONS ─────
        with gr.Tab("Product Recommendations"):
            gr.Markdown(
                """
                <div style="text-align:center; padding: 10px 0;">
                    <h2 style="color:#1f2937; font-weight:600; margin-bottom:4px; font-size:1.3rem;">
                        Collaborative Filtering Recommendations
                    </h2>
                    <p style="color:#6b7280; font-size:0.88rem;">
                        Select a product to discover similar items based on customer rating patterns.
                    </p>
                </div>
                """
            )

            with gr.Row():
                product_dd = gr.Dropdown(
                    choices=PRODUCT_NAMES,
                    label="Select a Product",
                    interactive=True,
                    scale=3,
                )
                rec_btn   = gr.Button("Recommend", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)

            rec_output = gr.Dataframe(
                label="Recommended Products",
                interactive=False,
                wrap=True,
            )

            rec_btn.click(fn=recommend_products, inputs=product_dd, outputs=rec_output)
            clear_btn.click(fn=lambda: (None, pd.DataFrame()), inputs=[], outputs=[product_dd, rec_output])


# ──────────────────────────────────────────────
# 5.  LAUNCH
# ──────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
