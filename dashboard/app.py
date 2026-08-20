# Import libraries required for the logistics dashboard
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configure the Streamlit dashboard page
st.set_page_config(
    page_title="Interactive_Logistics_Dashboard",
    page_icon="🚚",
    layout="wide"
)


# Theme tokens (muted / desaturated accents in both modes)
THEMES = {
    "Dark": {
        "bg": "#0E0E10",
        "card": "#18181B",
        "card_alt": "#1D1D20",
        "border": "#27272A",
        "text": "#E4E4E7",
        "text_muted": "#8A8A93",
        "accent_blue": "#6C8CAA",
        "accent_amber": "#C9A96B",
        "accent_rose": "#BD8484",
        "accent_sage": "#82A695",
        "plotly_template": "plotly_dark",
    },
    "Light": {
        "bg": "#F7F7F6",
        "card": "#FFFFFF",
        "card_alt": "#F1F1EF",
        "border": "#E3E3E0",
        "text": "#1C1C1F",
        "text_muted": "#75757C",
        "accent_blue": "#5B7C99",
        "accent_amber": "#B08F52",
        "accent_rose": "#A66B6B",
        "accent_sage": "#5F8A78",
        "plotly_template": "plotly_white",
    },
}


# Sidebar: theme toggle + filters
st.sidebar.header("Appearance")
theme_choice = st.sidebar.radio(
    "Theme",
    options=["Dark", "Light"],
    index=0,
    horizontal=True
)
T = THEMES[theme_choice]

# Muted, theme-aware palette used consistently across every chart
PALETTE = [T["accent_blue"], T["accent_amber"], T["accent_rose"], T["accent_sage"]]
DEMAND_COLORS = {
    "Low Demand": T["accent_blue"],
    "Normal Demand": T["accent_amber"],
    "High Demand": T["accent_rose"],
}


# Global CSS — card system, typography, muted styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {T['bg']};
        color: {T['text']};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {T['card']};
        border-right: 1px solid {T['border']};
    }}

    h1, h2, h3, h4, h5, p, span, label, div {{
        color: {T['text']};
    }}

    /* Banner */
    .banner {{
        background: linear-gradient(90deg, {T['card_alt']} 0%, {T['card']} 100%);
        border: 1px solid {T['border']};
        border-radius: 10px;
        padding: 10px 18px;
        margin-bottom: 18px;
        font-size: 0.85rem;
        color: {T['text_muted']};
    }}

    /* KPI / metric cards */
    .metric-card {{
        background-color: {T['card']};
        border: 1px solid {T['border']};
        border-radius: 14px;
        padding: 18px 20px;
        height: 100%;
    }}
    .metric-label {{
        font-size: 0.78rem;
        color: {T['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 1.9rem;
        font-weight: 700;
        color: {T['text']};
        line-height: 1.1;
    }}
    .metric-sub {{
        font-size: 0.75rem;
        color: {T['text_muted']};
        margin-top: 4px;
    }}

    /* Chart / content cards */
    .chart-card {{
        background-color: {T['card']};
        border: 1px solid {T['border']};
        border-radius: 14px;
        padding: 18px 20px 6px 20px;
        margin-bottom: 20px;
    }}
    .chart-title {{
        font-size: 1rem;
        font-weight: 600;
        color: {T['text']};
        margin-bottom: 2px;
    }}
    .chart-caption {{
        font-size: 0.82rem;
        color: {T['text_muted']};
        margin-bottom: 10px;
    }}

    /* Section eyebrow */
    .section-eyebrow {{
        font-size: 0.78rem;
        color: {T['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 6px;
        margin-bottom: 2px;
    }}
    .section-title {{
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 4px;
    }}
    .section-desc {{
        font-size: 0.88rem;
        color: {T['text_muted']};
        margin-bottom: 16px;
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        color: {T['text_muted']} !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {T['text']} !important;
        border-bottom-color: {T['accent_amber']} !important;
    }}

    /* Dataframe container */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {T['border']};
        border-radius: 10px;
        overflow: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


def metric_card(label, value, sub=""):
    """Render a single KPI stat as a bordered card (matches the reference layout)."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def chart_card_open(title, caption):
    st.markdown(
        f"""
        <div class="chart-card">
            <div class="chart-title">{title}</div>
            <div class="chart-caption">{caption}</div>
        """,
        unsafe_allow_html=True
    )


def chart_card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def style_fig(fig, height=360):
    """Apply consistent muted theming to every Plotly figure."""
    fig.update_layout(
        template=T["plotly_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text_muted"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T["text"])),
    )
    fig.update_xaxes(gridcolor=T["border"], zerolinecolor=T["border"])
    fig.update_yaxes(gridcolor=T["border"], zerolinecolor=T["border"])
    return fig



# Load the logistics dataset
current_dir = Path(__file__).parent
data_path = current_dir.parent / "data" / "Daily_Demand_Forecasting_Orders.csv"
df = pd.read_csv(data_path, sep=';')

df.columns = [
    "week_of_month",
    "day_of_week",
    "non_urgent_orders",
    "urgent_orders",
    "order_type_a",
    "order_type_b",
    "order_type_c",
    "fiscal_sector_orders",
    "traffic_controller_orders",
    "banking_orders_1",
    "banking_orders_2",
    "banking_orders_3",
    "total_orders"
]


# Header
st.title("Interactive Logistics Demand Analytics Dashboard")
st.markdown(
    """
    <div class="banner">
    Explore logistics order patterns, operational demand, and order activity
    across different weekdays and weeks of the month.
    </div>
    """,
    unsafe_allow_html=True
)


# Sidebar filters
st.sidebar.header("Dashboard Filters")
st.sidebar.write(
    "Narrow down the dataset below. All visuals on the dashboard "
    "will reflect only the days that match your selections."
)

selected_weeks = st.sidebar.multiselect(
    "Week of Month",
    options=sorted(df["week_of_month"].unique()),
    default=sorted(df["week_of_month"].unique())
)

selected_days = st.sidebar.multiselect(
    "Day of Week",
    options=sorted(df["day_of_week"].unique()),
    default=sorted(df["day_of_week"].unique())
)

filtered_df = df[
    df["week_of_month"].isin(selected_weeks)
    & df["day_of_week"].isin(selected_days)
]

total_orders = filtered_df["total_orders"].sum()
average_orders = filtered_df["total_orders"].mean()
maximum_orders = filtered_df["total_orders"].max()
number_of_days = len(filtered_df)


# Tabsls

tab_overview, tab_clusters = st.tabs([
    "📊 Dashboard Overview",
    "🧩 Logistics Demand Profiles"
])

with tab_overview:

    # --- KPI row ---
    st.markdown('<div class="section-eyebrow">Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Key Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Orders", f"{total_orders:,.0f}", "Sum across filtered days")
    with col2:
        metric_card("Average Orders", f"{average_orders:,.2f}", "Per day, filtered range")
    with col3:
        metric_card("Peak Orders", f"{maximum_orders:,.2f}", "Busiest single day")
    with col4:
        metric_card("Observations", f"{number_of_days}", "Days in current filter")

    st.write("")

    # --- Two-column chart row: distribution + weekday averages ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        chart_card_open(
            "📊 Demand Distribution",
            "Spread of total daily orders across the filtered range."
        )
        fig = go.Figure(
            data=[go.Histogram(
                x=filtered_df["total_orders"],
                nbinsx=10,
                marker_color=T["accent_blue"],
                marker_line_color=T["border"],
                marker_line_width=1,
            )]
        )
        fig.update_layout(xaxis_title="Total Orders", yaxis_title="Frequency")
        st.plotly_chart(style_fig(fig), width='stretch', config={"displayModeBar": False})
        chart_card_close()

    with chart_col2:
        weekday_summary = (
            filtered_df.groupby("day_of_week")["total_orders"].mean().reset_index()
        )
        chart_card_open(
            "📅 Average Demand by Weekday",
            "Which weekdays run busiest, useful for staffing plans."
        )
        fig = go.Figure(
            data=[go.Bar(
                x=weekday_summary["day_of_week"].astype(str),
                y=weekday_summary["total_orders"],
                marker_color=T["accent_amber"],
            )]
        )
        fig.update_layout(xaxis_title="Day of Week", yaxis_title="Average Total Orders")
        st.plotly_chart(style_fig(fig), width='stretch', config={"displayModeBar": False})
        chart_card_close()

    # --- Two-column chart row: urgent/non-urgent + order types ---
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        chart_card_open(
            "📦 Urgent vs Non-Urgent Orders",
            "A high urgent share may call for faster rush-order capacity."
        )
        order_comparison = pd.DataFrame({
            "Order Category": ["Urgent Orders", "Non-Urgent Orders"],
            "Average Orders": [
                filtered_df["urgent_orders"].mean(),
                filtered_df["non_urgent_orders"].mean()
            ]
        })
        fig = go.Figure(
            data=[go.Bar(
                x=order_comparison["Order Category"],
                y=order_comparison["Average Orders"],
                marker_color=[T["accent_rose"], T["accent_sage"]],
            )]
        )
        fig.update_layout(xaxis_title="", yaxis_title="Average Orders")
        st.plotly_chart(style_fig(fig), width='stretch', config={"displayModeBar": False})
        chart_card_close()

    with chart_col4:
        chart_card_open(
            "📦 Order Type Analysis",
            "Average demand contributed by each order type (A, B, C)."
        )
        order_type_summary = pd.DataFrame({
            "Order Type": ["Order Type A", "Order Type B", "Order Type C"],
            "Average Orders": [
                filtered_df["order_type_a"].mean(),
                filtered_df["order_type_b"].mean(),
                filtered_df["order_type_c"].mean()
            ]
        })
        fig = go.Figure(
            data=[go.Bar(
                x=order_type_summary["Order Type"],
                y=order_type_summary["Average Orders"],
                marker_color=PALETTE[:3],
            )]
        )
        fig.update_layout(xaxis_title="", yaxis_title="Average Orders")
        st.plotly_chart(style_fig(fig), width='stretch', config={"displayModeBar": False})
        chart_card_close()

    # --- Operational summary table ---
    chart_card_open(
        "🏢 Operational Demand Summary",
        "Average order volume for every operational category, side by side."
    )
    operational_summary = pd.DataFrame({
        "Operational Category": [
            "Urgent Orders", "Non-Urgent Orders", "Banking Orders 1",
            "Banking Orders 2", "Banking Orders 3",
            "Traffic Controller Orders", "Fiscal Sector Orders"
        ],
        "Average": [
            filtered_df["urgent_orders"].mean(),
            filtered_df["non_urgent_orders"].mean(),
            filtered_df["banking_orders_1"].mean(),
            filtered_df["banking_orders_2"].mean(),
            filtered_df["banking_orders_3"].mean(),
            filtered_df["traffic_controller_orders"].mean(),
            filtered_df["fiscal_sector_orders"].mean()
        ]
    })
    operational_summary["Average"] = operational_summary["Average"].round(2)
    st.dataframe(operational_summary, width="stretch", hide_index=True)
    chart_card_close()

    # --- Correlation heatmap ---
    chart_card_open(
        "🔎 Operational Correlation Analysis",
        "Values near 1 rise together; near 0 little relation; negative move opposite."
    )
    correlation_columns = [
        "non_urgent_orders", "urgent_orders", "fiscal_sector_orders",
        "traffic_controller_orders", "banking_orders_1", "banking_orders_2",
        "banking_orders_3", "total_orders"
    ]
    correlation_matrix = filtered_df[correlation_columns].corr()

    muted_scale = [T["card_alt"], T["accent_blue"], T["accent_amber"]] if theme_choice == "Dark" \
        else [T["card_alt"], T["accent_blue"], T["accent_rose"]]

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale=muted_scale,
            zmin=-1, zmax=1,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=10, color=T["text"]),
            colorbar=dict(outlinewidth=0),
        )
    )
    st.plotly_chart(style_fig(fig, height=480), width='stretch', config={"displayModeBar": False})
    chart_card_close()

with tab_clusters:

    st.markdown('<div class="section-eyebrow">Unsupervised learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧩 Logistics Demand Profiles</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Groups days with similar operational order patterns into '
        'three demand profiles — Low, Normal, and High — using K-Means clustering on all '
        'operational order categories.</div>',
        unsafe_allow_html=True
    )

    cluster_features = [
        "non_urgent_orders", "urgent_orders", "order_type_a", "order_type_b",
        "order_type_c", "fiscal_sector_orders", "traffic_controller_orders",
        "banking_orders_1", "banking_orders_2", "banking_orders_3"
    ]

    if len(filtered_df) < 3:
        st.warning(
            "Not enough data points in the current filter selection to run "
            "K-Means clustering (need at least 3). Try widening your filters."
        )
    else:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(filtered_df[cluster_features])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)

        cluster_df = filtered_df.copy()
        cluster_df["cluster"] = cluster_labels

        cluster_order = (
            cluster_df.groupby("cluster")["total_orders"]
            .mean().sort_values().index.tolist()
        )
        demand_labels = {
            cluster_order[0]: "Low Demand",
            cluster_order[1]: "Normal Demand",
            cluster_order[2]: "High Demand"
        }
        cluster_df["demand_profile"] = cluster_df["cluster"].map(demand_labels)
        label_order = ["Low Demand", "Normal Demand", "High Demand"]

        # --- KPI row: cluster sizes as metric cards ---
        cluster_sizes = (
            cluster_df["demand_profile"].value_counts()
            .reindex(label_order).fillna(0).astype(int)
        )

        size_cols = st.columns(3)
        for i, label in enumerate(label_order):
            with size_cols[i]:
                metric_card(
                    label,
                    f"{cluster_sizes[label]}",
                    "days in this profile"
                )

        st.write("")

        # --- Two-column row: cluster size chart + PCA scatter ---
        clus_col1, clus_col2 = st.columns(2)

        with clus_col1:
            chart_card_open(
                "Cluster Sizes",
                "Day count per demand profile in the current filter."
            )
            fig = go.Figure(
                data=[go.Bar(
                    x=cluster_sizes.index,
                    y=cluster_sizes.values,
                    marker_color=[DEMAND_COLORS[l] for l in cluster_sizes.index],
                )]
            )
            fig.update_layout(xaxis_title="", yaxis_title="Number of Days")
            st.plotly_chart(style_fig(fig), width='stretch', config={"displayModeBar": False})
            chart_card_close()

        with clus_col2:
            pca = PCA(n_components=2, random_state=42)
            pca_coords = pca.fit_transform(scaled_features)
            explained_var = pca.explained_variance_ratio_

            chart_card_open(
                "PCA Cluster Visualization",
                "10 operational variables compressed to 2D for visualization."
            )
            fig = go.Figure()
            for label in label_order:
                mask = cluster_df["demand_profile"].values == label
                fig.add_trace(go.Scatter(
                    x=pca_coords[mask, 0],
                    y=pca_coords[mask, 1],
                    mode="markers",
                    name=label,
                    marker=dict(color=DEMAND_COLORS[label], size=9, opacity=0.85,
                                line=dict(width=1, color=T["border"])),
                ))
            fig.update_layout(
                xaxis_title=f"PC1 ({explained_var[0]*100:.1f}% var)",
                yaxis_title=f"PC2 ({explained_var[1]*100:.1f}% var)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color=T["text"])),
            )
            st.plotly_chart(style_fig(fig), width='stretch', config={"displayModeBar": False})
            chart_card_close()

        # --- Operational characteristics ---
        chart_card_open(
            "Operational Characteristics by Demand Profile",
            "Average order volume per category, broken down by demand profile."
        )
        profile_summary = (
            cluster_df.groupby("demand_profile")[cluster_features + ["total_orders"]]
            .mean().reindex(label_order).round(2)
        )
        st.dataframe(profile_summary, width="stretch")

        fig = go.Figure()
        for i, label in enumerate(label_order):
            fig.add_trace(go.Bar(
                x=cluster_features,
                y=profile_summary.loc[label, cluster_features],
                name=label,
                marker_color=DEMAND_COLORS[label],
            ))
        fig.update_layout(
            barmode="group",
            xaxis_title="",
            yaxis_title="Average Orders",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color=T["text"])),
        )
        st.plotly_chart(style_fig(fig, height=420), width='stretch', config={"displayModeBar": False})
        chart_card_close()