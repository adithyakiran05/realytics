import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Dealytics")

# -----------------------
# LOAD DATA
# -----------------------
if not os.path.exists("data/processed/final_output.csv"):
    st.error("Run decision pipeline first")
    st.stop()

df = pd.read_csv("data/processed/final_output.csv")

if os.path.exists("data/processed/lbo_output.csv"):
    lbo = pd.read_csv("data/processed/lbo_output.csv")
else:
    lbo = pd.DataFrame()

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.header("Company Analysis")

selected_company = st.sidebar.selectbox(
    "Select Company",
    df["Company"].unique()
)

st.sidebar.header("Filters")

min_score = st.sidebar.slider(
    "Minimum Decision Score",
    float(df["Decision_Score"].min()),
    float(df["Decision_Score"].max()),
    float(df["Decision_Score"].min())
)

filtered_df = df[df["Decision_Score"] >= min_score]

if filtered_df.empty:
    st.warning("No companies match filter. Showing all.")
    filtered_df = df.copy()

# -----------------------
# KPI CARDS
# -----------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Companies", len(filtered_df))
col2.metric("Avg Score", round(filtered_df["Decision_Score"].mean(), 2))

if not lbo.empty:
    col3.metric("Max IRR", f"{round(lbo['IRR'].max()*100, 1)}%")
else:
    col3.metric("Max IRR", "N/A")

# -----------------------
# TOP DEAL
# -----------------------
st.subheader("Top Deal")

top_company = filtered_df.sort_values(by="Decision_Score", ascending=False).iloc[0]

st.success(
    f"{top_company['Company']} → Score: {round(top_company['Decision_Score'],2)} | Decision: {top_company['Decision']}"
)

# -----------------------
# MAIN TABLE
# -----------------------
st.subheader("Deal Rankings")

def highlight(row):
    if row["Decision"] == "Strong Buy":
        return ["background-color: #d4edda"] * len(row)
    elif row["Decision"] == "Consider":
        return ["background-color: #fff3cd"] * len(row)
    else:
        return ["background-color: #f8d7da"] * len(row)

st.dataframe(filtered_df.style.apply(highlight, axis=1))

# -----------------------
# COMPANY DRILL DOWN
# -----------------------
st.subheader("Company Deep Dive")

company_df = df[df["Company"] == selected_company]
company_lbo = lbo[lbo["Company"] == selected_company]

st.write("### Key Metrics")
st.dataframe(company_df)

if not company_lbo.empty:
    st.write("### IRR Breakdown")
    st.dataframe(company_lbo)

    pivot_company = company_lbo.pivot_table(
        index="Entry_Multiple",
        columns="Scenario",
        values="IRR"
    )

    st.line_chart(pivot_company)

# -----------------------
# INVESTMENT MEMO
# -----------------------
st.subheader("Investment Memo")

def generate_memo(row):
    return f"""
**Company:** {row['Company']}

**Recommendation:** {row['Decision']}

**Upside:**
- Revenue Growth: {round(row['Revenue_Growth'],2)}
- EBITDA Margin: {round(row['EBITDA_Margin'],2)}
- Predicted Growth: {round(row['Predicted_Growth'],2)}

**Risks:**
- {row['Risk_Reason'] if row['Risk_Reason'] else "Moderate risk"}

**Conclusion:**
This company is rated **{row['Decision']}** based on combined ML predictions,
risk factors, and expected LBO returns.
"""

st.markdown(generate_memo(company_df.iloc[0]))

# -----------------------
# RISK
# -----------------------
st.subheader("Risk Insights")
st.dataframe(filtered_df[["Company", "Risk_Score", "Risk_Reason"]])

# -----------------------
# SENTIMENT
# -----------------------
st.subheader("Public Sentiment")

if "Sentiment_Confidence" in filtered_df.columns:
    st.dataframe(filtered_df[["Company", "Public_Perception", "Sentiment_Confidence"]])
else:
    st.dataframe(filtered_df[["Company", "Public_Perception"]])

# -----------------------
# LBO ANALYSIS
# -----------------------
if not lbo.empty:

    st.subheader("LBO IRR Analysis")
    st.dataframe(lbo)

    pivot = lbo.pivot_table(
        index="Entry_Multiple",
        columns="Scenario",
        values="IRR"
    )

    if not pivot.empty:
        st.subheader("IRR Sensitivity")
        st.line_chart(pivot)

        # -----------------------
        # HEATMAP
        # -----------------------
        st.subheader("IRR Heatmap")

        fig, ax = plt.subplots()
        cax = ax.imshow(pivot, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)

        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        plt.colorbar(cax)

        st.pyplot(fig)