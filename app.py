
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Student Performance Analytics",
    layout="wide"
)

st.title("Student Performance Analytics Dashboard")
st.caption("Python Data Analytics | Synthetic Dataset")

# 1. DATA COLLECTION

df = pd.read_csv("synthetic_student_grades.csv")

st.header("1. Data Exploration")

with st.expander("View Original Dataset"):
    st.dataframe(df)
    st.write("Total rows:", df.shape[0])
    st.write("Total columns:", df.shape[1])
    st.write("Missing values:")
    st.write(df.isnull().sum())
    st.write("Duplicate records:", df.duplicated().sum())

# 2. DATA CLEANING

df = df.drop_duplicates()

df = df.rename(columns={
    "Student_ID": "Student ID",
    "Final_Exam": "Final Exam"
})

assessments = [
    "Assignment_1",
    "Assignment_2",
    "Assignment_3",
    "Assignment_4",
    "Assignment_5",
    "Midterm",
    "Final Exam"
]

for column in assessments:
    df[column] = df[column].fillna(df[column].median())

df["Attendance"] = df["Attendance"].fillna(
    df["Attendance"].median()
)

# 3. DATA ANALYSIS

df["Average"] = df[assessments].mean(axis=1).round(2)

df["Result"] = np.where(
    df["Average"] >= 50, "Pass", "Fail"
)

# 4. INTERACTIVE FILTERS

st.sidebar.header("Dashboard Filters")

programs = st.sidebar.multiselect(
    "Select Program",
    df["Program"].unique(),
    default=df["Program"].unique()
)

grade_range = st.sidebar.slider(
    "Average Grade Range",
    0, 100, (0, 100)
)

filtered = df[
    (df["Program"].isin(programs)) &
    (df["Average"].between(
        grade_range[0], grade_range[1]
    ))
]

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# 5. KEY PERFORMANCE INDICATORS

st.header("2. Performance Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Students",
    filtered["Student ID"].nunique()
)

col2.metric(
    "Average Grade",
    round(filtered["Average"].mean(), 2)
)

col3.metric(
    "Highest Grade",
    filtered["Average"].max()
)

col4.metric(
    "Pass Rate",
    "{:.1f}%".format(
        (filtered["Result"] == "Pass").mean() * 100
    )
)

# 6. DATA VISUALIZATION

st.header("3. Data Visualization")

sns.set_theme(style="whitegrid")

col1, col2 = st.columns(2)

# Bar Chart

with col1:
    st.subheader("Average Grade by Assessment")

    fig, ax = plt.subplots(figsize=(7, 4))

    filtered[assessments].mean().plot(
        kind="bar",
        color="teal",
        ax=ax
    )

    ax.set_ylabel("Average Grade")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

# Histogram

with col2:
    st.subheader("Grade Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.histplot(
        data=filtered,
        x="Average",
        bins=10,
        kde=True,
        color="steelblue",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

col3, col4 = st.columns(2)

# Line Chart

with col3:
    st.subheader("Performance Across Assessments")

    fig, ax = plt.subplots(figsize=(7, 4))

    filtered[assessments].mean().plot(
        kind="line",
        marker="o",
        color="teal",
        ax=ax
    )

    ax.set_ylabel("Average Grade")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

# Scatter Plot

with col4:
    st.subheader("Attendance vs Academic Performance")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.scatterplot(
        data=filtered,
        x="Attendance",
        y="Average",
        hue="Program",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

# Box Plot

st.subheader("Performance Comparison by Program")

fig, ax = plt.subplots(figsize=(10, 4))

sns.boxplot(
    data=filtered,
    x="Program",
    y="Average",
    ax=ax
)

st.pyplot(fig)
plt.close(fig)

# 7. STATISTICAL ANALYSIS

st.header("4. Statistical Analysis")

st.dataframe(
    filtered[assessments + ["Average"]]
    .describe()
    .round(2)
)

st.subheader("Average Performance by Program")

st.dataframe(
    filtered.groupby("Program")["Average"]
    .agg(["count", "mean", "min", "max"])
    .round(2)
)

# 8. TOP PERFORMING STUDENTS

st.header("5. Top 10 Performing Students")

st.dataframe(
    filtered.nlargest(10, "Average")[
        ["Student ID", "Program", "Average", "Result"]
    ]
)

# 9. FILTERED DATA AND EXPORT

st.header("6. Filtered Dataset")

st.dataframe(filtered)

st.download_button(
    label="Download Analysis Results",
    data=filtered.to_csv(index=False),
    file_name="student_analysis.csv",
    mime="text/csv"
)

st.divider()

st.caption(
    "Independent portfolio project using synthetic data. "
    "Equal assessment weights and a 50% passing threshold "
    "are used for demonstration purposes only."
)
