
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.figure_factory import create_distplot


def EDA_Page():
    st.title("📊 Exploratory Data Analysis Playground")
    st.info("""
        Welcome to the EDA Playground! Explore your HR data with interactive visualizations.
        You can select different plot types and customize them to gain insights.
    """)


    # Check first if the data is there
    if 'final_df' not in st.session_state:
        st.warning("⚠️ Please upload a dataset on the 'Turnover Analysis' page first.")
        st.stop()  # Safer than return in some cases
    
    df = st.session_state['final_df'].copy()
    
    if 'EmployeeID' in df.columns:
        df = df.drop(columns=['EmployeeID'], errors='ignore')
    
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())
        
    st.sidebar.title("📌 EDA Plot Options")

    # Plotting options
    plot_type = st.sidebar.selectbox("Choose a Plot Type", [
        "Pie Chart", "Bar Chart", "Histogram", "Stacked Bar Chart", "Density Plot",
        "Scatter Plot"
    ])

    all_columns = df.columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    if plot_type == "Pie Chart":
        st.markdown("### Pie Chart")

        if not categorical_cols:
            st.warning("No categorical columns available for pie chart.")
            return

        select_column = st.sidebar.selectbox("Select a column for a pie chart", categorical_cols)

        pie_data = df[select_column].value_counts().reset_index()
        pie_data.columns = [select_column, 'Count']

        fig = px.pie(pie_data, names=select_column, values='Count',
                     title=f"Distribution of {select_column}",
                     hole=0.3)  # donut style

        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Bar Chart":
        st.markdown("### 📊 Bar Chart")

        if not categorical_cols:
            st.warning("No categorical columns available for bar chart (for the X-axis).")
            return

        select_column = st.sidebar.selectbox("Select main category (X-axis)", categorical_cols)

        group_column = st.sidebar.selectbox(
            "Optional: Select a categorical column to group by (Color)",
            [None] + [col for col in categorical_cols if col != select_column]
        )

        if group_column:
            # Grouped bar chart
            bar_data = df.groupby([select_column, group_column]).size().reset_index(name='Count')
            fig = px.bar(
                bar_data,
                x=select_column,
                y='Count',
                color=group_column,
                text='Count',
                barmode='group',  # or 'stack'
                color_continuous_scale='Viridis',  # Color scale for continuous data
                title=f"Count of {select_column} grouped by {group_column}"
            )
        else:
            # Simple bar chart
            bar_data = df[select_column].value_counts().reset_index()
            bar_data.columns = [select_column, 'Count']
            fig = px.bar(
                bar_data,
                x=select_column,
                y='Count',
                text='Count',
                color=select_column,  # Color bars based on categories in the select_column
                color_discrete_sequence=px.colors.qualitative.Set1,  # Use predefined color palette for discrete categories
                title=f"Count of {select_column}"
            )

        fig.update_traces(textposition='outside')  # Move text outside the bars for readability
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Histogram":
        st.markdown("### Histogram")

        if not numerical_cols:
            st.warning("No Numerical columns available for Histogram.")
            return

        select_column = st.sidebar.selectbox("Select a column for a Histogram", numerical_cols)

        fig = px.histogram(
            df,
            x=select_column,
            nbins=30,  # You can adjust the number of bins
            histnorm='probability density',
            title=f"Distribution of {select_column} with Normal Curve",
            color_discrete_sequence=["#636EFA"]  # Optional: customize color
        )

        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Stacked Bar Chart":
        st.markdown("### Stack Bar Chart")

        if not categorical_cols or len(categorical_cols) < 2:
            st.warning("At least two categorical columns are needed for a stacked bar chart.")
            return

        x_col = st.sidebar.selectbox("Select X-axis category", categorical_cols)
        color_col = st.sidebar.selectbox(
            "Select category to stack by (Color)",
            [col for col in categorical_cols if col != x_col]
        )

        # Create a grouped count DataFrame
        stacked_data = df.groupby([x_col, color_col]).size().reset_index(name='Count')

        fig = px.bar(
            stacked_data,
            x=x_col,
            y="Count",
            color=color_col,
            title=f"{x_col} distribution stacked by {color_col}",
            barmode="stack",  # 'stack' for stacked bars, 'group' for side-by-side
            color_discrete_sequence=px.colors.qualitative.Pastel  # More colorful
        )

        fig.update_layout(xaxis_title=x_col, yaxis_title="Count")

        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Density Plot":
        st.markdown("### 📈 Density Plot")

        if not numerical_cols:
            st.warning("No numerical columns available for Density Plot.")
            return

        select_column = st.sidebar.selectbox("Select a column for Density Plot", numerical_cols)

        # Prepare the data
        data = df[select_column].dropna()
        hist_data = [data]
        group_labels = [select_column]

        # Create the distribution plot (KDE + Histogram)
        fig = create_distplot(
            hist_data,
            group_labels,
            bin_size=1,  # Adjust based on your data scale
            show_hist=True,
            show_rug=True,
            curve_type='kde'
        )
        # Change KDE line color
        for trace in fig.data:
            if trace.type == 'scatter' and trace.mode == 'lines':  # Identify the KDE trace
                trace.line.color = "#FF5733"  # Customize this color as desired

        fig.update_layout(
            title_text=f"Density Plot of {select_column} with KDE",
            xaxis_title=select_column,
            yaxis_title="Density",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Scatter Plot":
        st.markdown("### 📍 Scatter Plot")

        if not numerical_cols or len(numerical_cols) < 2:
            st.warning("At least two numerical columns are needed for a Scatter Plot.")
            return

        x_col = st.sidebar.selectbox("Select X-axis column for Scatter Plot", numerical_cols)
        y_col = st.sidebar.selectbox("Select Y-axis column for Scatter Plot", [col for col in numerical_cols if col != x_col])

        # Optional: Select an additional categorical column for grouping (color)
        group_column = st.sidebar.selectbox(
            "Optional: Select a categorical column to group by (Color)",
            [None] + [col for col in categorical_cols if col != x_col and col != y_col]
        )

        # If a categorical column is selected, apply it for color grouping
        if group_column:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=group_column,  # Color the scatter plot points by the selected categorical column
                title=f"Scatter Plot of {x_col} vs {y_col} grouped by {group_column}",
            )
        else:
            # If no grouping column is selected, plot without color grouping
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"Scatter Plot of {x_col} vs {y_col}",
            )

        # Layout styling
        fig.update_layout(
            title_text=f"Scatter Plot of {x_col} vs {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
        )

        st.plotly_chart(fig, use_container_width=True)
    

    st.sidebar.title("📌 Time Series Analysis")

    
    plot_type = st.sidebar.selectbox("Choose a Plot Type", [
        "Time Series Analysis"])

