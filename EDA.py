import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.figure_factory import create_distplot
import plotly.graph_objects as go
from datetime import datetime
from plotly.subplots import make_subplots

def EDA_Page():
    st.title("📊 HR Analytics Dashboard")
    
    # Add a sidebar for navigation
    st.sidebar.title("🔍 Navigation")
    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        ["📊 Data Overview", "📈 Visual Analysis", "📉 Statistical Analysis", "🔄 Time Series Analysis"]
    )
    
    # Check if data exists in session state
    if 'final_df' not in st.session_state:
        st.warning("⚠️ Please upload a dataset on the 'Turnover Analysis' page first.")
        st.stop()
    
    # Get the dataframe from session state
    df = st.session_state['final_df'].copy()
    
    # Remove EmployeeID if present
    if 'EmployeeID' in df.columns:
        df = df.drop(columns=['EmployeeID'], errors='ignore')
    
    # Data Overview Section
    if analysis_type == "📊 Data Overview":
        st.header("📊 Data Overview")
        
        # Basic Information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Number of Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Data Preview with expandable sections
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head())
        
        with st.expander("📊 Column Information", expanded=False):
            st.write("### Column Types")
            col_types = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Unique Values': df.nunique(),
                'Missing Values': df.isnull().sum()
            })
            st.dataframe(col_types)
        
        with st.expander("📈 Basic Statistics", expanded=False):
            st.write("### Numerical Columns Statistics")
            st.dataframe(df.describe())
            
            st.write("### Categorical Columns Statistics")
            categorical_stats = pd.DataFrame({
                'Column': df.select_dtypes(include=['object']).columns,
                'Most Common Value': df.select_dtypes(include=['object']).apply(lambda x: x.mode()[0] if not x.empty else None),
                'Frequency': df.select_dtypes(include=['object']).apply(lambda x: x.mode()[0] if not x.empty else None)
            })
            st.dataframe(categorical_stats)
    
    # Visual Analysis Section
    elif analysis_type == "📈 Visual Analysis":
        st.header("📈 Visual Analysis")
        
        # Plot selection with better organization
        plot_type = st.selectbox(
            "Select Visualization Type",
            ["📊 Distribution Analysis", "📈 Correlation Analysis", "📉 Trend Analysis", "📊 Categorical Analysis"]
        )
        
        if plot_type == "📊 Distribution Analysis":
            st.subheader("Distribution Analysis")
            
            # Select column type
            col_type = st.radio("Select Column Type", ["Numerical", "Categorical"])
            
            if col_type == "Numerical":
                num_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(num_cols) > 0:
                    selected_col = st.selectbox("Select Column", num_cols)
                    
                    # Distribution plot options
                    plot_options = st.multiselect(
                        "Select Plot Types",
                        ["Histogram", "Density Plot", "Box Plot"],
                        default=["Histogram"]
                    )
                    
                    if "Histogram" in plot_options:
                        fig = px.histogram(
                            df, x=selected_col,
                            nbins=30,
                            title=f"Distribution of {selected_col}",
                            marginal="box"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if "Density Plot" in plot_options:
                        fig = create_distplot(
                            [df[selected_col].dropna()],
                            [selected_col],
                            show_hist=False,
                            show_rug=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if "Box Plot" in plot_options:
                        fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numerical columns available for analysis.")
            
            else:  # Categorical
                cat_cols = df.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    selected_col = st.selectbox("Select Column", cat_cols)
                    
                    # Categorical plot options
                    plot_options = st.multiselect(
                        "Select Plot Types",
                        ["Pie Chart", "Bar Chart", "Stacked Bar Chart"],
                        default=["Bar Chart"]
                    )
                    
                    if "Pie Chart" in plot_options:
                        fig = px.pie(
                            df, names=selected_col,
                            title=f"Distribution of {selected_col}",
                            hole=0.3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if "Bar Chart" in plot_options:
                        # Allow selection of grouping column
                        group_col = st.selectbox(
                            "Optional: Select a column to group by (Color)",
                            [None] + [col for col in cat_cols if col != selected_col]
                        )
                        
                        if group_col:
                            # Grouped bar chart
                            bar_data = df.groupby([selected_col, group_col]).size().reset_index(name='Count')
                            fig = px.bar(
                                bar_data,
                                x=selected_col,
                                y='Count',
                                color=group_col,
                                text='Count',
                                barmode='group',
                                title=f"Count of {selected_col} grouped by {group_col}"
                            )
                        else:
                            # Simple bar chart
                            bar_data = df[selected_col].value_counts().reset_index()
                            bar_data.columns = [selected_col, 'Count']
                            fig = px.bar(
                                bar_data,
                                x=selected_col,
                                y='Count',
                                text='Count',
                                title=f"Count of {selected_col}"
                            )
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if "Stacked Bar Chart" in plot_options:
                        group_col = st.selectbox(
                            "Select Grouping Column",
                            [col for col in cat_cols if col != selected_col]
                        )
                        if group_col:
                            fig = px.bar(
                                df.groupby([selected_col, group_col]).size().reset_index(name='count'),
                                x=selected_col, y='count', color=group_col,
                                title=f"{selected_col} by {group_col}",
                                barmode='stack'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No categorical columns available for analysis.")
        
        elif plot_type == "📈 Correlation Analysis":
            st.subheader("Correlation Analysis")
            
            # Select numerical columns for correlation
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(num_cols) >= 2:
                selected_cols = st.multiselect(
                    "Select Columns for Correlation",
                    num_cols,
                    default=num_cols[:2] if len(num_cols) >= 2 else []
                )
                
                if len(selected_cols) >= 2:
                    # Correlation matrix
                    corr_matrix = df[selected_cols].corr()
                    fig = px.imshow(
                        corr_matrix,
                        title="Correlation Matrix",
                        color_continuous_scale='RdBu'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Scatter plot matrix
                    fig = px.scatter_matrix(
                        df[selected_cols],
                        title="Scatter Plot Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least 2 columns for correlation analysis.")
            else:
                st.warning("Not enough numerical columns for correlation analysis.")
        
        elif plot_type == "📉 Trend Analysis":
            st.subheader("Trend Analysis")
            
            # Select date column
            date_cols = df.select_dtypes(include=['datetime64[ns]']).columns
            if len(date_cols) > 0:
                date_col = st.selectbox("Select Date Column", date_cols)
                
                # Select value column
                value_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(value_cols) > 0:
                    value_col = st.selectbox("Select Value Column", value_cols)
                    
                    # Group by time period
                    time_period = st.selectbox(
                        "Select Time Period",
                        ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
                    )
                    
                    # Create a copy of the data
                    trend_data = df[[date_col, value_col]].copy()
                    
                    # Group by selected time period
                    if time_period == "Daily":
                        trend_data = trend_data.groupby(trend_data[date_col].dt.date).agg({value_col: 'mean'}).reset_index()
                        trend_data[date_col] = pd.to_datetime(trend_data[date_col])
                    elif time_period == "Weekly":
                        trend_data = trend_data.groupby(trend_data[date_col].dt.to_period('W')).agg({value_col: 'mean'}).reset_index()
                        trend_data[date_col] = trend_data[date_col].dt.to_timestamp()
                    elif time_period == "Monthly":
                        trend_data = trend_data.groupby(trend_data[date_col].dt.to_period('M')).agg({value_col: 'mean'}).reset_index()
                        trend_data[date_col] = trend_data[date_col].dt.to_timestamp()
                    elif time_period == "Quarterly":
                        trend_data = trend_data.groupby(trend_data[date_col].dt.to_period('Q')).agg({value_col: 'mean'}).reset_index()
                        trend_data[date_col] = trend_data[date_col].dt.to_timestamp()
                    elif time_period == "Yearly":
                        trend_data = trend_data.groupby(trend_data[date_col].dt.to_period('Y')).agg({value_col: 'mean'}).reset_index()
                        trend_data[date_col] = trend_data[date_col].dt.to_timestamp()
                    
                    # Create trend line
                    fig = go.Figure()
                    
                    # Add actual data
                    fig.add_trace(go.Scatter(
                        x=trend_data[date_col],
                        y=trend_data[value_col],
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='blue')
                    ))
                    
                    # Add trend line
                    z = np.polyfit(range(len(trend_data)), trend_data[value_col], 1)
                    p = np.poly1d(z)
                    trend_line = p(range(len(trend_data)))
                    
                    fig.add_trace(go.Scatter(
                        x=trend_data[date_col],
                        y=trend_line,
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{value_col} Trend Analysis ({time_period})",
                        xaxis_title="Date",
                        yaxis_title=value_col,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add summary statistics
                    with st.expander("📊 Summary Statistics", expanded=False):
                        stats = pd.DataFrame({
                            'Start Date': [trend_data[date_col].min()],
                            'End Date': [trend_data[date_col].max()],
                            'Mean': [trend_data[value_col].mean()],
                            'Std Dev': [trend_data[value_col].std()],
                            'Min': [trend_data[value_col].min()],
                            'Max': [trend_data[value_col].max()],
                            'Trend': ['Increasing' if trend_line[-1] > trend_line[0] else 'Decreasing']
                        })
                        st.dataframe(stats)
                else:
                    st.warning("No numerical columns available for trend analysis.")
            else:
                st.warning("No date columns available for trend analysis.")
        
        elif plot_type == "📊 Categorical Analysis":
            st.subheader("Categorical Analysis")
            
            # Select categorical columns
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                # Select primary categorical column
                primary_col = st.selectbox("Select Primary Categorical Column", cat_cols)
                
                # Select secondary categorical column (optional)
                secondary_col = st.selectbox(
                    "Select Secondary Categorical Column (Optional)",
                    [None] + [col for col in cat_cols if col != primary_col]
                )
                
                # Select plot type
                plot_type = st.selectbox(
                    "Select Plot Type",
                    ["Bar Chart", "Pie Chart", "Stacked Bar Chart"]
                )
                
                if plot_type == "Bar Chart":
                    if secondary_col:
                        # Grouped bar chart
                        bar_data = df.groupby([primary_col, secondary_col]).size().reset_index(name='Count')
                        fig = px.bar(
                            bar_data,
                            x=primary_col,
                            y='Count',
                            color=secondary_col,
                            title=f"Distribution of {primary_col} by {secondary_col}",
                            barmode='group'
                        )
                    else:
                        # Simple bar chart
                        bar_data = df[primary_col].value_counts().reset_index()
                        bar_data.columns = [primary_col, 'Count']
                        fig = px.bar(
                            bar_data,
                            x=primary_col,
                            y='Count',
                            title=f"Distribution of {primary_col}"
                        )
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Pie Chart":
                    if secondary_col:
                        # Grouped pie chart
                        pie_data = df.groupby([primary_col, secondary_col]).size().reset_index(name='Count')
                        fig = px.sunburst(
                            pie_data,
                            path=[primary_col, secondary_col],
                            values='Count',
                            title=f"Distribution of {primary_col} by {secondary_col}"
                        )
                    else:
                        # Simple pie chart
                        pie_data = df[primary_col].value_counts().reset_index()
                        pie_data.columns = [primary_col, 'Count']
                        fig = px.pie(
                            pie_data,
                            names=primary_col,
                            values='Count',
                            title=f"Distribution of {primary_col}",
                            hole=0.3
                        )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Stacked Bar Chart":
                    if secondary_col:
                        # Stacked bar chart
                        stacked_data = df.groupby([primary_col, secondary_col]).size().reset_index(name='Count')
                        fig = px.bar(
                            stacked_data,
                            x=primary_col,
                            y='Count',
                            color=secondary_col,
                            title=f"Distribution of {primary_col} by {secondary_col}",
                            barmode='stack'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select a secondary categorical column for stacked bar chart.")
                
                # Add summary statistics
                with st.expander("📊 Summary Statistics", expanded=False):
                    if secondary_col:
                        cross_tab = pd.crosstab(df[primary_col], df[secondary_col])
                        st.write("### Cross Tabulation")
                        st.dataframe(cross_tab)
                    else:
                        value_counts = df[primary_col].value_counts()
                        st.write("### Value Counts")
                        st.dataframe(value_counts)
            else:
                st.warning("No categorical columns available for analysis.")
    
    # Statistical Analysis Section
    elif analysis_type == "📉 Statistical Analysis":
        st.header("📉 Statistical Analysis")
        
        # Select analysis type
        stat_type = st.selectbox(
            "Select Statistical Analysis",
            ["Descriptive Statistics", "Hypothesis Testing", "ANOVA Analysis"]
        )
        
        if stat_type == "Descriptive Statistics":
            st.subheader("Descriptive Statistics")
            
            # Select columns for analysis
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            selected_cols = st.multiselect(
                "Select Columns for Analysis",
                num_cols,
                default=num_cols[:2] if len(num_cols) >= 2 else []
            )
            
            if selected_cols:
                stats_df = df[selected_cols].describe().T
                stats_df['Skewness'] = df[selected_cols].skew()
                stats_df['Kurtosis'] = df[selected_cols].kurtosis()
                st.dataframe(stats_df)
        
        elif stat_type == "Hypothesis Testing":
            st.subheader("Hypothesis Testing")
            
            # Select columns for testing
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = df.select_dtypes(include=['object']).columns
            
            if len(num_cols) > 0 and len(cat_cols) > 0:
                # Select numerical column for testing
                num_col = st.selectbox("Select Numerical Column", num_cols)
                
                # Select categorical column for grouping
                cat_col = st.selectbox("Select Categorical Column for Grouping", cat_cols)
                
                # Select test type
                test_type = st.selectbox(
                    "Select Test Type",
                    ["T-Test", "Mann-Whitney U Test", "Kruskal-Wallis Test"]
                )
                
                if st.button("Run Test"):
                    with st.spinner("Performing hypothesis test..."):
                        from scipy import stats
                        
                        # Get unique categories
                        categories = df[cat_col].unique()
                        
                        if len(categories) == 2 and test_type in ["T-Test", "Mann-Whitney U Test"]:
                            group1 = df[df[cat_col] == categories[0]][num_col]
                            group2 = df[df[cat_col] == categories[1]][num_col]
                            
                            if test_type == "T-Test":
                                t_stat, p_value = stats.ttest_ind(group1, group2)
                                st.write(f"T-Test Results:")
                                st.write(f"T-statistic: {t_stat:.4f}")
                                st.write(f"P-value: {p_value:.4f}")
                                
                                # Interpretation
                                alpha = 0.05
                                if p_value < alpha:
                                    st.success("Reject null hypothesis: There is a significant difference between the groups.")
                                else:
                                    st.info("Fail to reject null hypothesis: No significant difference between the groups.")
                            
                            else:  # Mann-Whitney U Test
                                u_stat, p_value = stats.mannwhitneyu(group1, group2)
                                st.write(f"Mann-Whitney U Test Results:")
                                st.write(f"U-statistic: {u_stat:.4f}")
                                st.write(f"P-value: {p_value:.4f}")
                                
                                # Interpretation
                                alpha = 0.05
                                if p_value < alpha:
                                    st.success("Reject null hypothesis: There is a significant difference between the groups.")
                                else:
                                    st.info("Fail to reject null hypothesis: No significant difference between the groups.")
                        
                        elif test_type == "Kruskal-Wallis Test":
                            groups = [df[df[cat_col] == cat][num_col] for cat in categories]
                            h_stat, p_value = stats.kruskal(*groups)
                            st.write(f"Kruskal-Wallis Test Results:")
                            st.write(f"H-statistic: {h_stat:.4f}")
                            st.write(f"P-value: {p_value:.4f}")
                            
                            # Interpretation
                            alpha = 0.05
                            if p_value < alpha:
                                st.success("Reject null hypothesis: There is a significant difference between at least two groups.")
                            else:
                                st.info("Fail to reject null hypothesis: No significant difference between the groups.")
            else:
                st.warning("Need both numerical and categorical columns for hypothesis testing.")
        
        elif stat_type == "ANOVA Analysis":
            st.subheader("ANOVA Analysis")
            
            # Select columns for ANOVA
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = df.select_dtypes(include=['object']).columns
            
            if len(num_cols) > 0 and len(cat_cols) > 0:
                # Select numerical column for ANOVA
                num_col = st.selectbox("Select Numerical Column", num_cols)
                
                # Select categorical column for grouping
                cat_col = st.selectbox("Select Categorical Column for Grouping", cat_cols)
                
                # Select ANOVA type
                anova_type = st.selectbox(
                    "Select ANOVA Type",
                    ["One-Way ANOVA", "Two-Way ANOVA"]
                )
                
                if st.button("Run ANOVA"):
                    with st.spinner("Performing ANOVA..."):
                        from scipy import stats
                        
                        if anova_type == "One-Way ANOVA":
                            # Get unique categories
                            categories = df[cat_col].unique()
                            groups = [df[df[cat_col] == cat][num_col] for cat in categories]
                            
                            # Perform one-way ANOVA
                            f_stat, p_value = stats.f_oneway(*groups)
                            
                            st.write(f"One-Way ANOVA Results:")
                            st.write(f"F-statistic: {f_stat:.4f}")
                            st.write(f"P-value: {p_value:.4f}")
                            
                            # Interpretation
                            alpha = 0.05
                            if p_value < alpha:
                                st.success("Reject null hypothesis: There is a significant difference between at least two groups.")
                            else:
                                st.info("Fail to reject null hypothesis: No significant difference between the groups.")
                        
                        elif anova_type == "Two-Way ANOVA":
                            # Select second categorical column
                            cat_col2 = st.selectbox(
                                "Select Second Categorical Column",
                                [col for col in cat_cols if col != cat_col]
                            )
                            
                            # Perform two-way ANOVA
                            import statsmodels.api as sm
                            from statsmodels.formula.api import ols
                            
                            # Create formula for two-way ANOVA
                            formula = f"{num_col} ~ C({cat_col}) + C({cat_col2}) + C({cat_col}):C({cat_col2})"
                            model = ols(formula, data=df).fit()
                            anova_table = sm.stats.anova_lm(model, typ=2)
                            
                            st.write("Two-Way ANOVA Results:")
                            st.dataframe(anova_table)
                            
                            # Interpretation
                            alpha = 0.05
                            significant_effects = []
                            for effect in anova_table.index:
                                if anova_table.loc[effect, 'PR(>F)'] < alpha:
                                    significant_effects.append(effect)
                            
                            if significant_effects:
                                st.success(f"Significant effects found for: {', '.join(significant_effects)}")
                            else:
                                st.info("No significant effects found.")
            else:
                st.warning("Need both numerical and categorical columns for ANOVA analysis.")
    
    # Time Series Analysis Section
    elif analysis_type == "🔄 Time Series Analysis":
        st.header("🔄 Time Series Analysis")
        
        # Check for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        if len(datetime_cols) > 0:
            # Create two columns for controls
            col1, col2 = st.columns(2)
            
            with col1:
                date_col = st.selectbox("Select Date Column", datetime_cols)
                value_col = st.selectbox(
                    "Select Value Column",
                    df.select_dtypes(include=['int64', 'float64']).columns
                )
            
            with col2:
                # Add aggregation options
                aggregation = st.selectbox(
                    "Select Aggregation",
                    ["None", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
                )
                
                # Add smoothing option
                smoothing = st.slider(
                    "Smoothing Window (days)",
                    min_value=0,
                    max_value=30,
                    value=0,
                    help="Apply moving average smoothing to the data"
                )
            
            # Create a copy of the data for processing
            ts_data = df[[date_col, value_col]].copy()
            
            # Handle aggregation
            if aggregation != "None":
                if aggregation == "Daily":
                    ts_data = ts_data.groupby(ts_data[date_col].dt.date).agg({value_col: 'mean'}).reset_index()
                    ts_data[date_col] = pd.to_datetime(ts_data[date_col])
                elif aggregation == "Weekly":
                    ts_data = ts_data.groupby(ts_data[date_col].dt.to_period('W')).agg({value_col: 'mean'}).reset_index()
                    ts_data[date_col] = ts_data[date_col].dt.to_timestamp()
                elif aggregation == "Monthly":
                    ts_data = ts_data.groupby(ts_data[date_col].dt.to_period('M')).agg({value_col: 'mean'}).reset_index()
                    ts_data[date_col] = ts_data[date_col].dt.to_timestamp()
                elif aggregation == "Quarterly":
                    ts_data = ts_data.groupby(ts_data[date_col].dt.to_period('Q')).agg({value_col: 'mean'}).reset_index()
                    ts_data[date_col] = ts_data[date_col].dt.to_timestamp()
                elif aggregation == "Yearly":
                    ts_data = ts_data.groupby(ts_data[date_col].dt.to_period('Y')).agg({value_col: 'mean'}).reset_index()
                    ts_data[date_col] = ts_data[date_col].dt.to_timestamp()
            
            # Apply smoothing if selected
            if smoothing > 0:
                ts_data[value_col] = ts_data[value_col].rolling(window=smoothing, min_periods=1).mean()
            
            # Create the time series plot
            fig = go.Figure()
            
            # Add main line
            fig.add_trace(go.Scatter(
                x=ts_data[date_col],
                y=ts_data[value_col],
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            
            # Calculate trend line for summary statistics
            trend_line = np.poly1d(np.polyfit(
                range(len(ts_data)),
                ts_data[value_col],
                1
            ))(range(len(ts_data)))
            
            # Add trend line if requested
            if st.checkbox("Show Trend Line"):
                fig.add_trace(go.Scatter(
                    x=ts_data[date_col],
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash')
                ))
            
            # Add seasonal decomposition if requested
            if st.checkbox("Show Seasonal Decomposition"):
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                # Ensure data is sorted by date
                ts_data = ts_data.sort_values(date_col)
                
                # Set the frequency based on aggregation
                freq = {
                    "None": 'D',
                    "Daily": 'D',
                    "Weekly": 'W',
                    "Monthly": 'M',
                    "Quarterly": 'Q',
                    "Yearly": 'Y'
                }[aggregation]
                
                try:
                    # Create a time series with proper frequency
                    ts_series = pd.Series(
                        ts_data[value_col].values,
                        index=pd.date_range(
                            start=ts_data[date_col].min(),
                            periods=len(ts_data),
                            freq=freq
                        )
                    )
                    
                    # Perform decomposition
                    decomposition = seasonal_decompose(ts_series, model='additive')
                    
                    # Create subplots for decomposition
                    fig = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual')
                    )
                    
                    # Add traces
                    fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name='Original'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual'), row=4, col=1)
                    
                    # Update layout
                    fig.update_layout(height=1000, showlegend=False)
                    
                except Exception as e:
                    st.warning(f"Could not perform seasonal decomposition: {str(e)}")
            
            # Update layout for better visualization
            fig.update_layout(
                title=f"Time Series Analysis of {value_col}",
                xaxis_title="Date",
                yaxis_title=value_col,
                hovermode='x unified',
                showlegend=True,
                height=600
            )
            
            # Add range slider
            fig.update_xaxes(rangeslider_visible=True)
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Add summary statistics
            with st.expander("📊 Summary Statistics", expanded=False):
                stats = pd.DataFrame({
                    'Start Date': [ts_data[date_col].min()],
                    'End Date': [ts_data[date_col].max()],
                    'Mean': [ts_data[value_col].mean()],
                    'Std Dev': [ts_data[value_col].std()],
                    'Min': [ts_data[value_col].min()],
                    'Max': [ts_data[value_col].max()],
                    'Trend': ['Increasing' if trend_line[-1] > trend_line[0] else 'Decreasing']
                })
                st.dataframe(stats)
            
            # Add download option for the processed data
            csv = ts_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Processed Time Series Data",
                data=csv,
                file_name='time_series_data.csv',
                mime='text/csv'
            )
        else:
            st.warning("No datetime columns available for time series analysis.")

# Add this at the end of the file
if __name__ == "__main__":
    EDA_Page()

