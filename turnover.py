import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from datetime import datetime
import calendar
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def turnover_tab(tabs):
    with tabs[0]:
        st.header("Please Update the data in the specified format.")
        st.info("""
        - Kindly avoid adding any extra columns to the sheet, as they will not be processed.
        - If you do not have data for a required column, you may leave it blank. 
          Empty columns will be automatically removed during data processing.
        """)

        
        st.download_button(
            label="📥 Download Attrition Template",
            data="EmployeeID,Department,Job level,DOJ,DOB,DOE,Gender,DEIGroup,Country,Reason for Leaving\n1001,Purchase/HR,1 to 5,2022-12-19,1991-11-27,2023-12-18,Male/Female,Yes/No,India/China,Career Development",
            file_name="attrition_template.csv",
            mime="text/csv", key = "turnover"
        )
        
        
        
        st.header("Turnover Analysis Dashboard")
        st.subheader("1. Upload and Prepare Data")

        upload_file = st.file_uploader("⬆️ Upload Turnover CSV File", type='csv', key="turnover_upload")
        if upload_file is not None:
            try:
                df = pd.read_csv(upload_file, parse_dates=['DOJ', 'DOE', 'DOB'])
                df['DOJ'] = pd.to_datetime(df['DOJ'], errors='coerce')
                df['DOE'] = pd.to_datetime(df['DOE'], errors='coerce')
                df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
                today = datetime.now()
                df['Tenure'] = ((df['DOE'] - df['DOJ']).dt.days / 365.25).round(1)
                df['Age'] = ((today - df['DOB']).dt.days / 365.25).round(1)

                # Store the processed dataframe in session state
                st.session_state['final_df'] = df

                with st.expander("📊 Data Preview", expanded=False):
                    st.dataframe(df.head())
                    st.info(f"ℹ️ **Data Summary:** {df.shape[0]} rows and {df.shape[1]} columns.")
                
                with st.expander("🔢 Numeric Data Summary", expanded=False):
                    st.dataframe(df.describe(include=[np.number]))
                
                with st.expander("🔤 Categorical (Object) Data Summary", expanded=False):
                    st.dataframe(df.describe(include=[object]))

                with st.expander("🧹 Step 2: Data Cleaning", expanded=False):
                    st.info("This step helps identify and handle missing values in your dataset.")
                    
                    columns_to_exclude = ['DOE']  # exclude columns like 'DOE' that shouldn't be analyzed
                    cleanable_columns = [col for col in df.columns if col not in columns_to_exclude]

                    selected_columns = st.multiselect(
                        "🧼 Select columns to check for missing values:",
                        cleanable_columns,
                        default=cleanable_columns,
                        help="Choose columns to analyze for null values and cleaning",
                        key="clean_columns_turnover"
                    )

                    if selected_columns:
                        df_null = pd.DataFrame((df[selected_columns].isna().sum() / len(df) * 100),
                                                    columns=['Percent of Null_Count']).reset_index()
                        df_null.columns = ['Column', 'Percent of Null_Count']

                        df_null['Action'] = df_null['Percent of Null_Count'].apply(
                            lambda p: "No action required" if p == 0 else (
                                "Drop Rows" if p <= 5 else (
                                    "Impute" if p <= 30 else "Drop Column"
                                )
                            )
                        )

                        st.write("Missing Values Analysis:")
                        st.dataframe(df_null)
                        st.info("""
                        Action Guide:
                        - No action required: Column has no missing values
                        - Drop Rows: ≤ 5% missing values - rows will be removed
                        - Impute: 5-30% missing values - will be filled with median/mode
                        - Drop Column: > 30% missing values - column will be removed
                        """)

                        clean_button = st.button("🚀 Run Data Cleaning", key="clean_data_button_turnover")

                        if clean_button:
                            with st.spinner("Running Data Cleaning..."):
                                original_shape = df.shape

                                # Drop rows for selected columns with <= 5% missing values
                                rows_to_drop = df_null[df_null['Action'] == "Drop Rows"]['Column'].tolist()
                                df = df.dropna(subset=rows_to_drop)

                                # Impute columns with <= 30% missing
                                cols_impute = df_null[df_null['Action'] == "Impute"]['Column'].tolist()
                                for col in cols_impute:
                                    if df[col].dtype == 'object':
                                        mode_val = df[col].mode()[0]
                                        df[col].fillna(mode_val, inplace=True)
                                        st.info(f"Filled missing values in '{col}' with mode: {mode_val}")
                                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                                        mode_val = df[col].mode()[0]
                                        df[col].fillna(mode_val, inplace=True)
                                        st.info(f"Filled missing values in '{col}' with mode date")
                                    elif pd.api.types.is_numeric_dtype(df[col]):
                                        median_val = df[col].median()
                                        df[col].fillna(median_val, inplace=True)
                                        st.info(f"Filled missing values in '{col}' with median: {median_val:.2f}")

                                # Drop columns with > 30% missing
                                cols_to_drop = df_null[df_null['Action'] == "Drop Column"]['Column'].tolist()
                                if cols_to_drop:
                                    df.drop(columns=cols_to_drop, inplace=True)
                                    st.warning(f"Dropped columns with >30% missing values: {', '.join(cols_to_drop)}")

                                st.success(f"✅ Data cleaning complete! Rows changed from {original_shape[0]} to {df.shape[0]}")
                                st.session_state['cleaned_df_turnover'] = df

                    else:
                        st.warning("⚠️ Please select at least one column to proceed with cleaning.")

                if 'cleaned_df_turnover' in st.session_state:
                    df_cleaned = st.session_state['cleaned_df_turnover']

                with st.expander("🛡️ Step 3: Outlier Handling", expanded=False):
                    st.info("This step helps identify and handle outliers in numerical columns using the IQR method.")

                    # Check if df_cleaned exists
                    if 'cleaned_df_turnover' not in st.session_state:
                        st.warning("⚠️ Please complete the data cleaning step first.")
                        return
                    
                    df_cleaned = st.session_state['cleaned_df_turnover']
                    
                    # Detect numerical columns
                    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()

                    # Allow user to select from those columns
                    outlier_cols = st.multiselect(
                        "Select numerical columns for outlier analysis:",
                        options=numeric_cols,
                        default=numeric_cols,
                        help="Choose numerical columns to check for outliers",
                        key="outlier_columns_turnover"
                    )

                    if outlier_cols:
                        # Initial side-by-side boxplots BEFORE handling
                        col_before, col_after = st.columns(2)

                        with col_before:
                            st.subheader("Before Handling")
                            for col in outlier_cols:
                                fig = px.box(df_cleaned, y=col, color_discrete_sequence=["#636EFA"], title=f"📦 {col} (Before)")
                                fig.update_traces(line_color='orange', marker_color='red', selector=dict(type='box'))
                                st.plotly_chart(fig, use_container_width=True, key=f"before_plot_turnover_{col}")

                        # Method selector and button
                        method = st.selectbox(
                            "Select outlier handling method:", 
                            ["None", "Remove", "Impute"],
                            help="Choose how to handle detected outliers",
                            key="outlier_method_turnover"
                        )
                        
                        if method != "None":
                            handle_button = st.button("🛠️ Handle Outliers", key="handle_outliers_button_turnover")
                            if handle_button:
                                with st.spinner(f"Handling outliers using {method.lower()}..."):
                                    df_processed = df_cleaned.copy()
                                    total_outliers = 0
                                    for col in outlier_cols:
                                        Q1 = df_processed[col].quantile(0.25)
                                        Q3 = df_processed[col].quantile(0.75)
                                        IQR = Q3 - Q1
                                        lower_bound = Q1 - 1.5 * IQR
                                        upper_bound = Q3 + 1.5 * IQR
                                        outliers = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                                        num_outliers = outliers.sum()
                                        total_outliers += num_outliers

                                        if num_outliers > 0:
                                            st.info(f"Detected {num_outliers} outliers in '{col}'")
                                            if method == "Remove":
                                                df_processed = df_processed[~outliers]
                                                st.info(f"Removed {num_outliers} outliers from '{col}'")
                                            elif method == "Impute":
                                                median_val = df_processed[col].median()
                                                df_processed.loc[outliers, col] = median_val
                                                st.info(f"Imputed {num_outliers} outliers in '{col}' with median: {median_val:.2f}")

                                    st.success(f"✅ Outlier handling complete! Total outliers handled: {total_outliers}")
                                    st.session_state['final_df_turnover'] = df_processed

                        # Always show After plots if outlier handling is done
                        if 'final_df_turnover' in st.session_state and method != "None":
                            df_processed = st.session_state['final_df_turnover']
                            with col_after:
                                st.subheader("After Handling")
                                for col in outlier_cols:
                                    fig = px.box(df_processed, y=col, color_discrete_sequence=["#00CC96"], title=f"📦 {col} (After)")
                                    fig.update_traces(line_color='green', marker_color='blue', selector=dict(type='box'))
                                    st.plotly_chart(fig, use_container_width=True, key=f"after_plot_turnover_{col}")

                # Add column deletion option
                with st.expander("🗑️ Step 4: Column Management", expanded=False):
                    st.info("This step allows you to manage columns in your dataset.")
                    
                    if 'final_df_turnover' in st.session_state:
                        df_final = st.session_state['final_df_turnover']
                        
                        # Display current columns
                        st.subheader("Current Columns")
                        st.dataframe(pd.DataFrame([df_final.columns], index=["Columns"]))

                        
                        # Column deletion interface
                        st.subheader("Delete Columns")
                        columns_to_delete = st.multiselect(
                            "Select columns to delete:",
                            options=df_final.columns,
                            help="Select columns you want to remove from the dataset",
                            key="delete_columns_turnover"
                        )
                        
                        if columns_to_delete:
                            delete_button = st.button("🗑️ Delete Selected Columns", key="delete_columns_button_turnover")
                            if delete_button:
                                df_final = df_final.drop(columns=columns_to_delete)
                                st.session_state['final_df_turnover'] = df_final
                                st.success(f"✅ Successfully deleted columns: {', '.join(columns_to_delete)}")
                                st.dataframe(pd.DataFrame({"Remaining Columns": df_final.columns}))
                    else:
                        st.warning("⚠️ Please complete the previous steps first.")

                if 'final_df_turnover' in st.session_state:
                    df_final = st.session_state['final_df_turnover']
                
                    with st.expander("📊 Attrition Analysis", expanded=True):
                        st.subheader("Understanding Employee Departures")
                        st.markdown("""
                        ### 🎯 What is Attrition Analysis?
                        Attrition analysis helps you understand why employees leave your company. By analyzing patterns in employee departures, 
                        you can identify potential issues and take proactive steps to improve retention.

                        ### 📊 What You'll Find Here:
                        1. **Time-based Analysis**: See when employees are most likely to leave
                        2. **Demographic Insights**: Understand attrition patterns across different groups
                        3. **Trend Analysis**: Track attrition rates over time
                        4. **Correlation Analysis**: Discover relationships between different factors

                        ### 💡 How to Use This Dashboard:
                        - Start by selecting a time period using the filters below
                        - Explore different breakdowns to understand patterns
                        - Use the comparison tool to analyze multiple factors together
                        - Check the correlation heatmap for hidden relationships
                        """)
                
                        df_attrition = df_final[df_final['DOE'].notna()].copy()
                
                        if df_attrition.empty:
                            st.warning("""
                            ⚠️ **No attrition data available**
                            Please ensure your dataset includes:
                            - Date of Exit (DOE) information
                            - Employee demographic data
                            - Department and job level information
                            """)
                        else:
                            # Date column transformations
                            df_attrition['Exit_Year'] = df_attrition['DOE'].dt.year
                            df_attrition['Exit_Month_Num'] = df_attrition['DOE'].dt.month
                            df_attrition['Exit_Month'] = df_attrition['Exit_Month_Num'].apply(lambda x: calendar.month_name[x])
                            df_attrition['Exit_Quarter'] = df_attrition['DOE'].dt.to_period('Q').astype(str)
                            df_attrition['Exit_Month_Year'] = df_attrition['DOE'].dt.to_period('M').astype(str)
                
                    # ------------------- Filter Section -------------------
                    st.subheader("🔍 Filter Attrition Data")
                    with st.expander("Refine by Time Period", expanded=True):
                        st.info("""
                        ### 📅 Multiple Ways to Filter Time Period
                        Choose your preferred method to analyze attrition patterns over time:
                        1. **Quick Filters**: Simple year and month selection
                        2. **Date Range**: Select a specific date range
                        3. **Custom Period**: Choose a custom time period
                        """)
                        
                        # Method selection
                        filter_method = st.radio(
                            "Select Filter Method:",
                            ["Quick Filters", "Date Range", "Custom Period"],
                            horizontal=True,
                            help="Choose how you want to filter the time period"
                        )
                        
                        if filter_method == "Quick Filters":
                            col_filter1, col_filter2 = st.columns(2)
                            with col_filter1:
                                available_years = sorted(df_attrition['Exit_Year'].dropna().unique().astype(int).tolist())
                                selected_year = st.selectbox(
                                    "Select Year:",
                                    ["All"] + [str(year) for year in available_years],
                                    help="Choose a specific year or 'All' to see all years"
                                )
                            with col_filter2:
                                available_months = ["All"] + list(calendar.month_name)[1:]
                                selected_month = st.selectbox(
                                    "Select Month:",
                                    available_months,
                                    help="Choose a specific month or 'All' to see all months"
                                )
                            
                            df_filtered_attrition = df_attrition.copy()
                            filter_string = ""
                            if selected_year != "All":
                                df_filtered_attrition = df_filtered_attrition[df_filtered_attrition['Exit_Year'] == int(selected_year)]
                                filter_string += f" for Year: {selected_year}"
                            if selected_month != "All":
                                df_filtered_attrition = df_filtered_attrition[df_filtered_attrition['Exit_Month'] == selected_month]
                                if filter_string:
                                    filter_string += f" and Month: {selected_month}"
                                else:
                                    filter_string += f" for Month: {selected_month}"
                        
                        elif filter_method == "Date Range":
                            col1, col2 = st.columns(2)
                            with col1:
                                min_date = df_attrition['DOE'].min()
                                max_date = df_attrition['DOE'].max()
                                start_date = st.date_input(
                                    "Start Date",
                                    min_date,
                                    min_value=min_date,
                                    max_value=max_date,
                                    help="Select the start date for analysis"
                                )
                            with col2:
                                end_date = st.date_input(
                                    "End Date",
                                    max_date,
                                    min_value=min_date,
                                    max_value=max_date,
                                    help="Select the end date for analysis"
                                )
                            
                            df_filtered_attrition = df_attrition[
                                (df_attrition['DOE'].dt.date >= start_date) & 
                                (df_attrition['DOE'].dt.date <= end_date)
                            ].copy()
                            filter_string = f" from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                        
                        else:  # Custom Period
                            col1, col2 = st.columns(2)
                            with col1:
                                available_years = sorted(df_attrition['Exit_Year'].dropna().unique().astype(int).tolist())
                                year_range = st.slider(
                                    "Select Year Range",
                                    min_value=min(available_years),
                                    max_value=max(available_years),
                                    value=(min(available_years), max(available_years)),
                                    help="Slide to select a range of years"
                                )
                            with col2:
                                month_range = st.multiselect(
                                    "Select Months",
                                    options=list(calendar.month_name)[1:],
                                    default=list(calendar.month_name)[1:],
                                    help="Select specific months to analyze"
                                )
                            
                            df_filtered_attrition = df_attrition[
                                (df_attrition['Exit_Year'] >= year_range[0]) & 
                                (df_attrition['Exit_Year'] <= year_range[1]) &
                                (df_attrition['Exit_Month'].isin(month_range))
                            ].copy()
                            filter_string = f" for Years {year_range[0]}-{year_range[1]} and Months: {', '.join(month_range)}"
                        
                        # Show filter summary
                        st.info(f"""
                        ### 🔍 Current Filter Settings
                        - **Method**: {filter_method}
                        - **Time Period**: {filter_string}
                        - **Records Found**: {len(df_filtered_attrition)} out of {len(df_attrition)} total records
                        """)
                        
                if df_filtered_attrition.empty:
                    st.warning("""
                    ⚠️ **No data matches your selected filters**
                    Try adjusting your filter settings to find matching records.
                    """)
                else:
                    # Add advanced filters
                    with st.expander("🔍 Advanced Filters", expanded=False):
                        st.info("""
                        ### Additional Filtering Options
                        Filter the data further by demographic and categorical variables.
                        These filters will affect all visualizations below.
                        """)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # Department filter
                            departments = ["All"] + sorted(df_filtered_attrition['Department'].unique().tolist())
                            selected_dept = st.multiselect(
                                "Select Departments",
                                options=departments,
                                default=["All"],
                                help="Filter by specific departments"
                            )
                            
                            # Job Level filter
                            job_levels = ["All"] + sorted(df_filtered_attrition['Job level'].unique().tolist())
                            selected_level = st.multiselect(
                                "Select Job Levels",
                                options=job_levels,
                                default=["All"],
                                help="Filter by specific job levels"
                            )
                        
                        with col2:
                            # Gender filter
                            genders = ["All"] + sorted(df_filtered_attrition['Gender'].unique().tolist())
                            selected_gender = st.multiselect(
                                "Select Gender",
                                options=genders,
                                default=["All"],
                                help="Filter by gender"
                            )
                            
                            # DEI Group filter
                            dei_groups = ["All"] + sorted(df_filtered_attrition['DEIGroup'].unique().tolist())
                            selected_dei = st.multiselect(
                                "Select DEI Groups",
                                options=dei_groups,
                                default=["All"],
                                help="Filter by DEI groups"
                            )
                        
                        # Apply advanced filters
                        df_filtered = df_filtered_attrition.copy()
                        if "All" not in selected_dept:
                            df_filtered = df_filtered[df_filtered['Department'].isin(selected_dept)]
                        if "All" not in selected_level:
                            df_filtered = df_filtered[df_filtered['Job level'].isin(selected_level)]
                        if "All" not in selected_gender:
                            df_filtered = df_filtered[df_filtered['Gender'].isin(selected_gender)]
                        if "All" not in selected_dei:
                            df_filtered = df_filtered[df_filtered['DEIGroup'].isin(selected_dei)]
                        
                        # Save filtered data to session state for use in other pages
                        st.session_state['filtered_attrition_data'] = df_filtered
                        st.session_state['attrition_filters'] = {
                            'departments': selected_dept,
                            'job_levels': selected_level,
                            'genders': selected_gender,
                            'dei_groups': selected_dei,
                            'time_period': filter_string
                        }
                        
                        # Show filter summary
                        st.info(f"""
                        ### 🔍 Current Filter Settings
                        - **Departments**: {', '.join(selected_dept) if "All" not in selected_dept else "All"}
                        - **Job Levels**: {', '.join(selected_level) if "All" not in selected_level else "All"}
                        - **Gender**: {', '.join(selected_gender) if "All" not in selected_gender else "All"}
                        - **DEI Groups**: {', '.join(selected_dei) if "All" not in selected_dei else "All"}
                        - **Records Found**: {len(df_filtered)} out of {len(df_filtered_attrition)} records
                        """)
                    
                    # Update all visualizations with filtered data
                    if len(df_filtered) > 0:
                        # ------------------- Breakdown Charts -------------------
                        st.subheader(f"📉 Attrition Breakdown {filter_string}")
                        st.markdown("""
                        ### 📊 Understanding the Breakdown Charts
                        These charts show how attrition is distributed across different employee categories.
                        The charts displayed are based on your current filter selections.
                        """)
                        
                        # Determine which breakdowns to show based on active filters
                        active_filters = []
                        if "All" not in selected_dept:
                            active_filters.append("Department")
                        if "All" not in selected_level:
                            active_filters.append("Job level")
                        if "All" not in selected_gender:
                            active_filters.append("Gender")
                        if "All" not in selected_dei:
                            active_filters.append("DEIGroup")
                        
                        # Always show time-based breakdowns
                        time_breakdowns = {
                            "Exit_Year": "Year-wise Attrition",
                            "Exit_Month": "Month-wise Attrition",
                            "Exit_Quarter": "Quarter-wise Attrition"
                        }
                        
                        # Create bar charts for active filters and time breakdowns
                        plot_data = []
                        
                        # Add time-based breakdowns first
                        for col, title in time_breakdowns.items():
                            if col in df_filtered.columns:
                                counts = df_filtered[col].value_counts().sort_values(ascending=False)
                                if not counts.empty:
                                    fig = px.bar(
                                        x=counts.index,
                                        y=counts.values,
                                        title=title,
                                        labels={col: col, "y": "Number of Exited Employees"},
                                        color_discrete_sequence=px.colors.qualitative.Set2,
                                    )
                                    fig.update_traces(
                                        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
                                        text=counts.values,
                                        textposition='outside'
                                    )
                                    fig.update_layout(
                                        uniformtext_minsize=8,
                                        uniformtext_mode='hide',
                                        height=400,
                                        showlegend=False
                                    )
                                    plot_data.append((fig, title))
                        
                        # Add breakdowns for active filters
                        for col in active_filters:
                            if col in df_filtered.columns:
                                counts = df_filtered[col].value_counts().sort_values(ascending=False)
                                if not counts.empty:
                                    title = f"{col}-wise Attrition"
                                    fig = px.bar(
                                        x=counts.index,
                                        y=counts.values,
                                        title=title,
                                        labels={col: col, "y": "Number of Exited Employees"},
                                        color_discrete_sequence=px.colors.qualitative.Set2,
                                    )
                                    fig.update_traces(
                                        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
                                        text=counts.values,
                                        textposition='outside'
                                    )
                                    fig.update_layout(
                                        uniformtext_minsize=8,
                                        uniformtext_mode='hide',
                                        height=400,
                                        showlegend=False
                                    )
                                    plot_data.append((fig, title))
                        
                        # Display in rows of two columns
                        if plot_data:
                            for i in range(0, len(plot_data), 2):
                                col1, col2 = st.columns(2)
                                
                                # First chart
                                fig1, title1 = plot_data[i]
                                with col1:
                                    if fig1:
                                        st.plotly_chart(fig1, use_container_width=True)
                                
                                # Second chart (if exists)
                                if i + 1 < len(plot_data):
                                    fig2, title2 = plot_data[i + 1]
                                    with col2:
                                        if fig2:
                                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.info("No breakdown data available based on current filters.")

                        # ------------------- Comparative Analysis -------------------
                        with st.expander("📊 Comparative Analysis", expanded=False):
                            st.info("""
                            ### Compare Different Time Periods
                            Analyze how attrition patterns change between different time periods.
                            """)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                compare_method = st.radio(
                                    "Comparison Method",
                                    ["Year-over-Year", "Quarter-over-Quarter", "Month-over-Month"],
                                    horizontal=True
                                )
                            
                            with col2:
                                if compare_method == "Year-over-Year":
                                    years = sorted(df_filtered['Exit_Year'].unique())
                                    year1, year2 = st.select_slider(
                                        "Select Years to Compare",
                                        options=years,
                                        value=(years[0], years[-1])
                                    )
                                    df_compare1 = df_filtered[df_filtered['Exit_Year'] == year1]
                                    df_compare2 = df_filtered[df_filtered['Exit_Year'] == year2]
                                    title = f"Year-over-Year Comparison: {year1} vs {year2}"
                                
                                elif compare_method == "Quarter-over-Quarter":
                                    quarters = sorted(df_filtered['Exit_Quarter'].unique())
                                    quarter1, quarter2 = st.select_slider(
                                        "Select Quarters to Compare",
                                        options=quarters,
                                        value=(quarters[0], quarters[-1])
                                    )
                                    df_compare1 = df_filtered[df_filtered['Exit_Quarter'] == quarter1]
                                    df_compare2 = df_filtered[df_filtered['Exit_Quarter'] == quarter2]
                                    title = f"Quarter-over-Quarter Comparison: {quarter1} vs {quarter2}"
                                
                                else:  # Month-over-Month
                                    months = sorted(df_filtered['Exit_Month'].unique())
                                    month1, month2 = st.select_slider(
                                        "Select Months to Compare",
                                        options=months,
                                        value=(months[0], months[-1])
                                    )
                                    df_compare1 = df_filtered[df_filtered['Exit_Month'] == month1]
                                    df_compare2 = df_filtered[df_filtered['Exit_Month'] == month2]
                                    title = f"Month-over-Month Comparison: {month1} vs {month2}"
                            
                            # Create comparison chart
                            if not df_compare1.empty and not df_compare2.empty:
                                fig_compare = go.Figure()
                                fig_compare.add_trace(go.Bar(
                                    x=['Period 1', 'Period 2'],
                                    y=[len(df_compare1), len(df_compare2)],
                                    name='Total Attrition',
                                    marker_color=['#636EFA', '#EF553B']
                                ))
                                fig_compare.update_layout(
                                    title=title,
                                    xaxis_title="Time Period",
                                    yaxis_title="Number of Attritions",
                                    showlegend=True
                                )
                                st.plotly_chart(fig_compare, use_container_width=True)
                            
                            # ------------------- Seasonal Analysis -------------------
                            with st.expander("🌦️ Seasonal Analysis", expanded=False):
                                st.info("""
                                ### Analyze Seasonal Patterns
                                Identify patterns in attrition across different seasons and months.
                                """)
                                
                                # Monthly trend with seasonal decomposition
                                monthly_counts = df_filtered.groupby('Exit_Month').size()
                                fig_seasonal = px.line(
                                    x=monthly_counts.index,
                                    y=monthly_counts.values,
                                    title="Monthly Attrition Pattern",
                                    labels={'x': 'Month', 'y': 'Number of Attritions'}
                                )
                                fig_seasonal.update_layout(
                                    xaxis=dict(
                                        categoryorder='array',
                                        categoryarray=list(calendar.month_name)[1:]
                                    )
                                )
                                st.plotly_chart(fig_seasonal, use_container_width=True)
                                
                                # Quarterly breakdown
                                quarterly_counts = df_filtered.groupby('Exit_Quarter').size()
                                fig_quarterly = px.pie(
                                    values=quarterly_counts.values,
                                    names=quarterly_counts.index,
                                    title="Quarterly Distribution",
                                    hole=0.3
                                )
                                st.plotly_chart(fig_quarterly, use_container_width=True)
                            
                            # ------------------- Export Options -------------------
                            with st.expander("📤 Export Options", expanded=False):
                                st.info("""
                                ### Export Analysis Results
                                Download the filtered data and visualizations for further analysis.
                                """)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    # Export filtered data
                                    csv = df_filtered.to_csv(index=False)
                                    st.download_button(
                                        "📥 Download Filtered Data",
                                        csv,
                                        "filtered_attrition_data.csv",
                                        "text/csv",
                                        key="download_filtered_data"
                                    )
                                
                                with col2:
                                    # Create HTML content with all visualizations
                                    html_content = """
                                    <html>
                                    <head>
                                        <title>Attrition Analysis Visualizations</title>
                                        <style>
                                            body { font-family: Arial, sans-serif; margin: 20px; }
                                            .chart { margin: 20px 0; }
                                            h1, h2 { color: #333; }
                                        </style>
                                    </head>
                                    <body>
                                        <h1>Attrition Analysis Visualizations</h1>
                                    """
                                    
                                    # Add all visualizations to HTML content
                                    if 'fig_compare' in locals():
                                        html_content += f"""
                                        <div class="chart">
                                            <h2>Comparative Analysis</h2>
                                            {fig_compare.to_html(full_html=False)}
                                        </div>
                                        """
                                    
                                    if 'fig_seasonal' in locals():
                                        html_content += f"""
                                        <div class="chart">
                                            <h2>Seasonal Analysis</h2>
                                            {fig_seasonal.to_html(full_html=False)}
                                        </div>
                                        """
                                    
                                    if 'fig_quarterly' in locals():
                                        html_content += f"""
                                        <div class="chart">
                                            <h2>Quarterly Distribution</h2>
                                            {fig_quarterly.to_html(full_html=False)}
                                        </div>
                                        """
                                    
                                    # Add breakdown charts
                                    if 'plot_data' in locals():
                                        html_content += """
                                        <div class="chart">
                                            <h2>Breakdown Analysis</h2>
                                        """
                                        for fig, title in plot_data:
                                            if fig is not None:
                                                html_content += f"""
                                                <div class="chart">
                                                    <h3>{title}</h3>
                                                    {fig.to_html(full_html=False)}
                                                </div>
                                                """
                                        html_content += "</div>"
                                    
                                    html_content += """
                                    </body>
                                    </html>
                                    """
                                    
                                    # Export visualizations
                                    st.download_button(
                                        "📊 Export Visualizations",
                                        html_content,
                                        "attrition_visualizations.html",
                                        "text/html",
                                        key="download_visualizations"
                                    )
                     
            except KeyError as e:
                st.error(f"❌ Error: Column '{e}' not found. Please check the template.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                
            
    


