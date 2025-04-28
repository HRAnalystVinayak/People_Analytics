
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from datetime import datetime
import calendar
import plotly.graph_objects as go

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

                with st.expander("📊 Data Preview", expanded=False):
                    st.dataframe(df.head())
                    st.info(f"ℹ️ **Data Summary:** {df.shape[0]} rows and {df.shape[1]} columns.")
                
                with st.expander("🔢 Numeric Data Summary", expanded=False):
                    st.dataframe(df.describe(include=[np.number]))
                
                with st.expander("🔤 Categorical (Object) Data Summary", expanded=False):
                    st.dataframe(df.describe(include=[object]))

                with st.expander("🧹 Step 2: Data Cleaning", expanded=False):
                    st.info("""This section handles missing values based on their percentage.""")
                    df_null = pd.DataFrame((df.isna().sum()/len(df)*100), columns = ['Percent of Null_Count']).reset_index()
                    df_null['Action'] = df_null['Percent of Null_Count'].apply(lambda p: "No action required" if p == 0 else ("Drop Rows" if p <= 5 else ("Impute" if p <= 30 else "Drop Column")))
                    st.dataframe(df_null)
                    clean_button = st.button("🚀 Run Data Cleaning", key="clean_data_button")

                    if clean_button:
                        with st.spinner("Running Data Cleaning...."):
                            original_rows = df.shape[0]
                            original_columns = df.shape[1]
                            rows_to_drop = df_null[df_null['Action']== "Drop Rows"]['index'].tolist()
                            df = df.dropna(subset = rows_to_drop)

                            cols_impute = df_null[df_null['Action'] == "Impute"]['index'].tolist()
                            for col in cols_impute:
                                if df[col].dtype == 'object':
                                    df[col].fillna(df[col].mode()[0], inplace=True)
                                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                                    df[col].fillna(df[col].mode()[0], inplace=True)
                                elif pd.api.types.is_numeric_dtype(df[col]):
                                    if col in ['Tenure']:
                                        df[col].fillna(df[col].median(), inplace=True)
                                    elif col in ['Age']:
                                        df[col].fillna(df[col].mean(), inplace=True)
                                    else:
                                        df[col].fillna(df[col].median(), inplace=True)  
                        
                        cols_to_drop = df_null[df_null['Action'] == "Drop Column"]['index'].tolist()
                        df.drop(columns=cols_to_drop, inplace=True)
                        st.success("✅ Data cleaning complete!")
                        st.session_state['cleaned_df'] = df
                
                if 'cleaned_df' in st.session_state:
                    df_cleaned = st.session_state['cleaned_df']

                    with st.expander("🛡️ Step 3: Outlier Handling", expanded=False):
                        st.info("Detect and handle outliers in numerical columns using the IQR method.")
                        numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
                    
                        if numeric_cols:
                            col_before, col_after = st.columns(2)  # Create two columns side by side
                    
                            # Before Handling (col_before)
                            with col_before:
                                st.subheader("Before Handling")
                                for col in numeric_cols:
                                    fig = px.box(df_cleaned, y=col, color_discrete_sequence=["#636EFA"], title=f"📦 Box Plot of {col}")
                                    fig.update_traces(line_color='orange', marker_color='red', selector=dict(type='box'))
                                    st.plotly_chart(fig, use_container_width=True, key=f"before_plot_{col}")
                    
                            # Outlier Handling Method
                            method = st.selectbox("Select Outlier Handling", ["None", "Remove", "Impute"], key="outlier_method")
                            handle_button = st.button("🛠️ Handle Outliers", key="handle_outliers_button")
                            df_processed = df_cleaned.copy()
                    
                            if handle_button and method != "None":
                                with st.spinner(f"Handling outliers using {method.lower()}..."):
                                    for col in numeric_cols:
                                        Q1 = df_processed[col].quantile(0.25)
                                        Q3 = df_processed[col].quantile(0.75)
                                        IQR = Q3 - Q1
                                        lower_bound = Q1 - 1.5 * IQR
                                        upper_bound = Q3 + 1.5 * IQR
                                        outliers = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                                        num_outliers = outliers.sum()
                                        if num_outliers > 0:
                                            st.info(f"Detected {num_outliers} outliers in '{col}'.")
                                            if method == "Remove":
                                                df_processed = df_processed[~outliers]
                                                st.info(f"Removed {num_outliers} outliers from '{col}'.")
                                            elif method == "Impute":
                                                median_val = df_processed[col].median()
                                                df_processed.loc[outliers, col] = median_val
                                                st.info(f"Imputed {num_outliers} outliers in '{col}' with median.")
                                    st.success("✅ Outlier handling complete!")
                                    st.session_state['final_df'] = df_processed
                    
                            # After Handling (col_after)
                                with col_after:
                                    st.subheader("After Handling")
                                    for col in numeric_cols:
                                        fig = px.box(df_processed, y=col, color_discrete_sequence=["#636EFA"], title=f"📦 Box Plot of {col}")
                                        fig.update_traces(line_color='orange', marker_color='red', selector=dict(type='box'))
                                        st.plotly_chart(fig, use_container_width=True, key=f"after_plot_{col}")

                       
                            elif method == "None":
                                st.info("Select a method to handle outliers to see the 'After' box plots.")
                                st.session_state['final_df'] = df_processed

                        
                        else:
                            st.warning("⚠️ No numerical columns to detect outliers.")    
            
            
                
                if 'final_df' in st.session_state:
                    df_final = st.session_state['final_df']
                
                    with st.expander("📊 Attrition Analysis", expanded=True):
                        st.subheader("Understanding Employee Departures")
                        st.markdown("""
                        This section helps you analyze employee attrition, which is the rate at which employees leave the company.
                        Understanding why employees leave is crucial for improving employee retention and overall organizational health.
                        We'll explore attrition patterns based on various factors. The analysis focuses on employees who have a recorded Date of Exit (DOE).
                        """)
                
                        df_attrition = df_final[df_final['DOE'].notna()].copy()
                
                        if df_attrition.empty:
                            st.warning("⚠️ No attrition data available. Please ensure the dataset includes 'Date of Exit' (DOE) information.")
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
                        col_filter1, col_filter2 = st.columns(2)
                        with col_filter1:
                            available_years = sorted(df_attrition['Exit_Year'].dropna().unique().astype(int).tolist())
                            selected_year = st.selectbox("Year:", ["All"] + [str(year) for year in available_years])
                        with col_filter2:
                            available_months = ["All"] + list(calendar.month_name)[1:]
                            selected_month = st.selectbox("Month:", available_months)
        
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
        
                    if df_filtered_attrition.empty and (selected_year != "All" or selected_month != "All"):
                        st.info("No attrition data matches the selected time period. Try adjusting the year or month filters.")
                    elif not df_filtered_attrition.empty:
                    # ------------------- Breakdown Charts -------------------
                        st.subheader(f"📉 Attrition Breakdown {filter_string}")
                        st.markdown("Explore how attrition is distributed across different employee categories.")
        
                        analysis_cols = {
                            "Gender": "Gender-wise Attrition",
                            "Exit_Year": "Year-wise Attrition",
                            "Exit_Month": "Month-wise Attrition",
                            "Department": "Department-wise Attrition",
                            "DEIGroup": "DEI Group-wise Attrition",
                            "Country": "Country-wise Attrition",
                            "Job level": "Job Level-wise Attrition",
                        }

                        # Create bar charts in rows of 2 columns for better alignment
                    plot_data = []
                    for col, title in analysis_cols.items():
                        if col in df_filtered_attrition.columns:
                            counts = df_filtered_attrition[col].value_counts().sort_values(ascending=False)
                            if not counts.empty:
                                fig = px.bar(
                                    x=counts.index,
                                    y=counts.values,
                                    title=title,
                                    labels={col: col, "y": "Number of Exited Employees"},
                                    color_discrete_sequence=px.colors.qualitative.Set2,
                                )
                                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                                # Optional: remove the invisible bar trace unless you need extra spacing
                                # fig.add_trace(go.Bar(x=counts.index, y=counts.values, text=counts.values,
                                #                      textposition='outside', marker_color='rgba(0,0,0,0)'))
                                plot_data.append((fig, title))
                            else:
                                plot_data.append((None, title))
                    
                    # Display in rows of two columns
                    for i in range(0, len(plot_data), 2):
                        col1, col2 = st.columns(2)
                        
                        # First chart
                        fig1, title1 = plot_data[i]
                        with col1:
                            if fig1:
                                st.plotly_chart(fig1, use_container_width=True)
                            else:
                                st.info(f"No attrition data available for {title1} based on current filters.")
                        
                        # Second chart (if exists)
                        if i + 1 < len(plot_data):
                            fig2, title2 = plot_data[i + 1]
                            with col2:
                                if fig2:
                                    st.plotly_chart(fig2, use_container_width=True)
                                else:
                                    st.info(f"No attrition data available for {title2} based on current filters.")
                    
                                    
                    
                    st.markdown(f"### 📊 Compare Attrition Across Dimensions {filter_string}")
                    with st.expander("Compare Categories", expanded=True):
                        group_cols = [
                            'Gender', 'Exit_Year', 'Exit_Month',
                            'Department', 'DEIGroup', 'Country', 'Job level',
                        ]
                        valid_group_cols = [col for col in group_cols if col in df_filtered_attrition.columns]
    
                        if valid_group_cols:
                            col1, col2 = st.columns(2)
                            with col1:
                                primary_group = st.selectbox("Group By (X-axis):", valid_group_cols, index=0)
                            with col2:
                                secondary_group = st.selectbox("Compare By (Color):", ["None"] + [col for col in valid_group_cols if col != primary_group])
    
                            if secondary_group != "None":
                                grouped_data = df_filtered_attrition.groupby([primary_group, secondary_group]).size().reset_index(name='Attrition Count')
                                if not grouped_data.empty:
                                    fig_grouped = px.bar(
                                        grouped_data,
                                        x=primary_group,
                                        y='Attrition Count',
                                        color=secondary_group,
                                        barmode='group',
                                        title=f"Attrition: {primary_group} by {secondary_group}",
                                        color_discrete_sequence=px.colors.qualitative.Bold,
                                        text='Attrition Count' # Display count on bars
                                    )
                                    fig_grouped.update_traces(textposition='outside', marker = dict(line = dict(color = 'white', width = 1.5)))
                                    st.plotly_chart(fig_grouped, use_container_width=True)
                            else:
                                single_group_data = df_filtered_attrition.groupby(primary_group).size().reset_index(name='Attrition Count')
                                if not single_group_data.empty:
                                    fig_single = px.bar(
                                        single_group_data,
                                        x=primary_group,
                                        y='Attrition Count',
                                        title=f"Attrition by {primary_group}",
                                        color_discrete_sequence=px.colors.qualitative.Pastel1,
                                        text='Attrition Count' # Display count on bars
                                    )
                                    fig_single.update_traces(textposition='outside')
                                    st.plotly_chart(fig_single, use_container_width=True)
                        else:
                            st.info("No suitable columns available for comparison.")
    
                   # ------------------- Correlation Heatmap -------------------
                    st.markdown(f"### 🌡️ Explore Relationships with a Correlation Heatmap {filter_string}")
                    show_heatmap = st.checkbox("Show Correlation Heatmap", value=False)
                    if show_heatmap:
                        numerical_df = df_filtered_attrition.select_dtypes(include=np.number)
                        if not numerical_df.empty:
                            corr = numerical_df.corr()
                            fig_heatmap = px.imshow(
                                corr,
                                labels=dict(x="Feature", y="Feature", color="Correlation"),
                                x=numerical_df.columns,
                                y=numerical_df.columns,
                                color_continuous_scale="Sunsetdark",
                                title="Correlation Matrix of Numerical Features",
                            )
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                        else:
                            st.info("No numerical features available to display a correlation heatmap.")
                
                    st.subheader(f"📈 Attrition Trends Over Time {filter_string}")
                    with st.expander("Visualize Trends", expanded=True):
                        if 'Exit_Year' in df_filtered_attrition.columns:
                            yearly = df_filtered_attrition['Exit_Year'].value_counts().sort_index()
                            df_yearly = pd.DataFrame({'Year': yearly.index, 'Attrition Count': yearly.values})
                            fig_yearly = px.line(df_yearly, x='Year', y='Attrition Count', title="Yearly Attrition Trend", markers=True,
                                                 color_discrete_sequence=['#636EFA'])
                            st.plotly_chart(fig_yearly, use_container_width=True)
    
                        if 'Exit_Quarter' in df_filtered_attrition.columns:
                            quarterly = df_filtered_attrition['Exit_Quarter'].value_counts().sort_index()
                            fig_quarter = px.line(x=quarterly.index, y=quarterly.values, title="Quarterly Attrition Trend", markers=True,
                                                  labels={'x': 'Quarter', 'y': 'Attrition Count'},
                                                  color_discrete_sequence=['#EF553B'])
                            st.plotly_chart(fig_quarter, use_container_width=True)
    
                        if 'Exit_Month_Year' in df_filtered_attrition.columns:
                            monthly = df_filtered_attrition['Exit_Month_Year'].value_counts().sort_index()
                            fig_month = px.line(x=monthly.index, y=monthly.values, title="Monthly Attrition Trend", markers=True,
                                                labels={'x': 'Month-Year', 'y': 'Attrition Count'},
                                                color_discrete_sequence=['#00CC96'])
                            st.plotly_chart(fig_month, use_container_width=True)
        

            
            
            except KeyError as e:
                st.error(f"❌ Error: Column '{e}' not found. Please check the template.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
        else:
            st.info("👆 Please upload a file to begin the analysis.")
    
if __name__ == "__main__":
    turnover_tab(tabs)