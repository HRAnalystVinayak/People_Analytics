
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from datetime import datetime
import calendar
import matplotlib.pyplot as plt
import seaborn as sns

def recruitment_metrics(tabs):
    with tabs[2]:
        st.header("Please Update the data in the specified format.")
        st.info("""
        - Kindly avoid adding any extra columns to the sheet, as they will not be processed.
        - If you do not have data for a required column, you may leave it blank. 
          Empty columns will be automatically removed during data processing.
        """)

        
        st.download_button(
            label="📥 Download Recruitment Analysis Template",
            data="Department,Position Name,Job Level,DEI,Country,Date of Requisition Received,Source of Hire,Gender,Candidate Experience (Years),Interviewed or Not,Shortlisted or Not,Offer Accepted or Not,Joined or Dropout\nPurchase/HR,Cordinator/Analyst,1 to 5,yes/no,India/Chaina,2025-02-02,Job_portal/Referral,Male/Female,1 to 2 (numeric),yes/no,yes/no,joined/dropout",
            file_name="attrition_template.csv",
            mime="text/csv", key = "recruitment"
        )
        
        
        st.header(" Recruitment_metrics Analysis Dashboard")
        st.subheader("1. Upload and Prepare Data")

        upload_file = st.file_uploader("⬆️ Upload Diversity Analysis CSV File", type='csv', key="recruitment_metrics")
        if upload_file is not None:
            try:
                df = pd.read_csv(upload_file, parse_dates=['Date of Requisition Received'])
                df['Date of Requisition Received'] = pd.to_datetime(df['Date of Requisition Received'], errors='coerce')
                today = datetime.now()
                df['Days'] = (today - df['Date of Requisition Received']).dt.days

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
                    clean_button = st.button("🚀 Run Data Cleaning", key="clean_data_button_tab2")

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
                                    st.plotly_chart(fig, use_container_width=True, key=f"before_plot_tab2{col}")
                    
                            # Outlier Handling Method
                            method = st.selectbox("Select Outlier Handling", ["None", "Remove", "Impute"], key="outlier_method_tab2")
                            handle_button = st.button("🛠️ Handle Outliers", key="handle_outliers_button_tab2")
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
                    df_final.columns = df_final.columns.str.strip()
                    
                    with st.expander("Recruitment Analysis 🚀", expanded=True):
                        st.subheader("Understanding the Recruitment Process")
                        st.markdown("""
                        This section analyzes the recruitment process, focusing on conversion rates at each stage.
                        We'll look at how candidates progress from initial application to joining the company.
                        """)
                
                        # Convert date columns safely
                        if 'Date of Requisition Received' in df_final.columns:
                            df_final['Date of Requisition Received'] = pd.to_datetime(df_final['Date of Requisition Received'], errors='coerce')
                
                        st.subheader("📊 Overall Recruitment Funnel Conversion")
                
                        # Step 1: Overall counts
                        total_requisitions = len(df_final)
                        interviewed = df_final[df_final['Interviewed or Not'].astype(str).str.contains('Yes', case=False)]
                        interviewed_count = len(interviewed)
                
                        shortlisted = interviewed[interviewed['Shortlisted or Not'].astype(str).str.contains('Yes', case=False)]
                        shortlisted_count = len(shortlisted)
                
                        offered = shortlisted[shortlisted['Offer Accepted or Not'].astype(str).str.contains('Yes', case=False)]
                        offer_accepted_count = len(offered)
                
                        joined = offered[offered['Joined or Dropout'].astype(str).str.contains('Joined', case=False)]
                        joined_count = len(joined)
                
                        col1,col2 = st.columns(2)

                        with col1:
                        
                            st.markdown(f"**Total Requisitions Received**: {total_requisitions:,}")
                            st.markdown(f"**Interviewed**: {interviewed_count:,} ({(interviewed_count / total_requisitions * 100 if total_requisitions else 0):.2f}%)")
                            st.markdown(f"**Shortlisted**: {shortlisted_count:,} ({(shortlisted_count / interviewed_count * 100 if interviewed_count else 0):.2f}%)")
                        
                        with col2:
                        
                            st.markdown(f"**Offers Accepted**: {offer_accepted_count:,} ({(offer_accepted_count / shortlisted_count * 100 if shortlisted_count else 0):.2f}%)")
                            st.markdown(f"**Joined**: {joined_count:,} ({(joined_count / offer_accepted_count * 100 if offer_accepted_count else 0):.2f}%)")

    
                # --- CONFIG ---
                analysis_options = [
                    "Time to Hire",
                    "Total Profiles",
                    "Interview Ratio",
                    "Offer Ratio",
                    "Join Ratio",
                    "Source Effectiveness",
                    "Funnel Conversion Analysis",
                    "Aging of Requisitions"
                ]
                
                breakdown_options = ['Department', 'Country', 'Position Name', 'Job Level', None]
                comparison_options = [None, 'Gender', 'DEI', 'Source of Hire', 'Country']
                
                # --- START EXPANDER ---
                with st.expander("Recruitment Analyses", expanded=True):
                    selected_analysis = st.selectbox("Select an analysis to view:", analysis_options)
                    group_by = st.selectbox("Break down by:", breakdown_options)
                    compare_by = st.selectbox("Compare by (optional):", comparison_options)
                
                    if selected_analysis and group_by:
                        st.subheader(f"📊 {selected_analysis} Analysis")
                
                        # Filter & Define Metric Logic
                        df = df_final.copy()
                
                        if selected_analysis == "Time to Hire":
                            df = df[df['Joined or Dropout'].astype(str).str.contains('Joined', case=False)]
                            metric_column = 'Days'
                            metrics = ['min', 'mean', 'median', 'max', 'count', 'std']
                            rename_map = {'min': 'Min', 'mean': 'Mean', 'median': 'Median', 'max': 'Max', 'count': 'Count', 'std': 'Std Dev'}
                
                        elif selected_analysis == "Total Profiles":
                            metric_column = 'Candidate ID'
                            df[metric_column] = 1
                            metrics = ['count']
                            rename_map = {'count': 'Total Profiles'}
                
                        elif selected_analysis == "Interview Ratio":
                            metric_column = 'Interviewed'
                            df[metric_column] = df['Interviewed'].astype(int)
                            metrics = ['mean']
                            rename_map = {'mean': 'Interview Ratio'}
                
                        elif selected_analysis == "Offer Ratio":
                            metric_column = 'Offered'
                            df[metric_column] = df['Offered'].astype(int)
                            metrics = ['mean']
                            rename_map = {'mean': 'Offer Ratio'}
                
                        elif selected_analysis == "Join Ratio":
                            metric_column = 'Joined'
                            df[metric_column] = df['Joined'].astype(int)
                            metrics = ['mean']
                            rename_map = {'mean': 'Join Ratio'}
                
                        elif selected_analysis == "Source Effectiveness":
                            metric_column = 'Joined'
                            df[metric_column] = df['Joined'].astype(int)
                            group_by = 'Source of Hire' if not group_by else group_by
                            metrics = ['mean', 'count']
                            rename_map = {'mean': 'Effectiveness (Join %)', 'count': 'Candidates'}
                
                        elif selected_analysis == "Funnel Conversion Analysis":
                            metric_column = 'Funnel Stage'
                            df[metric_column] = df[metric_column].astype(str)
                            df['Count'] = 1
                            grouped = df.groupby([group_by, compare_by, metric_column] if compare_by else [group_by, metric_column])['Count'].sum().reset_index()
                            fig = px.bar(
                                grouped,
                                x=group_by,
                                y='Count',
                                color=metric_column,
                                barmode='stack',
                                facet_col=compare_by if compare_by else None,
                                title=f"Funnel Conversion by {group_by}" + (f" and {compare_by}" if compare_by else "")
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.stop()
                
                        elif selected_analysis == "Aging of Requisitions":
                            metric_column = 'Requisition Aging Days'
                            df[metric_column] = df[metric_column].astype(int)
                            metrics = ['min', 'mean', 'median', 'max', 'count', 'std']
                            rename_map = {'min': 'Min', 'mean': 'Mean', 'median': 'Median', 'max': 'Max', 'count': 'Count', 'std': 'Std Dev'}
                
                        # --- Table + Graph ---
                        if compare_by:
                            grouped = df.groupby([group_by, compare_by])[metric_column]
                        else:
                            grouped = df.groupby(group_by)[metric_column]
                
                        stats = grouped.agg(metrics).reset_index().rename(columns=rename_map)
                
                        # 🧾 Show Table
                        st.markdown("### 📋 Summary Table")
                        st.dataframe(stats, use_container_width=True)
                
                        # 📊 Plot
                        melted = stats.melt(
                            id_vars=[group_by, compare_by] if compare_by else [group_by],
                            var_name='Metric',
                            value_name='Value'
                        )
                        fig = px.bar(
                            melted,
                            x=group_by,
                            y='Value',
                            color='Metric',
                            barmode='group',
                            facet_col=compare_by if compare_by else None,
                            title=f"{selected_analysis} by {group_by}" + (f" and {compare_by}" if compare_by else "")
                        )
                        fig.update_layout(height=600, xaxis_title=group_by, yaxis_title='Value')
                        st.markdown("### 📊 Visual Summary")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select a valid breakdown option.")
                
                                
                




            



            
            
                    
            
            except KeyError as e:
                st.error(f"❌ Error: Column '{e}' not found. Please check the template.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
        else:
            st.info("👆 Please upload a file to begin the analysis.")
    
