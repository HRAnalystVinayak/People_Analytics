
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from datetime import datetime
import calendar

def diversity_metrics(tabs):
    with tabs[1]:
        st.header("Please Update the data in the specified format.")
        st.info("""
        - Kindly avoid adding any extra columns to the sheet, as they will not be processed.
        - If you do not have data for a required column, you may leave it blank. 
          Empty columns will be automatically removed during data processing.
        """)

        
        st.download_button(
            label="📥 Download Diversity Analysis Template",
            data="EmployeeID,Department,Job Level,Gender,LGBTQ,Indigenous,Ethnicity,Disability,Minority,Veteran,DOB,DOJ,Country\n1001,Purchase/HR,1 to 5 ,Male/Female,Yes/No,Yes/No,Asian/Hispic,Yes/No,Yes/No,Yes/No,1991-11-27,2022-12-19,India/China",
            file_name="attrition_template.csv",
            mime="text/csv", key = "diversity"
        )

        
        st.header(" Diversity Analysis Dashboard")
        st.subheader("1. Upload and Prepare Data")

        upload_file = st.file_uploader("⬆️ Upload Diversity Analysis CSV File", type='csv', key="diversity_upload")
        if upload_file is not None:
            try:
                df = pd.read_csv(upload_file, parse_dates=['DOJ', 'DOB'])
                
                df['DOJ'] = pd.to_datetime(df['DOJ'], errors='coerce')
                df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
                
                today = datetime.now()
                df['Tenure'] = ((today - df['DOJ']).dt.days / 365.25).round(1)
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
                
                    # ---- DIVERSITY ANALYSIS SECTION ----
                    with st.expander("🌍 Workforce Diversity Dashboard", expanded=True):
                        st.title("🔍 Diversity Insights")
                        st.markdown("""
                        Understanding diversity within the workforce allows us to foster a more inclusive, equitable, and welcoming workplace.
                        This dashboard offers a breakdown of demographic attributes and their intersections.
                        """)
                
                        # Define diversity dimensions
                        diversity_cols = ["Gender", "LGBTQ", "Indigenous", "Ethnicity", "Disability", "Minority", "Veteran", "Country"]
                        valid_cols = [col for col in diversity_cols if col in df_final.columns]
                
                        if not valid_cols:
                            st.warning("⚠️ No diversity-related columns available in the dataset.")
                        else:
                            st.header("📊 Diversity by Dimension")
                            st.markdown("Each chart below shows how employees are distributed across selected diversity categories.")
                
                            # Plot diversity columns in 2-column layout
                            plots = []
                            for col in valid_cols:
                                counts = df_final[col].value_counts()
                                if not counts.empty:
                                    fig = px.bar(
                                        x=counts.index,
                                        y=counts.values,
                                        title=f"{col} Distribution",
                                        labels={'x': col, 'y': "Number of Employees"},
                                        color_discrete_sequence=px.colors.qualitative.Pastel
                                    )
                                    fig.update_traces(marker_line_width=1, marker_line_color="gray")
                                    plots.append((fig, col))
                                else:
                                    plots.append((None, col))
                
                            for i in range(0, len(plots), 2):
                                col1, col2 = st.columns(2)
                                fig1, title1 = plots[i]
                                with col1:
                                    st.markdown(f"**{title1}**")
                                    if fig1:
                                        st.plotly_chart(fig1, use_container_width=True)
                                    else:
                                        st.info(f"No data available for {title1}.")
                
                                if i + 1 < len(plots):
                                    fig2, title2 = plots[i + 1]
                                    with col2:
                                        st.markdown(f"**{title2}**")
                                        if fig2:
                                            st.plotly_chart(fig2, use_container_width=True)
                                        else:
                                            st.info(f"No data available for {title2}.")
                
                            # ---- INTERSECTIONALITY ANALYSIS ----
                            st.header("🔗 Intersectional Diversity")
                            st.markdown("Analyze how two diversity attributes intersect and influence workforce makeup.")
                
                            if len(valid_cols) >= 2:
                                c1, c2 = st.columns(2)
                                with c1:
                                    group_by = st.selectbox("Group By:", valid_cols, index=0)
                                with c2:
                                    compare_with = st.selectbox("Compare With:", [None] + [col for col in valid_cols if col != group_by])
                
                                if compare_with:
                                    ctab = pd.crosstab(df_final[group_by], df_final[compare_with])
                                    st.dataframe(ctab)
                
                                    fig_inter = px.bar(
                                        ctab.reset_index().melt(id_vars=group_by),
                                        x=group_by, y='value', color=compare_with,
                                        barmode='group',
                                        title=f"{group_by} by {compare_with}",
                                        labels={'value': 'Number of Employees'},
                                        color_discrete_sequence=px.colors.qualitative.Set2
                                    )
                                    st.plotly_chart(fig_inter, use_container_width=True)
                                else:
                                    # Single category pie chart
                                    single_counts = df_final[group_by].value_counts().reset_index()
                                    single_counts.columns = [group_by, 'Count']
                                    st.dataframe(single_counts)
                
                                    fig_pie = px.pie(single_counts, names=group_by, values='Count',
                                                     title=f"{group_by} Distribution",
                                                     color_discrete_sequence=px.colors.qualitative.Set3)
                                    st.plotly_chart(fig_pie, use_container_width=True)
                            else:
                                st.info("Add more diversity dimensions for intersectional analysis.")
                
                            # ---- SUMMARY STATISTICS ----
                            st.header("📌 Diversity Summary")
                            st.markdown("Key statistics highlighting employee distribution across diversity categories.")
                            
                            total_employees = len(df_final)
                            
                            # You can adjust number of columns in the grid layout here
                            num_columns = 2  
                            columns = st.columns(num_columns)
                            
                            for idx, col in enumerate(valid_cols):
                                with columns[idx % num_columns]:
                                    st.markdown(f"**{col}**")
                                    unique_vals = df_final[col].nunique()
                            
                                    if unique_vals == 0:
                                        st.info("No data available.")
                                    elif unique_vals == 1:
                                        st.success(f"All employees fall under '{df_final[col].iloc[0]}' category.")
                                    else:
                                        for val, count in df_final[col].value_counts().items():
                                            st.markdown(f"- {val}: {count} ({(count / total_employees * 100):.2f}%)")

                            # ---- AGE AND TENURE ----
                            if 'Age' in df_final.columns:
                                st.header("👶 Age Diversity")
                                fig_age = px.histogram(df_final, x="Age", nbins=20, title="Employee Age Distribution",
                                                       color_discrete_sequence=px.colors.sequential.Viridis)
                                st.plotly_chart(fig_age, use_container_width=True)
                
                            if 'Tenure' in df_final.columns:
                                st.header("⌛ Tenure Diversity")
                                fig_tenure = px.histogram(df_final, x="Tenure", nbins=20, title="Employee Tenure (Years)",
                                                          color_discrete_sequence=px.colors.sequential.Plasma)
                                st.plotly_chart(fig_tenure, use_container_width=True)

                
















            
            except KeyError as e:
                st.error(f"❌ Error: Column '{e}' not found. Please check the template.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
        else:
            st.info("👆 Please upload a file to begin the analysis.")
    
