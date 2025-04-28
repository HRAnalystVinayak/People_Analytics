import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from datetime import datetime
import calendar

def Engagement_metrics(tabs):
    with tabs[3]:
        st.header("Please Update the data in the specified format.")
        st.info("""
        - Kindly avoid adding any extra columns to the sheet, as they will not be processed.
        - If you do not have data for a required column, you may leave it blank. 
          Empty columns will be automatically removed during data processing.
        """)

        
        st.download_button(
            label="📥 Download Employee Engagement Analysis Template",
            data="EmployeeID,Department,DOJ,Age,Attrition\n1001,Purchase/HR,2019-12-12,25,Yes",
            file_name="attrition_template.csv",
            mime="text/csv", key = "Engagement"
        )
        
        st.header(" Engagement_metrics Analysis Dashboard")
        st.subheader("1. Upload and Prepare Data")

        upload_file = st.file_uploader("⬆️ Upload Engagement_metrics Analysis CSV File", type='csv', key="Engagement_metrics")
        if upload_file is not None:
            try:
                df = pd.read_csv(upload_file, parse_dates=['DOJ', 'DOE', 'DOB'])
                
            except KeyError as e:
                st.error(f"❌ Error: Column '{e}' not found. Please check the template.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
        else:
            st.info("👆 Please upload a file to begin the analysis.")
    
