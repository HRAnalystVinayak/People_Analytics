
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from turnover import turnover_tab 
from diversity_metrics import diversity_metrics
from recruitment_metrics import recruitment_metrics
from Engagement_metrics import Engagement_metrics
from performance_metrics import performance_metrics



from EDA import EDA_Page                # make sure turnover_tab is defined in turnover.py

# App config
st.set_page_config(page_title="HR Analytics App", layout="wide")

# Horizontal navigation menu
selected = option_menu(
    menu_title=None,
    options=["Home", "EDA"],
    icons=["house", "bar-chart"],
    orientation="horizontal"
)

# Page 1: HR Analytics Dashboard
if selected == "Home":
    st.title("Application to HR Analytics")

    # Main Analytics Type Selector
    analytics_type = st.radio(
        "Choose your Analytics Type:",
        ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"],
        horizontal=True
    )

    # Tabs based on analytics type
    if analytics_type == "Descriptive":
        tabs = st.tabs([
            "Turnover Analysis", 
            "Diversity Metric", 
            "Recruitment Metrics", 
            "Engagement Results", 
            "Performance Scores"
        ])
        turnover_tab(tabs)
        diversity_metrics(tabs)
        recruitment_metrics(tabs)
        Engagement_metrics(tabs)
        performance_metrics(tabs)
        

    elif analytics_type == "Diagnostic":
        tabs = st.tabs([
            "Recruitment effectiveness", 
            "Diversity gaps", 
            "Declining employee engagement", 
            "Performance issues"
        ])
        # You can add diagnostic_tab(tabs) here if needed

    elif analytics_type == "Predictive":
        tabs = st.tabs([
            "Turnover prediction", 
            "Recruitment success forecasting", 
            "Future workforce needs", 
            "Engagement and productivity trends",
            "Performance outcomes"
        ])
        turnover_prediction(tabs)
        
        # Add logic or functions for predictive tabs

    elif analytics_type == "Prescriptive":
        tabs = st.tabs([
            "Turnover reduction strategies", 
            "Optimizing recruitment channels", 
            "Engagement improvement plans", 
            "Training and development recommendations"
        ])
        # Add logic or functions for prescriptive tabs

# Page 2: EDA
elif selected == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    EDA_Page()
    
