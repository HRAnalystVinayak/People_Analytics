import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from datetime import datetime
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

warnings.filterwarnings("ignore")

def predict_turnover(tabs):
    with tabs[0]:
        with st.expander("📋 Data Template Guide & Format", expanded=True):
            st.header("Please Update the data in the specified format.")
            st.info("""
            - Kindly avoid adding any extra columns to the sheet, as they will not be processed.
            - If you do not have data for a required column, you may leave it blank.
              Empty columns will be automatically removed during data processing.
            """)

            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                ### 📥 Required Fields
                
                **Employee Details:**
                - `EmployeeID`: Unique ID (e.g. E0001)
                - `Department`: Department name
                - `Job level`: Level 1-5
                - `Gender`: Employee gender
                - `DEIGroup`: Diversity group (Yes/No)
                - `Country`: Work location
                
                **Dates:**
                - `DOJ`: Join date (YYYY-MM-DD)
                - `DOB`: Birth date (YYYY-MM-DD)
                - `DOE`: Exit date if applicable
                
                **Performance Metrics:**
                - `Engagement Score`: 1-5 scale
                - `Performance Score`: 1-5 scale
                - `Training Hours`: Total hours
                - `Number of Projects`: Count
                """)

            with col2:
                st.markdown("""
                ### Additional Fields
                
                **Work Related:**
                - `Salary`: Annual salary
                - `Overtime`: Yes/No
                - `Work-Life Balance`: 1-5 scale
                - `Promotion in Last 3 Years`: Yes/No
                
                **Exit Information:**
                - `Reason for Leaving`: If applicable
                - `Turnover`: Target variable (Yes/No)
                
                ### Important Notes:
                - All dates must be in YYYY-MM-DD format
                - Scores should be on 1-5 scale
                - Yes/No fields must use exact spelling
                - Leave fields blank if data unavailable
                """)

                st.download_button(
                    label="📥 Download Turnover Prediction Template",
                    data="EmployeeID,Department,Job level,DOJ,DOB,DOE,Gender,DEIGroup,Country,Reason for Leaving,Salary,Engagement Score,Performance Score,Training Hours,Number of Projects,Overtime,Promotion in Last 3 Years,Work-Life Balance,Turnover\nE0001,Sales,1 to 5,2022-11-23,1991-11-27,2025-11-31,Male,Yes,Brazil,Laid off,119677.26,3,4,500,3,Yes,No,4,Yes",
                    file_name="attrition_template.csv",
                    mime="text/csv",
                    key="recruitment",
                    help="Click to download a template CSV file with the required columns and format"
                )



        st.header("Turnover Prediction Dashboard")
        with st.expander("📤 Step 1: Upload and Prepare Data", expanded=True):
            st.info("""
            Please upload a CSV file containing employee turnover data. 
            The file should include dates in YYYY-MM-DD format for:
            - Date of Joining (DOJ)
            - Date of Exit (DOE) if applicable 
            - Date of Birth (DOB)
            """)
            
            upload_file = st.file_uploader("⬆️ Upload Turnover Prediction CSV File", type='csv', key="turnover_prediction")
            
            if upload_file is not None:
                try:
                    with st.spinner("Processing data..."):
                        # Read and parse dates
                        df = pd.read_csv(upload_file, parse_dates=['DOJ', 'DOE', 'DOB'])
                        df['DOJ'] = pd.to_datetime(df['DOJ'], errors='coerce')
                        df['DOE'] = pd.to_datetime(df['DOE'], errors='coerce') 
                        df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
                        
                        # Calculate tenure and age
                        today = datetime.now()
                        df['Effective_DOE'] = df['DOE'].fillna(today)
                        df['Tenure'] = ((df['Effective_DOE'] - df['DOJ']).dt.days / 365.25).round(1)
                        df['Age'] = ((today - df['DOB']).dt.days / 365.25).round(1)
                        df = df.drop(columns=['Effective_DOE'])
                        
                        st.success("✅ Data loaded and processed successfully!")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.info("Please ensure your file has the required date columns (DOJ, DOE, DOB) in YYYY-MM-DD format")
                    st.stop()

        if 'df' in locals():
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
                    help="Choose columns to analyze for null values and cleaning"
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

                    clean_button = st.button("🚀 Run Data Cleaning", key="clean_data_button_tab2")

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
                            st.session_state['cleaned_df'] = df

                else:
                    st.warning("⚠️ Please select at least one column to proceed with cleaning.")

                if 'cleaned_df' in st.session_state:
                    df_cleaned = st.session_state['cleaned_df']

        with st.expander("🛡️ Step 3: Outlier Handling", expanded=False):
            st.info("This step helps identify and handle outliers in numerical columns using the IQR method.")

            # Check if df_cleaned exists
            if 'cleaned_df' not in st.session_state:
                st.warning("⚠️ Please complete the data cleaning step first.")
                return
            
            df_cleaned = st.session_state['cleaned_df']
            
            # Detect numerical columns
            numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()

            # Allow user to select from those columns
            outlier_cols = st.multiselect(
                "Select numerical columns for outlier analysis:",
                options=numeric_cols,
                default=numeric_cols,
                help="Choose numerical columns to check for outliers"
            )

            if outlier_cols:
                # Initial side-by-side boxplots BEFORE handling
                col_before, col_after = st.columns(2)

                with col_before:
                    st.subheader("Before Handling")
                    for col in outlier_cols:
                        fig = px.box(df_cleaned, y=col, color_discrete_sequence=["#636EFA"], title=f"📦 {col} (Before)")
                        fig.update_traces(line_color='orange', marker_color='red', selector=dict(type='box'))
                        st.plotly_chart(fig, use_container_width=True, key=f"before_plot_tab2_{col}")

                # Method selector and button
                method = st.selectbox(
                    "Select outlier handling method:", 
                    ["None", "Remove", "Impute"],
                    help="Choose how to handle detected outliers",
                    key="outlier_method_tab2"
                )
                
                if method != "None":
                    handle_button = st.button("🛠️ Handle Outliers", key="handle_outliers_button_tab2")
                    df_processed = df_cleaned.copy()

                    if handle_button:
                        with st.spinner(f"Handling outliers using {method.lower()}..."):
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
                            st.session_state['final_df'] = df_processed

                            with col_after:
                                st.subheader("After Handling")
                                for col in outlier_cols:
                                    fig = px.box(df_processed, y=col, color_discrete_sequence=["#00CC96"], title=f"📦 {col} (After)")
                                    fig.update_traces(line_color='green', marker_color='blue', selector=dict(type='box'))
                                    st.plotly_chart(fig, use_container_width=True, key=f"after_plot_tab2_{col}")

                else:
                    st.info("Select a method to handle outliers to see the 'After' box plots.")
                    st.session_state['final_df'] = df_cleaned

            else:
                st.warning("⚠️ No numerical columns selected to detect outliers.")

        st.title("📊 Employee Turnover Prediction Model Trainer")

        # Check if the final_df exists in the session state
        if 'final_df' not in st.session_state:
            st.warning("Please upload and process your data first.")
            st.stop()

        df = st.session_state['final_df'].copy()  # Create a copy to avoid modifying the original

        st.subheader("Raw Data Preview")
        st.dataframe(df.head())


        # Drop unnecessary columns - Make this interactive
        drop_columns = st.multiselect("Select columns to drop:", df.columns)
        df = df.drop(columns=drop_columns)
        st.success(f"Dropped columns: {', '.join(drop_columns)}")

        with st.expander("🛡️ Step 5: One Hot Encoding", expanded=False):
            st.subheader("One-Hot Encoding")

            encode_cols = st.multiselect("Select categorical columns for One-Hot Encoding:",df.columns)


            if encode_cols:
                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore') # Added handle_unknown
                try:
                    encoded = encoder.fit_transform(df[encode_cols])
                    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(encode_cols))
                    df = df.drop(columns=encode_cols)
                    df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
                    st.dataframe(df.head())
                    st.success(f"One-Hot Encoded columns: {', '.join(encode_cols)}")

                except Exception as e:
                    st.error(f"Error during One-Hot Encoding: {e}")

        with st.expander("🛡️ Step 6: Numerical Scaling", expanded=False):

            scale_cols = st.multiselect("Select numerical columns to scale:",df.columns)
            if scale_cols:
                scaler = MinMaxScaler()
                try:
                    df[scale_cols] = scaler.fit_transform(df[scale_cols])
                    st.success(f"Scaled columns: {', '.join(scale_cols)}")
                except Exception as e:
                    st.error(f"Error during Min-Max Scaling: {e}")

            st.subheader("Processed Data Preview")
            st.dataframe(df.head())
            st.write("Processed Data Description:", df.describe())

        # Step 7: Model Training Setup
        with st.expander("🛡️ Step 7: Model Training Setup", expanded=False):
            target_column = st.selectbox("Select the target variable (the column you want to predict):", df.columns)

            if target_column not in df.columns:
                st.error("Please select a valid target variable.")
                st.stop()

            try:
                X = df.drop(columns=[target_column])
                y = df[target_column].map({'No': 0, 'Yes': 1}) if df[target_column].dtype == 'object' else df[target_column]
            except KeyError:
                st.error(f"Target column '{target_column}' not found in the DataFrame.")
                st.stop()

            test_size = st.slider("Test set size (%):", 10, 50, 20) / 100
            random_state = 42
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            st.info(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples (test size = {test_size*100}%)")





        # Step 8: Imbalance Detection
        with st.expander("🛡️ Step 8: Handle Imbalanced Target Variable", expanded=False):
            y_ratio = y_train.value_counts(normalize=True) * 100
            y_classes = y_ratio.index.tolist()

            if len(y_ratio) < 2:
                st.warning("Only one class found in the target variable. Please check your data.")
            else:
                majority_class_ratio = round(y_ratio.iloc[0], 2)
                minority_class_ratio = round(y_ratio.iloc[1], 2)

                if majority_class_ratio >= 99:
                    scenario = "Extremely imbalanced"
                    action = "Use cost-sensitive learning or anomaly detection techniques."
                elif majority_class_ratio >= 95:
                    scenario = "Highly imbalanced"
                    action = "Must use SMOTE, undersampling, or cost-sensitive learning."
                elif majority_class_ratio >= 80:
                    scenario = "Getting imbalanced"
                    action = "Use resampling, class weights, or ensemble methods."
                elif majority_class_ratio >= 60:
                    scenario = "Still manageable"
                    action = "Try training as-is, maybe adjust metrics (Precision, Recall, F1-score)."
                else:
                    scenario = "Ideal balance"
                    action = "Train as usual, no special tricks needed."

                class_summary = pd.DataFrame({
                    "Class Ratio": [f"{majority_class_ratio}% : {minority_class_ratio}%"],
                    "Scenario": [scenario],
                    "What to Do": [action]
                })

                st.dataframe(class_summary)


                balance_button = st.button("⚙️ Run Balancing Strategy", key="balance_data_button_tab2")

                if balance_button:
                    with st.spinner("Applying Balancing Strategy..."):

                        imbalance_ratio = y_train.value_counts(normalize=True) * 100
                        majority = imbalance_ratio.iloc[0]

                        if majority >= 95:
                            strategy = "SMOTE + Undersampling (SMOTEENN)"
                            sampler = SMOTEENN(random_state=42)
                        elif majority >= 80:
                            strategy = "SMOTE Oversampling"
                            sampler = SMOTE(random_state=42)
                        elif majority >= 60:
                            strategy = "Random Undersampling"
                            sampler = RandomUnderSampler(random_state=42)
                        else:
                            strategy = None

                        if strategy:
                            X_train_bal, y_train_bal = sampler.fit_resample(X_train, y_train)
                            st.success(f"✅ Applied strategy: {strategy}")
                            st.write(f"Training shape before: {X_train.shape}, after: {X_train_bal.shape}")
                            st.session_state['X_train'] = X_train_bal
                            st.session_state['y_train'] = y_train_bal
                        else:
                            st.info("✅ Dataset is already balanced. No action taken.")
                            st.session_state['X_train'] = X_train
                            st.session_state['y_train'] = y_train

                        st.session_state['X_test'] = X_test
                        st.session_state['y_test'] = y_test




        # Available Models
        with st.expander("🛡️ Step 9: Train and Compare Models", expanded=False):

            model_dict = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
                "Decision Tree": DecisionTreeClassifier(random_state=random_state),
                "Random Forest": RandomForestClassifier(random_state=random_state),
                "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state),
                "LightGBM": LGBMClassifier(random_state=random_state)
                                    }
            selected_models = st.multiselect("Select models to train & compare:", list(model_dict.keys()))

            results = []
            if st.button("🚀 Train Selected Models"):
                if not selected_models:
                    st.warning("Please select at least one model to train.")
                else:
                    with st.spinner("Training models..."):
                        results = []
                        for model_name in selected_models:
                            model = model_dict[model_name]
                            try:
                                model.fit(st.session_state['X_train'], st.session_state['y_train'])
                                y_pred = model.predict(X_test)
                                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

                                results.append({
                                    "Model": model_name,
                                    "Accuracy": accuracy_score(y_test, y_pred),
                                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                                    "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                                    "ROC AUC": roc_auc_score(y_test, y_pred_proba)
                                })
                                st.success(f"Trained {model_name} successfully!")
                            except Exception as e:
                                st.error(f"Error training {model_name}: {e}")

                        if results:
                            result_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

                            st.subheader("📊 Model Performance Comparison:")
                            st.dataframe(result_df.style.format({
                                "Accuracy": "{:.2f}", "Precision": "{:.2f}", "Recall": "{:.2f}",
                                "F1-Score": "{:.2f}", "ROC AUC": "{:.2f}"
                            }))

                            # Selecting best model based on combined metric
                            result_df['Custom Score'] = result_df['F1-Score'] + result_df['Recall'] + result_df['Precision']
                            best_row = result_df.loc[result_df['Custom Score'].idxmax()]
                            best_model_name = best_row['Model']

                            st.info(f"""
                            ### ✅ Model Evaluation Summary

                            Based on combined performance (Precision + Recall + F1-Score), **{best_model_name}** is selected for hyperparameter tuning.

                            | Metric | Value |
                            |--------|-------|
                            | Accuracy | {best_row['Accuracy']:.2f} |
                            | Precision | {best_row['Precision']:.2f} |
                            | Recall | {best_row['Recall']:.2f} |
                            | F1-Score | {best_row['F1-Score']:.2f} |
                            | ROC AUC | {best_row['ROC AUC']:.2f} |

                            ---
                            🎯 Proceeding to Optuna tuning...
                            """)

                            st.session_state.best_model_name = best_model_name
                            st.session_state.X_train_optuna = st.session_state['X_train']
                            st.session_state.y_train_optuna = st.session_state['y_train']

                            # Objective function for tuning
                            def objective(trial):
                                if st.session_state.best_model_name == "Logistic Regression":
                                    # Define all possible parameter combinations
                                    param_combinations = [
                                        {
                                            "solver": "liblinear",
                                            "penalty": "l1",
                                            "C": trial.suggest_float("C_liblinear_l1", 0.01, 10),
                                            "max_iter": trial.suggest_int("max_iter_liblinear_l1", 100, 1000)
                                        },
                                        {
                                            "solver": "liblinear",
                                            "penalty": "l2",
                                            "C": trial.suggest_float("C_liblinear_l2", 0.01, 10),
                                            "max_iter": trial.suggest_int("max_iter_liblinear_l2", 100, 1000)
                                        },
                                        {
                                            "solver": "saga",
                                            "penalty": "l1",
                                            "C": trial.suggest_float("C_saga_l1", 0.01, 10),
                                            "max_iter": trial.suggest_int("max_iter_saga_l1", 100, 1000)
                                        },
                                        {
                                            "solver": "saga",
                                            "penalty": "l2",
                                            "C": trial.suggest_float("C_saga_l2", 0.01, 10),
                                            "max_iter": trial.suggest_int("max_iter_saga_l2", 100, 1000)
                                        },
                                        {
                                            "solver": "saga",
                                            "penalty": "elasticnet",
                                            "C": trial.suggest_float("C_saga_elasticnet", 0.01, 10),
                                            "max_iter": trial.suggest_int("max_iter_saga_elasticnet", 100, 1000),
                                            "l1_ratio": trial.suggest_float("l1_ratio_saga_elasticnet", 0.0, 1.0)
                                        }
                                    ]

                                    # Select one combination
                                    combination_idx = trial.suggest_int("combination", 0, len(param_combinations) - 1)
                                    params = param_combinations[combination_idx]

                                    try:
                                        model = LogisticRegression(**params)
                                        return cross_val_score(model, st.session_state.X_train_optuna, st.session_state.y_train_optuna, scoring="f1", cv=3).mean()
                                    except Exception as e:
                                        return float("inf")  # Return infinity for invalid parameter combinations

                                elif st.session_state.best_model_name == "Decision Tree":
                                    params = {
                                        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
                                        "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
                                        "max_depth": trial.suggest_int("max_depth", 2, 20),
                                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                                        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
                                        "random_state": 42
                                    }
                                    model = DecisionTreeClassifier(**params)

                                elif st.session_state.best_model_name == "xgboost":
                                    params = {
                                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                                        "max_depth": trial.suggest_int("max_depth", 2, 12),
                                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
                                    }
                                    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')

                                elif st.session_state.best_model_name == "LightGBM":
                                    params = {
                                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                                        "max_depth": trial.suggest_int("max_depth", 2, 12),
                                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                                        "num_leaves": trial.suggest_int("num_leaves", 20, 100)
                                    }
                                    model = LGBMClassifier(**params)

                                else:
                                    raise ValueError("Model not supported for tuning")

                                # Return mean F1 score from cross-validation
                                return cross_val_score(model, st.session_state.X_train_optuna, st.session_state.y_train_optuna, scoring="f1", cv=3).mean()

                            if st.session_state.best_model_name:
                                with st.spinner(f"🔍 Tuning {st.session_state.best_model_name} using Optuna..."):
                                    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
                                    study.optimize(objective, n_trials=30)

                                st.success(f"🎯 Optuna tuning complete for {st.session_state.best_model_name}")
                                st.write("📈 Best F1-Score:", study.best_value)
                                st.write("🔧 Best Hyperparameters:")
                                st.json(study.best_params)
                                # After selecting and tuning the best model
                                st.session_state.best_model_name_tuned = st.session_state.best_model_name
                                st.session_state.best_params = study.best_params
                                st.session_state.tuned_study = study


        # ------------------- Step 11: Train Best Model & Make Predictions ---------------------
        with st.expander("🚀 Step 11: Train Best Model & Make Predictions", expanded=False):
            if "current_step" not in st.session_state:
                st.session_state["current_step"] = 1
                st.session_state["prediction_data"] = {} # Initialize a dictionary to store prediction data
                st.session_state["predicted_turnover"] = None

            # Function to advance to the next step
            def next_step():
                st.session_state["current_step"] += 1

            # Function to go back to the previous step
            def prev_step():
                st.session_state["current_step"] -= 1

            if st.session_state.get("best_model_name_tuned"):
                st.subheader(f"Step {st.session_state['current_step']}: Input Data for Prediction")
                
                # Get the original dataframe before any transformations
                original_df = st.session_state['cleaned_df'].copy()
                
                # Create tabs for different input methods
                tab1, tab2 = st.tabs(["📥 Upload CSV File", "✍️ Manual Input"])
                
                with tab1:
                    st.write("### Upload CSV File for Batch Prediction")
                    st.info("Upload a CSV file containing employee data to predict turnover for multiple employees at once.")
                    
                    # Add template download button
                    template_df = original_df.drop(columns=['Turnover']).head()
                    csv_template = template_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Input Template",
                        data=csv_template,
                        file_name='turnover_prediction_template.csv',
                        mime='text/csv',
                        help="Download a template CSV file with the required columns"
                    )
                    
                    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
                    
                    if uploaded_file is not None:
                        try:
                            batch_df = pd.read_csv(uploaded_file)
                            st.write("Uploaded Data Preview:")
                            st.dataframe(batch_df.head())
                            
                            if st.button("Predict for Uploaded Data"):
                                with st.spinner("Making predictions..."):
                                    # Create the model instance
                                    best_model_name = st.session_state["best_model_name_tuned"]
                                    best_params = st.session_state.get("best_params", {})

                                    # Process best_params for LogisticRegression
                                    if best_model_name == "Logistic Regression":
                                        processed_params = {
                                            "solver": best_params.get("solver", "liblinear"),
                                            "penalty": best_params.get("penalty", "l2"),
                                            "C": best_params.get("C", 1.0),
                                            "max_iter": best_params.get("max_iter", 100)
                                        }
                                        if "l1_ratio" in best_params:
                                            processed_params["l1_ratio"] = best_params["l1_ratio"]
                                        final_model = LogisticRegression(**processed_params)
                                    elif best_model_name == "Decision Tree":
                                        final_model = DecisionTreeClassifier(**best_params, random_state=42)
                                    elif best_model_name == "xgboost":
                                        final_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
                                    elif best_model_name == "LightGBM":
                                        final_model = LGBMClassifier(**best_params, random_state=42)
                                    else:
                                        st.error("Best model not found.")
                                        st.stop()

                                    # Fit the model
                                    final_model.fit(st.session_state['X_train'], st.session_state['y_train'])

                                    # Process the batch data
                                    processed_batch = batch_df.copy()
                                    
                                    # Convert date columns with flexible parsing
                                    date_columns = [col for col in processed_batch.columns 
                                                  if pd.api.types.is_datetime64_any_dtype(original_df[col])]
                                    for col in date_columns:
                                        try:
                                            # Try parsing with different formats
                                            processed_batch[col] = pd.to_datetime(
                                                processed_batch[col],
                                                format='mixed',
                                                dayfirst=True
                                            )
                                            # Convert to YYYY-MM-DD format
                                            processed_batch[col] = processed_batch[col].dt.strftime('%Y-%m-%d')
                                        except Exception as e:
                                            st.error(f"Error parsing date column '{col}': {str(e)}")
                                            st.info(f"Please ensure dates in column '{col}' are in a valid format (e.g., DD-MM-YYYY, YYYY-MM-DD)")
                                            st.stop()
                                    
                                    # Apply transformations
                                    categorical_cols = [col for col in processed_batch.columns 
                                                      if processed_batch[col].dtype == 'object' 
                                                      and col in st.session_state['final_df'].columns
                                                      and col not in date_columns]
                                    
                                    if categorical_cols:
                                        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                                        encoder.fit(st.session_state['final_df'][categorical_cols])
                                        for col in categorical_cols:
                                            if processed_batch[col].isna().any():
                                                most_frequent = st.session_state['final_df'][col].mode()[0]
                                                processed_batch[col] = processed_batch[col].fillna(most_frequent)
                                        encoded_batch = encoder.transform(processed_batch[categorical_cols])
                                        encoded_df_batch = pd.DataFrame(encoded_batch, columns=encoder.get_feature_names_out(categorical_cols))
                                        processed_batch = processed_batch.drop(columns=categorical_cols)
                                        processed_batch = pd.concat([processed_batch.reset_index(drop=True), encoded_df_batch.reset_index(drop=True)], axis=1)
                                    
                                    # Scale numerical columns
                                    numerical_cols_original = st.session_state['final_df'].select_dtypes(include=np.number).columns.tolist()
                                    numerical_cols_batch = [col for col in processed_batch.columns if col in numerical_cols_original]
                                    if numerical_cols_batch:
                                        scaler = MinMaxScaler()
                                        scaler.fit(st.session_state['final_df'][numerical_cols_batch])
                                        processed_batch[numerical_cols_batch] = scaler.transform(processed_batch[numerical_cols_batch])
                                    
                                    # Ensure column order matches training data
                                    missing_cols = set(st.session_state['X_train'].columns) - set(processed_batch.columns)
                                    for c in missing_cols:
                                        processed_batch[c] = 0
                                    processed_batch = processed_batch[st.session_state['X_train'].columns]
                                    
                                    # Make predictions
                                    predictions = final_model.predict(processed_batch)
                                    probabilities = final_model.predict_proba(processed_batch)[:, 1]
                                    
                                    # Create results dataframe
                                    results_df = batch_df.copy()
                                    results_df['Predicted Turnover'] = ['Yes' if p == 1 else 'No' for p in predictions]
                                    results_df['Probability'] = probabilities
                                    
                                    st.success("✅ Predictions completed!")
                                    st.write("### Prediction Results")
                                    st.dataframe(results_df)
                                    
                                    # Add download button for results
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Predictions",
                                        data=csv,
                                        file_name='turnover_predictions.csv',
                                        mime='text/csv',
                                    )
                        except Exception as e:
                            st.error(f"Error processing CSV file: {str(e)}")
                
                with tab2:
                    st.write("### Manual Input for Single Prediction")
                    st.info("Enter data for a single employee to predict turnover.")
                    
                    input_data = {}
                    for col in original_df.columns:
                        if col == 'Turnover':  # Skip the target variable
                            continue
                            
                        if original_df[col].dtype == 'object':
                            unique_vals = original_df[col].unique().tolist()
                            input_data[col] = st.selectbox(f"Select {col}:", unique_vals, key=f"predict_{col}_{st.session_state['current_step']}")
                        elif pd.api.types.is_datetime64_any_dtype(original_df[col]):
                            input_data[col] = st.date_input(f"Enter {col}:", key=f"predict_{col}_{st.session_state['current_step']}")
                        else:
                            input_data[col] = st.number_input(f"Enter {col}:", key=f"predict_{col}_{st.session_state['current_step']}")

                    st.session_state["prediction_data"] = input_data

                    if st.button("Predict Turnover"):
                        with st.spinner("Making prediction..."):
                            best_model_name = st.session_state["best_model_name_tuned"]
                            best_params = st.session_state.get("best_params", {})

                            # Process best_params for LogisticRegression
                            if best_model_name == "Logistic Regression":
                                # Extract the actual parameter values from the best_params
                                processed_params = {
                                    "solver": best_params.get("solver", "liblinear"),
                                    "penalty": best_params.get("penalty", "l2"),
                                    "C": best_params.get("C", 1.0),
                                    "max_iter": best_params.get("max_iter", 100)
                                }
                                if "l1_ratio" in best_params:
                                    processed_params["l1_ratio"] = best_params["l1_ratio"]
                                final_model = LogisticRegression(**processed_params)
                            elif best_model_name == "Decision Tree":
                                final_model = DecisionTreeClassifier(**best_params, random_state=42)
                            elif best_model_name == "xgboost":
                                final_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
                            elif best_model_name == "LightGBM":
                                final_model = LGBMClassifier(**best_params, random_state=42)
                            else:
                                st.error("Best model not found.")
                                st.stop()

                            final_model.fit(st.session_state['X_train'], st.session_state['y_train'])

                            # Prepare input data for prediction
                            predict_df = pd.DataFrame([st.session_state["prediction_data"]])

                            # Convert date inputs to datetime
                            date_columns = [col for col in predict_df.columns if pd.api.types.is_datetime64_any_dtype(original_df[col])]
                            for col in date_columns:
                                predict_df[col] = pd.to_datetime(predict_df[col])

                            # Apply the same transformations as training data
                            # 1. One-Hot Encoding for categorical columns
                            categorical_cols = [col for col in predict_df.columns 
                                              if predict_df[col].dtype == 'object' 
                                              and col in st.session_state['final_df'].columns
                                              and col not in date_columns]  # Exclude date columns
                            
                            if categorical_cols:
                                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                                encoder.fit(st.session_state['final_df'][categorical_cols]) # Fit on the original data
                                
                                # Handle any missing or invalid values in categorical columns
                                for col in categorical_cols:
                                    if predict_df[col].isna().any():
                                        # Replace NaN with the most frequent value from training
                                        most_frequent = st.session_state['final_df'][col].mode()[0]
                                        predict_df[col] = predict_df[col].fillna(most_frequent)
                                
                                encoded_predict = encoder.transform(predict_df[categorical_cols])
                                encoded_df_predict = pd.DataFrame(encoded_predict, columns=encoder.get_feature_names_out(categorical_cols))
                                predict_df = predict_df.drop(columns=categorical_cols)
                                predict_df = pd.concat([predict_df.reset_index(drop=True), encoded_df_predict.reset_index(drop=True)], axis=1)

                            # 2. Min-Max Scaling for numerical columns
                            numerical_cols_original = st.session_state['final_df'].select_dtypes(include=np.number).columns.tolist()
                            numerical_cols_predict = [col for col in predict_df.columns if col in numerical_cols_original]
                            if numerical_cols_predict:
                                scaler = MinMaxScaler()
                                scaler.fit(st.session_state['final_df'][numerical_cols_predict]) # Fit on original data
                                predict_df[numerical_cols_predict] = scaler.transform(predict_df[numerical_cols_predict])

                            # Ensure the order of columns matches the training data
                            missing_cols = set(st.session_state['X_train'].columns) - set(predict_df.columns)
                            for c in missing_cols:
                                predict_df[c] = 0 # Fill missing columns with 0 (assuming they were not present for this instance)
                            predict_df = predict_df[st.session_state['X_train'].columns]

                            prediction = final_model.predict(predict_df)
                            probability = final_model.predict_proba(predict_df)[:, 1]

                            st.session_state["predicted_turnover"] = "Yes" if prediction[0] == 1 else "No"
                            st.session_state["prediction_probability"] = f"{probability[0]:.2f}"

                    if st.session_state.get("predicted_turnover") is not None:
                        st.subheader("Prediction Result:")
                        if st.session_state["predicted_turnover"] == "Yes":
                            st.error(f"⚠️ Predicted Turnover: **{st.session_state['predicted_turnover']}** (Probability: {st.session_state['prediction_probability']})")
                        else:
                            st.success(f"✅ Predicted Turnover: **{st.session_state['predicted_turnover']}** (Probability: {st.session_state['prediction_probability']})")

                    else:
                        st.info("Please train and tune the models first to make predictions.")
                                        
