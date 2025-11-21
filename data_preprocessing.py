import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

def load_seer_medicare_data(file_path):
    df = pd.read_csv(file_path)
    df = df[(df['age'] >= 18) & (df['rcc_diagnosis'] == 1) & (df['follow_up_months'] >= 12)]
    df = df[~df['death_certificate_only'] & ~df['other_malignancies'] & ~df['incomplete_data']]
    return df

def handle_missing_data(df, threshold=0.15):
    missing_perc = df.isnull().mean()
    df = df.drop(columns=missing_perc[missing_perc >= threshold].index)
    imputer = IterativeImputer(max_iter=20, n_nearest_features=5, random_state=42)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

def propensity_score_matching(df_long_covid, df_control, caliper=0.2):
    covariates = ['age', 'sex', 'race', 'cancer_stage', 'grade', 'treatment', 'charlson_index']
    logit = sm.Logit(df_long_covid['long_covid'], pd.concat([df_long_covid[covariates], df_control[covariates]])).fit()
    ps_long = logit.predict(df_long_covid[covariates])
    ps_control = logit.predict(df_control[covariates])
    matched = []
    for i, ps in enumerate(ps_long):
        distances = np.abs(ps - ps_control)
        min_dist = np.min(distances)
        if min_dist <= caliper:
            idx = np.argmin(distances)
            matched.append(df_control.iloc[idx])
            ps_control = np.delete(ps_control, idx)
            df_control = df_control.drop(df_control.index[idx])
    matched_df = pd.concat(matched) if matched else pd.DataFrame()
    return matched_df

def standardize_variables(df):
    scaler = StandardScaler()
    continuous_vars = df.select_dtypes(include=['float64', 'int64']).columns.drop(['long_covid', 'progression'], errors='ignore')
    df[continuous_vars] = scaler.fit_transform(df[continuous_vars])
    return df

def prepare_time_series_data(df, window=30, step=7):
    data = []
    labels = []
    for patient_id, group in df.groupby('patient_id'):
        ts = group.sort_values('time')
        for i in range(0, len(ts) - window, step):
            window_data = ts.iloc[i:i+window].drop(columns=['progression', 'patient_id', 'time']).values
            label = ts.iloc[i+window]['progression']
            data.append(window_data)
            labels.append(label)
    return np.array(data), np.array(labels)

def split_data(data, labels, test_size=0.3):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size, random_state=42)
    return train_data, test_data, train_labels, test_labels
