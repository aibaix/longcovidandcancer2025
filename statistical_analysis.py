import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def compare_groups(df, group_col, continuous_vars, categorical_vars):
    results = {}
    group1 = df[df[group_col] == 1]
    group0 = df[df[group_col] == 0]
    for var in continuous_vars:
        if np.all(group1[var].dropna().apply(lambda x: x.is_integer())):
            stat, p = mannwhitneyu(group1[var].dropna(), group0[var].dropna())
        else:
            stat, p = ttest_ind(group1[var].dropna(), group0[var].dropna())
        results[var] = {'stat': stat, 'p': p}
    for var in categorical_vars:
        contingency = pd.crosstab(df[var], df[group_col])
        chi2, p, _, _ = chi2_contingency(contingency)
        results[var] = {'chi2': chi2, 'p': p}
    return results

def kaplan_meier_survival(df, group_col, time_col, event_col):
    kmf1 = KaplanMeierFitter()
    kmf0 = KaplanMeierFitter()
    group1 = df[df[group_col] == 1]
    group0 = df[df[group_col] == 0]
    kmf1.fit(group1[time_col], group1[event_col], label='Long COVID')
    kmf0.fit(group0[time_col], group0[event_col], label='Non-Long COVID')
    results = logrank_test(group1[time_col], group0[time_col], group1[event_col], group0[event_col])
    kmf1.plot()
    kmf0.plot()
    plt.savefig('survival_curve.png')
    plt.close()
    return results.summary

def cox_proportional_hazards(df, time_col, event_col, covariates):
    cph = CoxPHFitter()
    data = df[covariates + [time_col, event_col]]
    cph.fit(data, duration_col=time_col, event_col=event_col)
    cph.check_assumptions(data)
    return cph.summary

def mixed_effects_model(df, outcome, fixed_effects, random_effects, group='patient_id'):
    formula = f"{outcome} ~ {fixed_effects}"
    model = smf.mixedlm(formula, df, groups=df[group], re_formula=random_effects)
    result = model.fit()
    return result.summary()

def evaluate_regression_predictions(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    return rmse, r2

def landmark_analysis(df, landmarks=[90, 180, 270, 365], time_col='time', event_col='progression'):
    results = {}
    for landmark in landmarks:
        sub_df = df[df[time_col] >= landmark]
        results[landmark] = kaplan_meier_survival(sub_df, 'long_covid', time_col, event_col)
    return results

def counterfactual_analysis(df, intervention_col, outcome_col, model):
    simulated = df.copy()
    simulated[intervention_col] = 1
    preds_intervene = model.predict(simulated)
    preds_original = model.predict(df)
    risk_reduction = np.mean(preds_original - preds_intervene)
    return risk_reduction
