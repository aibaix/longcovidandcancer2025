import os
import pandas as pd
import numpy as np
from data_preprocessing import load_seer_medicare_data, handle_missing_data, propensity_score_matching, standardize_variables, prepare_time_series_data, split_data
from disease_network import build_disease_network, bootstrap_network, compute_centralities, detect_communities, visualize_network, permutation_test_network
from knowledge_graph import fetch_literature, build_knowledge_graph, integrate_disease_network_with_kg, visualize_kg, extract_key_pathways
from lstm_model import LSTMModel, TimeSeriesDataset, DataLoader, train_lstm, evaluate_lstm, cross_validate_lstm, interpret_model, plot_roc_curve, plot_calibration_curve
from molecular_docking import download_pdb, prepare_ligand, convert_to_pdbqt, run_autodock_vina, analyze_docking_results, sensitivity_analysis, validate_against_database
from statistical_analysis import compare_groups, kaplan_meier_survival, cox_proportional_hazards, mixed_effects_model, evaluate_regression_predictions, landmark_analysis, counterfactual_analysis

def main():
    # Data Preprocessing
    df = load_seer_medicare_data('seer_medicare.csv')
    df = handle_missing_data(df)
    df_long = df[df['long_covid'] == 1]
    df_control = df[df['long_covid'] == 0]
    matched_control = propensity_score_matching(df_long, df_control)
    full_df = pd.concat([df_long, matched_control]).reset_index(drop=True)
    full_df = standardize_variables(full_df)
    ts_data, ts_labels = prepare_time_series_data(full_df)
    train_data, test_data, train_labels, test_labels = split_data(ts_data, ts_labels)
    
    # Disease Network
    categories = ['rcc', 'other_renal', 'metabolic', 'cardiovascular', 'inflammatory', 'systemic']
    G = build_disease_network(full_df, categories)
    G = bootstrap_network(G, full_df, categories)
    betweenness, eigenvector, closeness, degree = compute_centralities(G)
    communities, modularity = detect_communities(G)
    visualize_network(G, 'disease_network.png')
    p_values = permutation_test_network(G, full_df, categories)
    
    # Knowledge Graph
    literature = fetch_literature('Long COVID RCC')
    kg = build_knowledge_graph(literature)
    integrated_G = integrate_disease_network_with_kg(G, kg)
    visualize_kg(integrated_G, 'knowledge_graph.png')
    pathways = extract_key_pathways(integrated_G, 'RCC')
    
    # LSTM Model
    input_size = train_data.shape[-1]
    train_dataset = TimeSeriesDataset(train_data, train_labels)
    test_dataset = TimeSeriesDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = LSTMModel(input_size)
    model = train_lstm(model, train_loader, train_loader)  # Using train as val for simplicity
    auc, pr_auc, brier, ece = evaluate_lstm(model, test_loader)
    cv_score = cross_validate_lstm(ts_data, ts_labels, input_size)
    feature_names = full_df.columns.drop(['progression', 'patient_id', 'time'])
    importances = interpret_model(model, torch.FloatTensor(test_data), feature_names)
    plot_roc_curve(test_labels, model(torch.FloatTensor(test_data)).squeeze().numpy(), 'roc.png')
    plot_calibration_curve(test_labels, model(torch.FloatTensor(test_data)).squeeze().numpy(), 'calibration.png')
    
    # Molecular Docking
    download_pdb('1TGK', 'tgf.pdb')
    download_pdb('3GUT', 'nfkb.pdb')
    convert_to_pdbqt('tgf.pdb', 'tgf.pdbqt', 'receptor')
    convert_to_pdbqt('nfkb.pdb', 'nfkb.pdbqt', 'ligand')
    center = [0, 0, 0]  # Example center
    scores = run_autodock_vina('tgf.pdbqt', 'nfkb.pdbqt', center)
    mean, std = analyze_docking_results('docking_out.pdbqt')
    params_grid = [{'center': [0,0,0], 'size': 40, 'exhaustiveness': 8}, {'center': [0,0,0], 'size': 50, 'exhaustiveness': 16}]
    sens_results = sensitivity_analysis('tgf.pdbqt', 'nfkb.pdbqt', params_grid)
    is_valid = validate_against_database(('TGFB1', 'NFKB1'))
    
    # Statistical Analysis
    continuous_vars = ['age', 'bmi', 'egfr', 'creatinine']
    categorical_vars = ['gender', 'hypertension', 'diabetes']
    group_comps = compare_groups(full_df, 'long_covid', continuous_vars, categorical_vars)
    survival_results = kaplan_meier_survival(full_df, 'long_covid', 'follow_up_months', 'progression')
    cox_summary = cox_proportional_hazards(full_df, 'follow_up_months', 'progression', ['age', 'stage', 'il6', 'long_covid'])
    mixed_summary = mixed_effects_model(full_df, 'egfr', 'time * long_covid', '1 + time')
    preds = model(torch.FloatTensor(ts_data)).squeeze().detach().numpy()
    rmse, r2 = evaluate_regression_predictions(ts_labels, preds)
    landmarks = landmark_analysis(full_df)
    risk_red = counterfactual_analysis(full_df, 'immunotherapy', 'progression', model)  # Model as predictor
    
    # Output results
    results = {
        'network_p_values': p_values,
        'lstm_metrics': {'auc': auc, 'pr_auc': pr_auc, 'brier': brier, 'ece': ece},
        'docking_mean': mean,
        'survival_p': survival_results['p'][0]
    }
    with open('results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
