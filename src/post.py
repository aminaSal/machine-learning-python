# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:01:09 2025

@author: salla
"""
from pipeline import prepare_data
import pdb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from itertools import combinations

# Charger les données de salaires
df = pd.read_csv('Data/data.csv', sep=',')
print(df)
df = df.drop(df.columns[0], axis=1) # Supprime toutes les colonnes qui sont entièrement vides
df = df.drop(columns=['salary', 'salary_currency'])
print(df.columns)
df.dtypes

# Séparation en variables explicatives (X) et cible (y)
X = df.drop(columns=['salary_in_usd'])
y = df['salary_in_usd']

# Diviser les données en ensembles d'entraînement et de test
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X, y, test_size=0.25, random_state=42)

# Liste des colonnes disponibles dans le DataFrame
all_columns = list(X.columns)

# Liste pour stocker les résultats
results = []

# Tester toutes les combinaisons de 4 variables explicatives
combinations_4 = combinations(all_columns, 4)

for selected_columns in combinations_4:
    print(f"Test avec les variables : {selected_columns}")
    
    # Extraire les données d'entraînement et de test en utilisant ces variables explicatives
    X_train_set_selected = X_train_set[list(selected_columns)]
    X_test_set_selected = X_test_set[list(selected_columns)]

    # Préparer les données en appliquant le pipeline de préparation
    X_train_set_selected = prepare_data(X_train_set_selected, train=True, selected_columns=selected_columns)
    #pdb.set_trace()
    X_test_set_selected = prepare_data(X_test_set_selected, train=False, selected_columns=selected_columns)
    
    # Régression Linéaire
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_set_selected, y_train_set)
    y_test_set_pred_linreg = lin_reg.predict(X_test_set_selected)
    r2_test_linreg = r2_score(y_test_set, y_test_set_pred_linreg)
    rmse_test_linreg = np.sqrt(mean_squared_error(y_test_set, y_test_set_pred_linreg))
    
    # Validation croisée pour la régression linéaire
    cross_val_r2s_linreg = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(X_train_set_selected, y_train_set):
        X_train_fold, X_val_fold = X_train_set_selected.iloc[train_index], X_train_set_selected.iloc[val_index]
        y_train_fold, y_val_fold = y_train_set.iloc[train_index], y_train_set.iloc[val_index]
        
        clone_lin_reg = LinearRegression()
        clone_lin_reg.fit(X_train_fold, y_train_fold)
        y_val_fold_pred = clone_lin_reg.predict(X_val_fold)
        val_r2 = r2_score(y_val_fold, y_val_fold_pred)
        cross_val_r2s_linreg.append(val_r2)
    
    cross_val_r2_linreg_mean = np.mean(cross_val_r2s_linreg)
    
    # Random Forest
    # Avec hyperparamètres
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    # Sans hyperparamètres
    #rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train_set_selected, y_train_set)
    y_test_set_pred_rf = rf_reg.predict(X_test_set_selected)
    r2_test_rf = r2_score(y_test_set, y_test_set_pred_rf)
    rmse_test_rf = np.sqrt(mean_squared_error(y_test_set, y_test_set_pred_rf))
    
    # Validation croisée pour Random Forest
    cross_val_r2s_rf = []
    for train_index, val_index in kf.split(X_train_set_selected, y_train_set):
        X_train_fold, X_val_fold = X_train_set_selected.iloc[train_index], X_train_set_selected.iloc[val_index]
        y_train_fold, y_val_fold = y_train_set.iloc[train_index], y_train_set.iloc[val_index]
        
        #Avec hyperparamètres
        clone_rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
        #Sans hyperparamètres
        #clone_rf_reg = RandomForestRegressor(random_state=42)
        clone_rf_reg.fit(X_train_fold, y_train_fold)
        y_val_fold_pred_rf = clone_rf_reg.predict(X_val_fold)
        val_r2_rf = r2_score(y_val_fold, y_val_fold_pred_rf)
        cross_val_r2s_rf.append(val_r2_rf)
    
    cross_val_r2_rf_mean = np.mean(cross_val_r2s_rf)

    # Enregistrer les résultats
    results.append({
        'Variables': ', '.join(selected_columns),
        'R² Test Linéaire': r2_test_linreg,
        'RMSE Test Linéaire': rmse_test_linreg,
        'R² moyen (validation croisée) Linéaire': cross_val_r2_linreg_mean,
        'R² Test Random Forest': r2_test_rf,
        'RMSE Test Random Forest': rmse_test_rf,
        'R² moyen (validation croisée) Random Forest': cross_val_r2_rf_mean
    })

# Trier les résultats par R² moyen (validation croisée) pour Random Forest, puis pour Régression Linéaire
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by=['R² moyen (validation croisée) Random Forest', 'R² moyen (validation croisée) Linéaire'], 
    ascending=False
)

# Enregistrer les résultats triés dans un fichier CSV
results_df.to_csv('Data/Results.csv', index=False, sep=';')
































