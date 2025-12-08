# -*- coding: utf-8 -*- 
"""
Created on Tue Jan 14 19:00:04 2025

@author: salla
"""

import pandas as pd
import pickle
import os
import pdb

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

print("Début du script")

def prepare_data(df, train, selected_columns=None):
    print("Début de la fonction prepare_data")
    
    # 1. Validation des entrées
    if selected_columns is None or len(selected_columns) != 4:
        raise ValueError("Veuillez fournir exactement 4 variables explicatives dans 'selected_columns'.")
    
    df = df[list(selected_columns)]
    
    # 2. Vérification des colonnes présentes dans df
    num_cols = ['salary_in_usd', 'work_year', 'remote_ratio']
    
    # Vérifie si les colonnes numériques sont présentes dans le DataFrame
    num_cols_existing = [col for col in num_cols if col in df.columns]
    cat_cols = [col for col in selected_columns if col not in num_cols_existing]
    
    # Sépare les colonnes numériques et catégoriques
    df_num = df[num_cols_existing] if num_cols_existing else pd.DataFrame()
    df_cat = df[cat_cols] if cat_cols else pd.DataFrame()
    
    # 3. Gestion des valeurs manquantes
    if train:
        # Imputation pour les colonnes numériques
        num_imputer = SimpleImputer(strategy='mean')
        if not df_num.empty:
            num_imputer.fit(df_num)
        #pdb.set_trace()
        # Imputation pour les colonnes catégoriques
        cat_imputer = None
        if not df_cat.empty:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            cat_imputer.fit(df_cat)
    
        # Sauvegarde des imputeurs
        with open('Data/salary_imputer.pickle', 'wb') as f:
            pickle.dump((num_imputer, cat_imputer), f)
    else:
        # Chargement des imputeurs
        with open('Data/salary_imputer.pickle', 'rb') as f:
            num_imputer, cat_imputer = pickle.load(f)
    
    # Application de l'imputation
    if not df_num.empty:
        df_num = pd.DataFrame(num_imputer.transform(df_num), columns=num_cols_existing)
    
    if cat_imputer and not df_cat.empty:
        df_cat = pd.DataFrame(cat_imputer.transform(df_cat), columns=cat_cols)
    else:
        df_cat = pd.DataFrame()
    
    # 4. Encodage des variables catégorielles
    if train and not df_cat.empty:
        cat_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        df_cat_encoded = pd.DataFrame(cat_encoder.fit_transform(df_cat),
                                      columns=cat_encoder.get_feature_names_out(cat_cols))
        
        # Sauvegarde de l'encodeur
        with open('Data/salary_cat_encoder.pickle', 'wb') as f:
            pickle.dump(cat_encoder, f)
    elif not df_cat.empty:
        # Chargement de l'encodeur
        with open('Data/salary_cat_encoder.pickle', 'rb') as f:
            cat_encoder = pickle.load(f)
        
        df_cat_encoded = pd.DataFrame(cat_encoder.transform(df_cat),
                                      columns=cat_encoder.get_feature_names_out(cat_cols))
    else:
        df_cat_encoded = pd.DataFrame()

    # 5. Réassemblage du DataFrame
    df_final = pd.concat([df_num, df_cat_encoded], axis=1)
    
    # 6. Normalisation des données
    if train and not df_final.empty:
        scaler = MinMaxScaler()
        df_final = pd.DataFrame(scaler.fit_transform(df_final), columns=df_final.columns)
        
        # Sauvegarde du scaler
        with open('Data/salary_scaler.pickle', 'wb') as f:
            pickle.dump(scaler, f)
    elif not df_final.empty:
        # Chargement du scaler
        with open('Data/salary_scaler.pickle', 'rb') as f:
            scaler = pickle.load(f)
        df_final = pd.DataFrame(scaler.transform(df_final), columns=df_final.columns)
    
    return df_final

print("Fin du script")