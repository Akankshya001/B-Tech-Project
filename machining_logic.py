import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import joblib
import os
import warnings
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
from scipy.optimize import differential_evolution

warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────────────────────
# SECTION 1: CONSTANTS & GLOBALS
# ───────────────────────────────────────────────────────────────────────────

TARGETS = {
    'surface_roughness_Ra_um': 'Ra Model',
    'tool_wear_VB_mm':         'Tool Wear Model',
    'cutting_force_N':         'Cutting Force Model',
}

# ───────────────────────────────────────────────────────────────────────────
# SECTION 2: DYNAMIC MATERIAL DATABASE
# ───────────────────────────────────────────────────────────────────────────

class MaterialDatabase:
    MATERIAL_PROPERTIES = {
        'Inconel_718': {'hardness_HRC': 35, 'tensile_strength_MPa': 1380, 'thermal_conductivity_W_mK': 11.4, 'density_g_cm3': 8.19, 'machinability_index': 0.15, 'recommended_tool': 'Carbide/CBN', 'coolant': 'MQL/Flood', 'cutting_speed_range_mpm': (20, 80), 'feed_range_mm_rev': (0.05, 0.3), 'doc_range_mm': (0.1, 2.0)},
        'Ti6Al4V': {'hardness_HRC': 36, 'tensile_strength_MPa': 950, 'thermal_conductivity_W_mK': 6.7, 'density_g_cm3': 4.43, 'machinability_index': 0.25, 'recommended_tool': 'Uncoated Carbide', 'coolant': 'Flood/Cryogenic', 'cutting_speed_range_mpm': (30, 120), 'feed_range_mm_rev': (0.1, 0.4), 'doc_range_mm': (0.5, 3.0)},
        'AISI_4340': {'hardness_HRC': 54, 'tensile_strength_MPa': 1862, 'thermal_conductivity_W_mK': 42.7, 'density_g_cm3': 7.85, 'machinability_index': 0.57, 'recommended_tool': 'CBN', 'coolant': 'Dry/MQL', 'cutting_speed_range_mpm': (80, 250), 'feed_range_mm_rev': (0.05, 0.25), 'doc_range_mm': (0.05, 0.5)},
        'SS_316L': {'hardness_HRC': 24, 'tensile_strength_MPa': 485, 'thermal_conductivity_W_mK': 16.3, 'density_g_cm3': 7.99, 'machinability_index': 0.45, 'recommended_tool': 'Coated Carbide', 'coolant': 'Flood', 'cutting_speed_range_mpm': (60, 200), 'feed_range_mm_rev': (0.1, 0.5), 'doc_range_mm': (0.5, 4.0)},
        'CFRP': {'hardness_HRC': None, 'tensile_strength_MPa': 600, 'thermal_conductivity_W_mK': 5.0, 'density_g_cm3': 1.6, 'machinability_index': 0.20, 'recommended_tool': 'Diamond/PCD', 'coolant': 'Dry/Air Blast', 'cutting_speed_range_mpm': (100, 500), 'feed_range_mm_rev': (0.05, 0.2), 'doc_range_mm': (0.5, 3.0)}
    }

    def __init__(self, db_path='./htm_database/'):
        self.db_path = db_path
        self.data_store = {}
        self.combined_df = None
        os.makedirs(db_path, exist_ok=True)

    def add_from_csv(self, filepath, material_name, column_mapping=None):
        df = pd.read_csv(filepath)
        if column_mapping: df.rename(columns=column_mapping, inplace=True)
        df['material'] = material_name
        self.data_store[material_name] = df
        self.combined_df = pd.concat(self.data_store.values(), ignore_index=True)

    def add_single_record(self, material_name, record_dict):
        record_dict['material'] = material_name
        new_row = pd.DataFrame([record_dict])
        if material_name in self.data_store:
            self.data_store[material_name] = pd.concat([self.data_store[material_name], new_row], ignore_index=True)
        else:
            self.data_store[material_name] = new_row
        self.combined_df = pd.concat(self.data_store.values(), ignore_index=True)

    def generate_synthetic_data(self, material_name, n_samples=300, seed=42):
        rng = np.random.default_rng(seed)
        props = self.MATERIAL_PROPERTIES[material_name]
        Vc = rng.uniform(*props['cutting_speed_range_mpm'], n_samples)
        f = rng.uniform(*props['feed_range_mm_rev'], n_samples)
        doc = rng.uniform(*props['doc_range_mm'], n_samples)
        r_n = rng.choice([0.4, 0.8, 1.2], n_samples)
        tool_types = rng.choice(['Carbide', 'CBN', 'PCD'], n_samples)
        coolant_types = rng.choice(['Dry', 'MQL', 'Flood'], n_samples)
        Ra_theoretical = (f**2 / (32 * r_n)) * 1000
        Ra_final = Ra_theoretical * (1 + 0.15 * np.exp(-Vc / 50)) * (1 + 0.05 * doc)
        Ra_final += rng.normal(0, 0.05 * Ra_final)
        Fc = (props['tensile_strength_MPa'] * 0.3) * f * doc + rng.normal(0, 20, n_samples)
        df = pd.DataFrame({
            'material': material_name, 'cutting_speed_mpm': np.round(Vc, 2), 'feed_mm_rev': np.round(f, 4),
            'depth_of_cut_mm': np.round(doc, 3), 'nose_radius_mm': r_n, 'tool_type': tool_types, 'coolant_type': coolant_types,
            'surface_roughness_Ra_um': np.round(np.clip(Ra_final, 0.05, 10.0), 4),
            'tool_wear_VB_mm': np.round(np.clip(0.3 / (100 / (Vc**4) + 1e-5), 0.01, 0.8), 4),
            'cutting_force_N': np.round(np.clip(Fc, 10, 5000), 2),
        })
        self.data_store[material_name] = df
        self.combined_df = pd.concat(self.data_store.values(), ignore_index=True)
        return df

# ───────────────────────────────────────────────────────────────────────────
# SECTION 3: MODELS & UNCERTAINTY
# ───────────────────────────────────────────────────────────────────────────

def prepare_features(df, suite_feature_cols=None):
    df = df.copy()
    df['MRR_cm3_min'] = df['cutting_speed_mpm'] * df['feed_mm_rev'] * df['depth_of_cut_mm'] * 100
    df['Ra_theoretical'] = (df['feed_mm_rev']**2 / (32 * df['nose_radius_mm'])) * 1000
    df['feed_speed_ratio'] = df['feed_mm_rev'] / (df['cutting_speed_mpm'] + 1e-6)
    cat_cols = ['material', 'tool_type', 'coolant_type']
    df = pd.get_dummies(df, columns=cat_cols)
    if suite_feature_cols is not None:
        df = df.reindex(columns=suite_feature_cols, fill_value=0)
    return df

class HTMModelSuite:
    def __init__(self):
        self.models = {}; self.unc_models = {}; self.feature_cols = None

    def _tune_and_train(self, X_train, y_train, n_trials=30):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'random_state': 42
            }
            m = XGBRegressor(**params)
            return cross_val_score(m, X_train, y_train, cv=3, scoring='r2').mean()
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_m = XGBRegressor(**study.best_params, random_state=42)
        best_m.fit(X_train, y_train)
        return best_m

    def train_all(self, df_enc, n_trials=25):
        self.feature_cols = [c for c in df_enc.columns if c not in TARGETS]
        X = df_enc[self.feature_cols]
        for target in TARGETS:
            y = df_enc[target]
            self.models[target] = self._tune_and_train(X, y, n_trials)
            bag = BaggingRegressor(estimator=XGBRegressor(n_estimators=100), n_estimators=15, random_state=42)
            bag.fit(X, y)
            self.unc_models[target] = bag

    def save_models(self, path='./htm_database/models/'):
        os.makedirs(path, exist_ok=True)
        for target, model in self.models.items():
            joblib.dump(model, os.path.join(path, f"{target}.pkl"))
        joblib.dump(self.feature_cols, os.path.join(path, 'feature_cols.pkl'))

# ───────────────────────────────────────────────────────────────────────────
# SECTION 4: INFERENCE (Forward, Inverse, Pareto, Comparison)
# ───────────────────────────────────────────────────────────────────────────

def forward_predict(db, suite, material, vc, f, doc, r_n, tool, cool):
    rec = {'cutting_speed_mpm': vc, 'feed_mm_rev': f, 'depth_of_cut_mm': doc, 'nose_radius_mm': r_n, 'tool_type': tool, 'coolant_type': cool, 'material': material}
    df_in = prepare_features(pd.DataFrame([rec]), suite.feature_cols)
    res = {TARGETS[t]: float(suite.models[t].predict(df_in)[0]) for t in TARGETS}
    # Uncertainty Calculation
    p_dist = np.array([est.predict(df_in)[0] for est in suite.unc_models['surface_roughness_Ra_um'].estimators_])
    res['Ra_Lower_CI'], res['Ra_Upper_CI'] = np.quantile(p_dist, 0.05), np.quantile(p_dist, 0.95)
    res['MRR (cm3/min)'] = vc * f * doc * 100
    return res

def inverse_optimise(db, suite, material, target_Ra, tool, cool, r_n=0.8):
    props = db.MATERIAL_PROPERTIES[material]
    bounds = [props['cutting_speed_range_mpm'], props['feed_range_mm_rev'], props['doc_range_mm']]
    def obj(x): return abs(forward_predict(db, suite, material, x[0], x[1], x[2], r_n, tool, cool)['Ra Model'] - target_Ra)
    res = differential_evolution(obj, bounds, maxiter=20, popsize=10, seed=42)
    return res.x

def get_pareto_front(db, suite, material, tool, cool, r_n=0.8, n=100):
    props = db.MATERIAL_PROPERTIES[material]; rng = np.random.default_rng(42)
    v, f, d = rng.uniform(*props['cutting_speed_range_mpm'], n), rng.uniform(*props['feed_range_mm_rev'], n), rng.uniform(*props['doc_range_mm'], n)
    res = []
    for i in range(n):
        p = forward_predict(db, suite, material, v[i], f[i], d[i], r_n, tool, cool)
        res.append({'Ra': p['Ra Model'], 'MRR': p['MRR (cm3/min)'], 'Vc': v[i], 'f': f[i], 'doc': d[i]})
    df = pd.DataFrame(res); costs = np.column_stack([df['Ra'], -df['MRR']])
    is_eff = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs): is_eff[i] = not np.any(np.all(costs <= c, axis=1) & np.any(costs < c, axis=1))
    return df[is_eff].sort_values('MRR')

def recommend_strategy(db, suite, material, target_Ra, target_MRR=None):
    props = db.MATERIAL_PROPERTIES[material]; rng = np.random.default_rng(0); cands = []
    for t in ['Carbide', 'CBN', 'PCD']:
        for c in ['Dry', 'MQL', 'Flood']:
            for _ in range(20):
                v, f, d, rn = rng.uniform(*props['cutting_speed_range_mpm']), rng.uniform(*props['feed_range_mm_rev']), rng.uniform(*props['doc_range_mm']), rng.choice([0.4, 0.8, 1.2])
                p = forward_predict(db, suite, material, v, f, d, rn, t, c)
                cands.append({'tool': t, 'coolant': c, 'Vc': v, 'f': f, 'doc': d, 'rn': rn, 'Ra': p['Ra Model'], 'MRR': p['MRR (cm3/min)']})
    df = pd.DataFrame(cands); df['score'] = abs(df['Ra'] - target_Ra)
    if target_MRR: df['score'] += abs(df['MRR'] - target_MRR) * 0.1
    return df.nsmallest(5, 'score')

def compare_materials(db, suite, vc, f, doc, r_n, tool, cool):
    results = []
    for mat in db.MATERIAL_PROPERTIES.keys():
        p = forward_predict(db, suite, mat, vc, f, doc, r_n, tool, cool)
        results.append({'Material': mat, 'Ra (µm)': p['Ra Model'], 'Tool Wear (mm)': p['Tool Wear Model'], 'Force (N)': p['Cutting Force Model']})
    dk=pd.DataFrame(results)
    # Setup radar categories and angles
    cats = ['Ra (µm)', 'Tool Wear (mm)', 'Force (N)']
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Initialize polar plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    for _, row in dk.iterrows():
        # Normalise values for visual comparison
        vals = [row[c] for c in cats]
        max_vals = [dk[c].max() for c in cats]
        norm = [v / (m + 1e-9) for v, m in zip(vals, max_vals)]
        norm += norm[:1]
        ax.plot(angles, norm, linewidth=1.5, label=row['Material'])
        ax.fill(angles, norm, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.close(fig)
    return dk,fig

# ───────────────────────────────────────────────────────────────────────────
# SECTION 5: CONTINUOUS LEARNING
# ───────────────────────────────────────────────────────────────────────────

def continuous_learning_update(db, suite, new_records_list, n_trials=10):
    """Ingest new experimental data and retrain system."""
    for rec in new_records_list:
        mat = rec.get('material')
        if mat: db.add_single_record(mat, rec)
    df_enc = prepare_features(db.combined_df)
    suite.train_all(df_enc, n_trials=n_trials)
    suite.save_models()
    return "System updated with new experimental data."
