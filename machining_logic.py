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

# ======================================================================
# FINAL ADD-ON CODE: OOD + COST ESTIMATOR + EDM MODEL ROUTE
# Paste this below your existing HTM code.
# Your existing code already contains:
# MaterialDatabase, prepare_features, HTMModelSuite, forward_predict, etc.
# ======================================================================

from dataclasses import dataclass
from typing import Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ======================================================================
# SECTION 6: OOD DETECTOR
# ======================================================================

SAFE = "SAFE"
CAUTION = "CAUTION"
OOD = "OOD"


@dataclass
class OODResult:
    risk_level: str
    is_ood: bool
    isolation_score: float
    bound_violations: list
    reason: str

    def to_dict(self):
        return {
            "risk_level": self.risk_level,
            "is_ood": self.is_ood,
            "isolation_score": round(self.isolation_score, 4),
            "bound_violations": self.bound_violations,
            "reason": self.reason
        }


class SimpleOODDetector:
    """
    OOD detector for both conventional machining and EDM.
    Uses:
    1. Hard numeric range check
    2. Isolation Forest anomaly detection
    """

    def __init__(self, numeric_cols, contamination=0.05):
        self.numeric_cols = numeric_cols
        self.contamination = contamination
        self.bounds = {}
        self.scaler = StandardScaler()
        self.iso = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42
        )
        self.is_fitted = False

    def fit(self, df):
        clean = df[self.numeric_cols].copy()
        clean = clean.replace([np.inf, -np.inf], np.nan)

        for col in self.numeric_cols:
            clean[col] = clean[col].fillna(clean[col].median())

        for col in self.numeric_cols:
            self.bounds[col] = (
                float(clean[col].min()),
                float(clean[col].max())
            )

        X_scaled = self.scaler.fit_transform(clean)
        self.iso.fit(X_scaled)
        self.is_fitted = True
        return self

    def check(self, input_dict):
        if not self.is_fitted:
            return OODResult(
                risk_level="UNKNOWN",
                is_ood=False,
                isolation_score=0.0,
                bound_violations=[],
                reason="OOD detector has not been fitted."
            )

        violations = []

        for col, (lo, hi) in self.bounds.items():
            if col in input_dict:
                val = input_dict[col]
                span = hi - lo
                allowed_lo = lo - 0.20 * span
                allowed_hi = hi + 0.20 * span

                if val < allowed_lo or val > allowed_hi:
                    violations.append(
                        f"{col}={val} outside training range [{lo:.4f}, {hi:.4f}]"
                    )

        x = pd.DataFrame([{c: input_dict.get(c, np.nan) for c in self.numeric_cols}])

        for col in self.numeric_cols:
            if pd.isna(x.loc[0, col]):
                lo, hi = self.bounds[col]
                x.loc[0, col] = (lo + hi) / 2

        x_scaled = self.scaler.transform(x)
        iso_pred = self.iso.predict(x_scaled)[0]
        raw_score = float(self.iso.score_samples(x_scaled)[0])
        isolation_score = float(np.clip(-raw_score / 0.5, 0, 1))

        if violations:
            return OODResult(
                risk_level=OOD,
                is_ood=True,
                isolation_score=isolation_score,
                bound_violations=violations,
                reason="; ".join(violations)
            )

        if iso_pred == -1:
            return OODResult(
                risk_level=CAUTION,
                is_ood=False,
                isolation_score=isolation_score,
                bound_violations=[],
                reason=f"Input is unusual compared with training data. Isolation score={isolation_score:.3f}"
            )

        return OODResult(
            risk_level=SAFE,
            is_ood=False,
            isolation_score=isolation_score,
            bound_violations=[],
            reason="Input lies within expected training distribution."
        )


# ======================================================================
# SECTION 7: COST ESTIMATOR
# ======================================================================

def ra_to_iso_n(ra_um):
    thresholds = [
        (0.025, "N1"), (0.05, "N2"), (0.1, "N3"), (0.2, "N4"),
        (0.4, "N5"), (0.8, "N6"), (1.6, "N7"), (3.2, "N8"),
        (6.3, "N9"), (12.5, "N10"), (25.0, "N11")
    ]

    for limit, grade in thresholds:
        if ra_um <= limit:
            return grade

    return "N12"


def estimate_conventional_cost(
    predicted_ra_um,
    predicted_tool_wear_mm,
    predicted_mrr_cm3_min,
    tool_type="Carbide",
    coolant_type="MQL",
    part_volume_cm3=100.0,
    machine_rate_usd_hr=75.0,
    labour_rate_usd_hr=35.0,
    overhead_rate_usd_hr=20.0,
    setup_time_min=30.0,
    batch_size=1,
    tool_life_criterion_mm=0.30,
    scrap_rate_pct=2.0
):
    tool_cost_defaults = {
        "Carbide": 8.0,
        "CBN": 45.0,
        "PCD": 60.0,
        "Diamond": 80.0,
        "Coated Carbide": 12.0,
        "Uncoated Carbide": 6.0
    }

    coolant_cost_defaults = {
        "Dry": 0.00,
        "MQL": 0.05,
        "Flood": 0.15
    }

    mrr = max(float(predicted_mrr_cm3_min), 0.001)

    machining_time_min = part_volume_cm3 / mrr

    wear_rate = max(float(predicted_tool_wear_mm), 1e-6)
    parts_per_edge = tool_life_criterion_mm / wear_rate
    tool_changes_per_part = 1.0 / max(parts_per_edge, 0.001)

    cost_per_edge = tool_cost_defaults.get(tool_type, 10.0)
    tool_cost = tool_changes_per_part * cost_per_edge

    total_rate = machine_rate_usd_hr + labour_rate_usd_hr + overhead_rate_usd_hr
    machining_cost = (machining_time_min / 60.0) * total_rate

    coolant_unit = coolant_cost_defaults.get(coolant_type, 0.10)

    if coolant_type == "Flood":
        coolant_litres = machining_time_min * 5.0
    elif coolant_type == "MQL":
        coolant_litres = machining_time_min * 0.05
    else:
        coolant_litres = 0.0

    coolant_cost = coolant_litres * coolant_unit

    setup_cost_total = (setup_time_min / 60.0) * (
        labour_rate_usd_hr + machine_rate_usd_hr
    )
    setup_cost_per_part = setup_cost_total / max(batch_size, 1)

    subtotal = machining_cost + tool_cost + coolant_cost + setup_cost_per_part
    scrap_allowance = subtotal * (scrap_rate_pct / 100.0)
    total_per_part = subtotal + scrap_allowance

    return {
        "machining_time_min": round(machining_time_min, 3),
        "cycle_time_min": round(machining_time_min + setup_time_min / max(batch_size, 1), 3),
        "MRR_cm3_min": round(mrr, 4),
        "surface_quality_class": ra_to_iso_n(predicted_ra_um),
        "tool_changes_per_part": round(tool_changes_per_part, 4),
        "tool_cost_usd": round(tool_cost, 4),
        "machining_cost_usd": round(machining_cost, 4),
        "coolant_cost_usd": round(coolant_cost, 4),
        "setup_cost_usd": round(setup_cost_per_part, 4),
        "scrap_allowance_usd": round(scrap_allowance, 4),
        "total_cost_per_part_usd": round(total_per_part, 4),
        "total_cost_batch_usd": round(total_per_part * batch_size, 4),
        "cost_per_cm3_usd": round(total_per_part / max(part_volume_cm3, 1e-6), 6)
    }


def estimate_edm_cost(
    predicted_ra_um,
    predicted_mrr_g_min=None,
    predicted_ewr_g_min=None,
    cavity_volume_cm3=5.0,
    material_density_g_cm3=7.8,
    machine_rate_usd_hr=80.0,
    labour_rate_usd_hr=35.0,
    overhead_rate_usd_hr=25.0,
    electrode_cost_usd=20.0,
    dielectric_cost_usd_hr=3.0,
    setup_time_min=40.0,
    batch_size=1,
    scrap_rate_pct=3.0
):
    if predicted_mrr_g_min is None or predicted_mrr_g_min <= 0:
        predicted_mrr_g_min = 0.05

    mrr_cm3_min = predicted_mrr_g_min / max(material_density_g_cm3, 1e-6)
    machining_time_min = cavity_volume_cm3 / max(mrr_cm3_min, 1e-6)

    total_rate = machine_rate_usd_hr + labour_rate_usd_hr + overhead_rate_usd_hr
    machining_cost = (machining_time_min / 60.0) * total_rate

    dielectric_cost = (machining_time_min / 60.0) * dielectric_cost_usd_hr

    if predicted_ewr_g_min is None:
        electrode_wear_cost = 0.15 * electrode_cost_usd
    else:
        wear_ratio = predicted_ewr_g_min / max(predicted_mrr_g_min, 1e-6)
        electrode_wear_cost = electrode_cost_usd * min(max(wear_ratio, 0), 1)

    setup_cost_total = (setup_time_min / 60.0) * (
        machine_rate_usd_hr + labour_rate_usd_hr
    )
    setup_cost_per_part = setup_cost_total / max(batch_size, 1)

    subtotal = machining_cost + dielectric_cost + electrode_wear_cost + setup_cost_per_part
    scrap_allowance = subtotal * (scrap_rate_pct / 100.0)
    total_per_part = subtotal + scrap_allowance

    return {
        "edm_machining_time_min": round(machining_time_min, 3),
        "cycle_time_min": round(machining_time_min + setup_time_min / max(batch_size, 1), 3),
        "MRR_g_min": round(predicted_mrr_g_min, 6),
        "MRR_cm3_min_estimated": round(mrr_cm3_min, 6),
        "surface_quality_class": ra_to_iso_n(predicted_ra_um),
        "machining_cost_usd": round(machining_cost, 4),
        "dielectric_cost_usd": round(dielectric_cost, 4),
        "electrode_wear_cost_usd": round(electrode_wear_cost, 4),
        "setup_cost_usd": round(setup_cost_per_part, 4),
        "scrap_allowance_usd": round(scrap_allowance, 4),
        "total_cost_per_part_usd": round(total_per_part, 4),
        "total_cost_batch_usd": round(total_per_part * batch_size, 4),
        "cost_per_cm3_usd": round(total_per_part / max(cavity_volume_cm3, 1e-6), 6)
    }


# ======================================================================
# SECTION 8: EDM DATABASE + EDM MODEL
# ======================================================================

EDM_TARGETS = [
    "surface_roughness_um",
    "out_of_roundness_um",
    "material_removal_rate_g_min",
    "electrode_wear_rate_g_min"
]

EDM_NUMERIC_FEATURES = [
    "peak_current_A",
    "gap_voltage_V",
    "pulse_on_us",
    "pulse_off_us",
    "flushing_pressure_kgf_cm2",
    "machining_time_min"
]

EDM_CATEGORICAL_FEATURES = [
    "workpiece_material",
    "electrode_material",
    "dielectric_fluid"
]


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class EDMModelSuite:
    """
    Separate EDM model suite.
    EDM is kept separate from conventional machining because EDM is thermal erosion,
    not chip-removal machining.
    """

    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.feature_cols = EDM_NUMERIC_FEATURES + EDM_CATEGORICAL_FEATURES
        self.ood_detector = None
        self.is_trained = False

    def train(self, edm_df):
        available_features = [
            c for c in self.feature_cols
            if c in edm_df.columns
        ]

        for target in EDM_TARGETS:
            if target not in edm_df.columns:
                continue

            sub = edm_df.dropna(subset=[target]).copy()

            if len(sub) < 8:
                print(f"Skipping {target}: not enough data.")
                continue

            X = sub[available_features].copy()
            y = sub[target].astype(float)

            num_cols = [
                c for c in EDM_NUMERIC_FEATURES
                if c in X.columns
            ]

            cat_cols = [
                c for c in EDM_CATEGORICAL_FEATURES
                if c in X.columns
            ]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_cols),
                    ("cat", make_ohe(), cat_cols)
                ],
                remainder="drop"
            )

            model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", RandomForestRegressor(
                        n_estimators=300,
                        random_state=42,
                        min_samples_leaf=2
                    ))
                ]
            )

            if len(sub) >= 20:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                self.metrics[target] = {
                    "MAE": float(mean_absolute_error(y_test, pred)),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "R2": float(r2_score(y_test, pred))
                }

            else:
                model.fit(X, y)

                self.metrics[target] = {
                    "MAE": None,
                    "RMSE": None,
                    "R2": None
                }

            self.models[target] = model
            print(f"EDM model trained for: {target}")

        numeric_cols = [
            c for c in EDM_NUMERIC_FEATURES
            if c in edm_df.columns
        ]

        self.ood_detector = SimpleOODDetector(
            numeric_cols=numeric_cols,
            contamination=0.05
        ).fit(edm_df)

        self.is_trained = True
        return self

    def predict(
        self,
        workpiece_material="Hardened EN31 steel",
        electrode_material="Copper",
        dielectric_fluid="EDM oil",
        peak_current_A=8,
        gap_voltage_V=60,
        pulse_on_us=300,
        pulse_off_us=30,
        flushing_pressure_kgf_cm2=18,
        machining_time_min=30,
        reject_ood=False,
        cavity_volume_cm3=5.0,
        material_density_g_cm3=7.8,
        batch_size=1
    ):
        if not self.is_trained:
            raise RuntimeError("EDMModelSuite must be trained before prediction.")

        row = {
            "workpiece_material": workpiece_material,
            "electrode_material": electrode_material,
            "dielectric_fluid": dielectric_fluid,
            "peak_current_A": peak_current_A,
            "gap_voltage_V": gap_voltage_V,
            "pulse_on_us": pulse_on_us,
            "pulse_off_us": pulse_off_us,
            "flushing_pressure_kgf_cm2": flushing_pressure_kgf_cm2,
            "machining_time_min": machining_time_min
        }

        ood_result = self.ood_detector.check(row)

        if reject_ood and ood_result.is_ood:
            return {
                "process_type": "EDM",
                "status": "REJECTED_BY_OOD",
                "ood": ood_result.to_dict()
            }

        X_input = pd.DataFrame([row])

        predictions = {}

        for target, model in self.models.items():
            predictions[target] = float(model.predict(X_input)[0])

        ra = predictions.get("surface_roughness_um", 3.2)
        mrr = predictions.get("material_removal_rate_g_min", None)
        ewr = predictions.get("electrode_wear_rate_g_min", None)

        cost = estimate_edm_cost(
            predicted_ra_um=ra,
            predicted_mrr_g_min=mrr,
            predicted_ewr_g_min=ewr,
            cavity_volume_cm3=cavity_volume_cm3,
            material_density_g_cm3=material_density_g_cm3,
            batch_size=batch_size
        )

        return {
            "process_type": "EDM",
            "status": "OK_WITH_WARNING" if ood_result.risk_level == CAUTION else "OK",
            "input": row,
            "ood": ood_result.to_dict(),
            "predictions": predictions,
            "cost_estimate": cost
        }

    def save(self, path="./htm_database/edm_models/"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "edm_model_suite.pkl"))

    @staticmethod
    def load(path="./htm_database/edm_models/edm_model_suite.pkl"):
        return joblib.load(path)


def load_edm_data(filepath="data/edm.csv"):
    """
    Expected EDM CSV columns:
        workpiece_material
        electrode_material
        dielectric_fluid
        peak_current_A
        gap_voltage_V
        pulse_on_us
        pulse_off_us
        flushing_pressure_kgf_cm2
        machining_time_min
        surface_roughness_um
        out_of_roundness_um
        material_removal_rate_g_min
        electrode_wear_rate_g_min
    """
    edm_df = pd.read_csv(filepath)
    return edm_df


# ======================================================================
# SECTION 9: CONVENTIONAL MACHINING WITH OOD + COST
# ======================================================================

def fit_conventional_ood(db):
    """
    Fit OOD detector on conventional machining database.
    Call this after db.combined_df is ready.
    """

    if db.combined_df is None:
        raise ValueError("Database is empty. Add or generate data first.")

    numeric_cols = [
        "cutting_speed_mpm",
        "feed_mm_rev",
        "depth_of_cut_mm",
        "nose_radius_mm"
    ]

    numeric_cols = [
        c for c in numeric_cols
        if c in db.combined_df.columns
    ]

    detector = SimpleOODDetector(
        numeric_cols=numeric_cols,
        contamination=0.05
    ).fit(db.combined_df)

    return detector


def predict_conventional_with_safety_and_cost(
    db,
    suite,
    ood_detector,
    material="Ti6Al4V",
    vc=60,
    f=0.2,
    doc=0.5,
    r_n=0.8,
    tool="Carbide",
    cool="MQL",
    reject_ood=False,
    part_volume_cm3=100.0,
    batch_size=1
):
    input_row = {
        "cutting_speed_mpm": vc,
        "feed_mm_rev": f,
        "depth_of_cut_mm": doc,
        "nose_radius_mm": r_n
    }

    ood_result = ood_detector.check(input_row)

    if reject_ood and ood_result.is_ood:
        return {
            "process_type": "Conventional machining",
            "status": "REJECTED_BY_OOD",
            "ood": ood_result.to_dict()
        }

    predictions = forward_predict(
        db=db,
        suite=suite,
        material=material,
        vc=vc,
        f=f,
        doc=doc,
        r_n=r_n,
        tool=tool,
        cool=cool
    )

    predicted_ra = predictions.get("Ra Model")
    predicted_tool_wear = predictions.get("Tool Wear Model")
    predicted_mrr = predictions.get("MRR (cm3/min)")

    cost = estimate_conventional_cost(
        predicted_ra_um=predicted_ra,
        predicted_tool_wear_mm=predicted_tool_wear,
        predicted_mrr_cm3_min=predicted_mrr,
        tool_type=tool,
        coolant_type=cool,
        part_volume_cm3=part_volume_cm3,
        batch_size=batch_size
    )

    return {
        "process_type": "Conventional machining",
        "status": "OK_WITH_WARNING" if ood_result.risk_level == CAUTION else "OK",
        "input": {
            "material": material,
            "cutting_speed_mpm": vc,
            "feed_mm_rev": f,
            "depth_of_cut_mm": doc,
            "nose_radius_mm": r_n,
            "tool_type": tool,
            "coolant_type": cool
        },
        "ood": ood_result.to_dict(),
        "predictions": predictions,
        "cost_estimate": cost
    }


# ======================================================================
# SECTION 10: FINAL UNIFIED PREDICTOR
# ======================================================================

def final_htm_predict(
    process_type,
    db=None,
    suite=None,
    conventional_ood=None,
    edm_suite=None,
    **kwargs
):
    """
    Final unified function.

    For conventional machining:
        final_htm_predict(
            process_type="conventional",
            db=db,
            suite=suite,
            conventional_ood=conventional_ood,
            material="Ti6Al4V",
            vc=60,
            f=0.2,
            doc=0.5,
            r_n=0.8,
            tool="Carbide",
            cool="MQL"
        )

    For EDM:
        final_htm_predict(
            process_type="edm",
            edm_suite=edm_suite,
            workpiece_material="Hardened EN31 steel",
            peak_current_A=8,
            gap_voltage_V=60,
            pulse_on_us=300,
            pulse_off_us=30
        )
    """

    p = process_type.lower().strip()

    if p in ["conventional", "turning", "milling", "cnc", "machining"]:
        if db is None or suite is None or conventional_ood is None:
            raise ValueError(
                "For conventional prediction, provide db, suite, and conventional_ood."
            )

        return predict_conventional_with_safety_and_cost(
            db=db,
            suite=suite,
            ood_detector=conventional_ood,
            **kwargs
        )

    elif p in ["edm", "die sinking edm", "die-sinking edm", "spark erosion"]:
        if edm_suite is None:
            raise ValueError(
                "For EDM prediction, provide edm_suite."
            )

        return edm_suite.predict(**kwargs)

    else:
        raise ValueError(
            "process_type must be either 'conventional' or 'edm'."
        )


# ======================================================================
# SECTION 11: EXAMPLE USAGE
# Keep this commented if you only want functions.
# ======================================================================

"""
# -------------------------
# 1. Train conventional model
# -------------------------

db = MaterialDatabase()

for mat in db.MATERIAL_PROPERTIES.keys():
    db.generate_synthetic_data(mat, n_samples=300)

df_enc = prepare_features(db.combined_df)

suite = HTMModelSuite()
suite.train_all(df_enc, n_trials=5)

conventional_ood = fit_conventional_ood(db)

# -------------------------
# 2. Train EDM model
# -------------------------

edm_df = load_edm_data("data/edm.csv")

edm_suite = EDMModelSuite()
edm_suite.train(edm_df)

# -------------------------
# 3. Conventional prediction
# -------------------------

cnc_result = final_htm_predict(
    process_type="conventional",
    db=db,
    suite=suite,
    conventional_ood=conventional_ood,
    material="Ti6Al4V",
    vc=60,
    f=0.2,
    doc=0.5,
    r_n=0.8,
    tool="Carbide",
    cool="MQL",
    part_volume_cm3=100,
    batch_size=10,
    reject_ood=False
)

print(cnc_result)

# -------------------------
# 4. EDM prediction
# -------------------------

edm_result = final_htm_predict(
    process_type="edm",
    edm_suite=edm_suite,
    workpiece_material="Hardened EN31 steel",
    electrode_material="Copper",
    dielectric_fluid="EDM oil",
    peak_current_A=8,
    gap_voltage_V=60,
    pulse_on_us=300,
    pulse_off_us=30,
    flushing_pressure_kgf_cm2=18,
    machining_time_min=30,
    cavity_volume_cm3=5,
    material_density_g_cm3=7.8,
    batch_size=10,
    reject_ood=False
)

print(edm_result)
"""
