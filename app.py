import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from machining_logic import (
    MaterialDatabase,
    HTMModelSuite,
    forward_predict,
    inverse_optimise,
    get_pareto_front,
    recommend_strategy,
    compare_materials,
    continuous_learning_update,
    prepare_features
)

warnings.filterwarnings("ignore")

APP_VERSION = "real_cnc_edm_toolwear_fix_v1"

def find_col(df, names):
    lower = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for n in names:
        key = n.lower().strip().replace(" ", "_")
        if key in lower:
            return lower[key]
    return None

def num_col(df, col, default=np.nan):
    if col is None or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")

def str_col(df, col, default):
    if col is None or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return df[col].astype(str).fillna(default)

def safe_range(series, fallback):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return fallback
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        hi = lo + 1e-6
    return (lo, hi)

def standardize_cnc_dataframe(raw_df, default_material="Unknown"):
    df = raw_df.copy()

    material_col = find_col(df, ["material", "workpiece_material"])
    speed_col = find_col(df, ["cutting_speed_mpm", "cutting_speed_m_min", "cutting_speed", "vc"])
    rpm_col = find_col(df, ["spindle_speed_rpm", "rpm"])
    dia_col = find_col(df, ["tool_diameter_mm", "diameter_mm"])

    feed_col = find_col(df, ["feed_mm_rev", "feed_rate_mm_tooth", "feed_mm_tooth", "feed"])
    doc_col = find_col(df, ["depth_of_cut_mm", "axial_depth_mm", "ap_mm", "doc_mm"])
    nose_col = find_col(df, ["nose_radius_mm", "tool_radius_mm", "corner_radius_mm"])

    tool_col = find_col(df, ["tool_type", "tool_material", "coating_type"])
    coolant_col = find_col(df, ["coolant_type", "cutting_condition", "cutting_environment"])

    ra_col = find_col(df, ["surface_roughness_Ra_um", "surface_roughness_um", "surface_finish_um", "ra_um"])
    wear_col = find_col(df, ["tool_wear_VB_mm", "tool_wear_vb_mm", "tool_wear_mm", "flank_wear_mm", "vb_mm"])

    force_col = find_col(df, ["cutting_force_N", "cutting_force_total_N"])
    fx_col = find_col(df, ["cutting_force_x_N"])
    fy_col = find_col(df, ["cutting_force_y_N"])
    fz_col = find_col(df, ["cutting_force_z_N"])

    if speed_col:
        cutting_speed = num_col(df, speed_col)
    elif rpm_col and dia_col:
        cutting_speed = np.pi * num_col(df, dia_col) * num_col(df, rpm_col) / 1000
    else:
        cutting_speed = pd.Series([np.nan] * len(df), index=df.index)

    if nose_col:
        nose_radius = num_col(df, nose_col)
    elif dia_col:
        nose_radius = num_col(df, dia_col) / 2
    else:
        nose_radius = pd.Series([0.8] * len(df), index=df.index)

    if force_col:
        force = num_col(df, force_col)
    elif fx_col and fy_col and fz_col:
        force = np.sqrt(num_col(df, fx_col)**2 + num_col(df, fy_col)**2 + num_col(df, fz_col)**2)
    else:
        force = 1000 * num_col(df, feed_col, 0.1) * num_col(df, doc_col, 1.0)

    std = pd.DataFrame({
        "material": str_col(df, material_col, default_material),
        "cutting_speed_mpm": cutting_speed,
        "feed_mm_rev": num_col(df, feed_col),
        "depth_of_cut_mm": num_col(df, doc_col),
        "nose_radius_mm": nose_radius,
        "tool_type": str_col(df, tool_col, "Carbide"),
        "coolant_type": str_col(df, coolant_col, "Dry"),
        "surface_roughness_Ra_um": num_col(df, ra_col),
        "tool_wear_VB_mm": num_col(df, wear_col),
        "cutting_force_N": force
    })

    std = std.replace([np.inf, -np.inf], np.nan)
    std = std.dropna()

    # Important: prevents old 0.8 mm saturation problem
    std["tool_wear_VB_mm"] = std["tool_wear_VB_mm"].clip(0.001, 0.6)

    return std

def build_db_from_frame(cnc_df):
    db = MaterialDatabase()
    db.combined_df = cnc_df.reset_index(drop=True)

    db.data_store = {
        mat: grp.reset_index(drop=True)
        for mat, grp in db.combined_df.groupby("material")
    }

    for mat, grp in db.combined_df.groupby("material"):
        db.MATERIAL_PROPERTIES[mat] = {
            "hardness_HRC": None,
            "tensile_strength_MPa": 1000,
            "thermal_conductivity_W_mK": None,
            "density_g_cm3": 7.8,
            "machinability_index": 0.35,
            "recommended_tool": str(grp["tool_type"].mode().iloc[0]),
            "coolant": str(grp["coolant_type"].mode().iloc[0]),
            "cutting_speed_range_mpm": safe_range(grp["cutting_speed_mpm"], (20, 200)),
            "feed_range_mm_rev": safe_range(grp["feed_mm_rev"], (0.01, 0.5)),
            "doc_range_mm": safe_range(grp["depth_of_cut_mm"], (0.1, 4.0)),
        }

    return db

def load_real_cnc_database():
    frames = []

    files = [
        ("data/Nickel Alloy CNC Milling.csv", "Nickel Alloy"),
        ("data/Ti6Al4V End Milling.csv", "Ti6Al4V"),
    ]

    loaded = []

    for path, mat in files:
        if os.path.exists(path):
            raw = pd.read_csv(path)
            std = standardize_cnc_dataframe(raw, default_material=mat)
            if len(std) > 0:
                frames.append(std)
                loaded.append(path)

    if len(frames) == 0:
        db = MaterialDatabase()
        for mat in ["Inconel_718", "Ti6Al4V", "AISI_4340", "SS_316L"]:
            db.generate_synthetic_data(mat, n_samples=300)
        db.combined_df["tool_wear_VB_mm"] = db.combined_df["tool_wear_VB_mm"].clip(0.001, 0.6)
        return db, "Real CNC files not found. Using corrected fallback data."

    cnc_df = pd.concat(frames, ignore_index=True)
    db = build_db_from_frame(cnc_df)

    return db, f"Real CNC data loaded: {len(cnc_df)} rows from {', '.join(loaded)}."

def safe_tool_wear(raw_value, db):
    raw_value = float(raw_value)
    wear = pd.to_numeric(db.combined_df["tool_wear_VB_mm"], errors="coerce").dropna()

    if len(wear) == 0:
        return float(np.clip(raw_value, 0.001, 0.6))

    q01 = float(wear.quantile(0.01))
    q99 = float(wear.quantile(0.99))

    return float(np.clip(raw_value, max(0.001, q01), min(0.6, q99)))

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HTM Machining Intelligence",
    layout="wide"
)

# ============================================================
# OOD DETECTOR
# ============================================================

SAFE = "SAFE"
CAUTION = "CAUTION"
OOD = "OOD"


class OODResult:
    def __init__(self, risk_level, is_ood, isolation_score, bound_violations, reason):
        self.risk_level = risk_level
        self.is_ood = is_ood
        self.isolation_score = isolation_score
        self.bound_violations = bound_violations
        self.reason = reason

    def to_dict(self):
        return {
            "risk_level": self.risk_level,
            "is_ood": self.is_ood,
            "isolation_score": round(self.isolation_score, 4),
            "bound_violations": self.bound_violations,
            "reason": self.reason
        }


class SimpleOODDetector:
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

        x_scaled = self.scaler.fit_transform(clean)
        self.iso.fit(x_scaled)
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


# ============================================================
# COST ESTIMATORS
# ============================================================

def ra_to_iso_n(ra_um):
    thresholds = [
        (0.025, "N1"),
        (0.05, "N2"),
        (0.1, "N3"),
        (0.2, "N4"),
        (0.4, "N5"),
        (0.8, "N6"),
        (1.6, "N7"),
        (3.2, "N8"),
        (6.3, "N9"),
        (12.5, "N10"),
        (25.0, "N11")
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


# ============================================================
# EDM MODEL
# ============================================================

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


def create_fallback_edm_data(n_samples=180, seed=42):
    rng = np.random.default_rng(seed)

    workpieces = [
        "Hardened EN31 steel",
        "Ti-13Zr-13Nb titanium alloy"
    ]

    electrodes = [
        "Copper",
        "Graphite",
        "Cu-TiC cermet tool tip (75% Cu, 25% TiC)"
    ]

    dielectrics = [
        "EDM oil",
        "commercial EDM oil",
        "kerosene"
    ]

    rows = []

    for _ in range(n_samples):
        workpiece = rng.choice(workpieces)
        electrode = rng.choice(electrodes)
        dielectric = rng.choice(dielectrics)

        current = rng.uniform(2, 18)
        voltage = rng.uniform(40, 120)
        ton = rng.uniform(50, 800)
        toff = rng.uniform(10, 200)
        flushing = rng.uniform(4, 25)
        time_min = rng.uniform(10, 70)

        duty = ton / (ton + toff)
        energy_index = current * voltage * duty

        mrr = 0.0008 * energy_index + rng.normal(0, 0.015)
        mrr = max(mrr, 0.005)

        ewr = mrr * rng.uniform(0.04, 0.22)

        ra = (
            0.8
            + 0.08 * current
            + 0.0025 * ton
            - 0.002 * toff
            + rng.normal(0, 0.25)
        )
        ra = float(np.clip(ra, 0.2, 12.0))

        oor = (
            3
            + 0.25 * current
            + 0.01 * ton
            - 0.03 * flushing
            + rng.normal(0, 1.2)
        )
        oor = max(oor, 0.5)

        rows.append({
            "workpiece_material": workpiece,
            "electrode_material": electrode,
            "dielectric_fluid": dielectric,
            "peak_current_A": round(current, 3),
            "gap_voltage_V": round(voltage, 3),
            "pulse_on_us": round(ton, 3),
            "pulse_off_us": round(toff, 3),
            "flushing_pressure_kgf_cm2": round(flushing, 3),
            "machining_time_min": round(time_min, 3),
            "surface_roughness_um": round(ra, 4),
            "out_of_roundness_um": round(oor, 4),
            "material_removal_rate_g_min": round(mrr, 6),
            "electrode_wear_rate_g_min": round(ewr, 6)
        })

    return pd.DataFrame(rows)


def load_edm_data(filepath="EDM Datasheet - Sheet1.csv"):
    if os.path.exists(filepath):
        return pd.read_csv(filepath), f"Real EDM CSV loaded from {filepath}"
    else:
        return create_fallback_edm_data(), f"{filepath} not found. Using built-in fallback EDM demo data."


class EDMModelSuite:
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
                continue

            x = sub[available_features].copy()
            y = sub[target].astype(float)

            num_cols = [
                c for c in EDM_NUMERIC_FEATURES
                if c in x.columns
            ]

            cat_cols = [
                c for c in EDM_CATEGORICAL_FEATURES
                if c in x.columns
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
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.2, random_state=42
                )

                model.fit(x_train, y_train)
                pred = model.predict(x_test)

                self.metrics[target] = {
                    "MAE": float(mean_absolute_error(y_test, pred)),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "R2": float(r2_score(y_test, pred))
                }
            else:
                model.fit(x, y)
                self.metrics[target] = {
                    "MAE": None,
                    "RMSE": None,
                    "R2": None
                }

            self.models[target] = model

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
            raise RuntimeError("EDM model is not trained.")

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

        x_input = pd.DataFrame([row])

        predictions = {}

        for target, model in self.models.items():
            predictions[target] = float(model.predict(x_input)[0])

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


# ============================================================
# CONVENTIONAL OOD + FINAL PREDICTORS
# ============================================================

def fit_conventional_ood(db):
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

    predictions["Tool Wear Model"] = safe_tool_wear(
        predictions["Tool Wear Model"],
        st.session_state.db
    )

    predicted_ra = predictions.get("Ra Model", 3.2)
    predicted_tool_wear = predictions.get("Tool Wear Model", 0.05)
    predicted_mrr = predictions.get("MRR (cm3/min)", vc * f * doc * 100)

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


def final_htm_predict(
    process_type,
    db=None,
    suite=None,
    conventional_ood=None,
    edm_suite=None,
    **kwargs
):
    p = process_type.lower().strip()

    if p in ["conventional", "turning", "milling", "cnc", "machining"]:
        return predict_conventional_with_safety_and_cost(
            db=db,
            suite=suite,
            ood_detector=conventional_ood,
            **kwargs
        )

    elif p in ["edm", "die sinking edm", "die-sinking edm", "spark erosion"]:
        return edm_suite.predict(**kwargs)

    else:
        raise ValueError("process_type must be either conventional or EDM.")


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if st.session_state.get("app_version") != APP_VERSION:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state["app_version"] = APP_VERSION

if "db" not in st.session_state:
    db, cnc_msg = load_real_cnc_database()
    st.session_state.db = db
    st.session_state.cnc_msg = cnc_msg

if "suite" not in st.session_state:
    suite = HTMModelSuite()
    df_enc = prepare_features(st.session_state.db.combined_df)
    suite.train_all(df_enc, n_trials=5)
    st.session_state.suite = suite

if "conventional_ood" not in st.session_state:
    st.session_state.conventional_ood = fit_conventional_ood(st.session_state.db)

if "edm_df" not in st.session_state:
    edm_df, edm_msg = load_edm_data("EDM Datasheet - Sheet1.csv")
    st.session_state.edm_df = edm_df
    st.session_state.edm_data_message = edm_msg

if "edm_suite" not in st.session_state:
    with st.spinner("Training EDM model..."):
        edm_suite = EDMModelSuite()
        edm_suite.train(st.session_state.edm_df)
        st.session_state.edm_suite = edm_suite


# ============================================================
# UI LAYOUT
# ============================================================

st.markdown(
    """
    <style>
    div.stTabs [data-baseweb="tab"] {
        font-size: 20px !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Basic")

process_type = st.sidebar.selectbox(
    "Machining Process",
    ["Conventional Machining", "EDM"]
)

selected_material = None
tool_choice = None
coolant_choice = None

if process_type == "Conventional Machining":
    selected_material = st.sidebar.selectbox(
        "Material",
        list(st.session_state.db.MATERIAL_PROPERTIES.keys())
    )

    mat_df = st.session_state.db.combined_df[
        st.session_state.db.combined_df["material"] == selected_material
    ]

    tool_options = sorted(mat_df["tool_type"].dropna().unique().tolist())
    coolant_options = sorted(mat_df["coolant_type"].dropna().unique().tolist())

    tool_choice = st.sidebar.selectbox(
        "Tool Type",
        tool_options if tool_options else ["Carbide", "CBN", "PCD"]
    )

    coolant_choice = st.sidebar.selectbox(
        "Coolant / Environment",
        coolant_options if coolant_options else ["Dry", "MQL", "Flood"]
    )
else:
    st.sidebar.warning("EDM UI is selected. Add EDM model block later if not already added.")


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "Predict",
    "Optimize",
    "Strategy Recommender",
    "Analytics",
    "Data Admin"
])


# ============================================================
# TAB 1: PREDICT
# ============================================================

with tabs[0]:
    st.subheader("Forward Prediction")

    if process_type == "Conventional Machining":
        c1, c2 = st.columns(2)

        with c1:
            vc = st.slider("Vc (m/min)", 20, 500, 80)
            f = st.slider("f (mm/rev)", 0.05, 0.5, 0.15)

        with c2:
            doc = st.slider("ap (mm)", 0.1, 4.0, 1.0)
            rn = st.selectbox("Nose Radius (mm)", [0.4, 0.8, 1.2])

        st.markdown("### Cost Inputs")

        c3, c4, c5 = st.columns(3)

        with c3:
            part_volume_cm3 = st.number_input(
                "Part Volume Removed (cm³)",
                min_value=0.1,
                value=100.0
            )

        with c4:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                value=10
            )

        with c5:
            reject_ood = st.checkbox(
                "Reject OOD Inputs",
                value=False
            )

        if st.button("Predict Conventional Outcomes"):
            result = final_htm_predict(
                process_type="conventional",
                db=st.session_state.db,
                suite=st.session_state.suite,
                conventional_ood=st.session_state.conventional_ood,
                material=selected_material,
                vc=vc,
                f=f,
                doc=doc,
                r_n=rn,
                tool=tool_choice,
                cool=coolant_choice,
                part_volume_cm3=part_volume_cm3,
                batch_size=batch_size,
                reject_ood=reject_ood
            )

            if result["status"] == "REJECTED_BY_OOD":
                st.error("Input rejected by OOD detector.")
                st.json(result["ood"])
            else:
                predictions = result["predictions"]
                cost = result["cost_estimate"]

                m1, m2, m3, m4 = st.columns(4)

                m1.metric(
                    "Surface Roughness Ra",
                    f"{predictions['Ra Model']:.3f} µm"
                )

                m2.metric(
                    "Tool Wear VB",
                    f"{predictions['Tool Wear Model']:.4f} mm"
                )

                m3.metric(
                    "Cutting Force",
                    f"{predictions['Cutting Force Model']:.1f} N"
                )

                m4.metric(
                    "Cost / Part",
                    f"${cost['total_cost_per_part_usd']:.2f}"
                )

                st.info(
                    f"Ra 95% Confidence Interval: "
                    f"[{predictions['Ra_Lower_CI']:.3f}, "
                    f"{predictions['Ra_Upper_CI']:.3f}] µm"
                )

                st.markdown("### OOD Check")
                st.json(result["ood"])

                st.markdown("### Prediction Details")
                st.dataframe(pd.DataFrame([predictions]), use_container_width=True)

                st.markdown("### Cost Estimate")
                st.dataframe(pd.DataFrame([cost]), use_container_width=True)

    else:
        st.info(st.session_state.edm_data_message)

        st.markdown("### EDM Cost Inputs")

        e1, e2, e3, e4 = st.columns(4)

        with e1:
            cavity_volume_cm3 = st.number_input(
                "Cavity Volume Removed (cm³)",
                min_value=0.1,
                value=5.0
            )

        with e2:
            material_density_g_cm3 = st.number_input(
                "Material Density (g/cm³)",
                min_value=0.1,
                value=7.8
            )

        with e3:
            edm_batch_size = st.number_input(
                "EDM Batch Size",
                min_value=1,
                value=10
            )

        with e4:
            edm_reject_ood = st.checkbox(
                "Reject EDM OOD Inputs",
                value=False
            )

        if st.button("Predict EDM Outcomes"):
            result = final_htm_predict(
                process_type="edm",
                edm_suite=st.session_state.edm_suite,
                workpiece_material=edm_workpiece_material,
                electrode_material=edm_electrode_material,
                dielectric_fluid=edm_dielectric_fluid,
                peak_current_A=peak_current_A,
                gap_voltage_V=gap_voltage_V,
                pulse_on_us=pulse_on_us,
                pulse_off_us=pulse_off_us,
                flushing_pressure_kgf_cm2=flushing_pressure_kgf_cm2,
                machining_time_min=machining_time_min,
                cavity_volume_cm3=cavity_volume_cm3,
                material_density_g_cm3=material_density_g_cm3,
                batch_size=edm_batch_size,
                reject_ood=edm_reject_ood
            )

            if result["status"] == "REJECTED_BY_OOD":
                st.error("EDM input rejected by OOD detector.")
                st.json(result["ood"])
            else:
                predictions = result["predictions"]
                cost = result["cost_estimate"]

                m1, m2, m3, m4 = st.columns(4)

                m1.metric(
                    "Surface Roughness",
                    f"{predictions.get('surface_roughness_um', 0):.3f} µm"
                )

                m2.metric(
                    "MRR",
                    f"{predictions.get('material_removal_rate_g_min', 0):.5f} g/min"
                )

                m3.metric(
                    "EWR",
                    f"{predictions.get('electrode_wear_rate_g_min', 0):.5f} g/min"
                )

                m4.metric(
                    "Cost / Part",
                    f"${cost['total_cost_per_part_usd']:.2f}"
                )

                st.markdown("### EDM OOD Check")
                st.json(result["ood"])

                st.markdown("### EDM Predictions")
                st.dataframe(pd.DataFrame([predictions]), use_container_width=True)

                st.markdown("### EDM Cost Estimate")
                st.dataframe(pd.DataFrame([cost]), use_container_width=True)


# ============================================================
# TAB 2: OPTIMIZE
# ============================================================

with tabs[1]:
    st.subheader("Parameter Search")

    if process_type == "Conventional Machining":
        t_ra = st.number_input(
            "Target Ra (µm)",
            0.1,
            8.0,
            1.0
        )

        if st.button("Calculate Optimal Vc, f, ap"):
            opt = inverse_optimise(
                st.session_state.db,
                st.session_state.suite,
                selected_material,
                t_ra,
                tool_choice,
                coolant_choice
            )

            st.success("Optimization Results")

            st.write(
                pd.DataFrame({
                    "Parameter": ["Vc", "f", "ap"],
                    "Value": opt
                })
            )

    else:
        st.warning(
            "EDM inverse optimization is not enabled yet. "
            "Use the EDM Predict tab for current EDM prediction and cost estimation."
        )


# ============================================================
# TAB 3: STRATEGY RECOMMENDER
# ============================================================

with tabs[2]:
    st.subheader("Autonomous Strategy Recommendation")

    if process_type == "Conventional Machining":
        req_ra = st.slider(
            "Desired Ra (µm)",
            0.4,
            8.0,
            1.0
        )

        req_mrr = st.number_input(
            "Minimum MRR Optional",
            0.0,
            50.0,
            0.0
        )

        if st.button("Find Best Strategy"):
            recs = recommend_strategy(
                st.session_state.db,
                st.session_state.suite,
                selected_material,
                req_ra,
                target_MRR=req_mrr
            )

            st.table(recs)

    else:
        st.warning(
            "EDM strategy recommender is not enabled yet. "
            "Train a larger EDM dataset first, then add EDM inverse optimization."
        )


# ============================================================
# TAB 4: ANALYTICS
# ============================================================

with tabs[3]:
    st.subheader("Analytics")

    if process_type == "Conventional Machining":
        st.markdown("### Tool Wear Training Data Check")
        st.write(st.session_state.db.combined_df["tool_wear_VB_mm"].describe())

        st.dataframe(
            st.session_state.db.combined_df[
                ["material", "cutting_speed_mpm", "feed_mm_rev", "depth_of_cut_mm", "tool_wear_VB_mm"]
            ].head(30),
            use_container_width=True
        )

        st.markdown("### Pareto & Material Comparison")

        c_a, c_b = st.columns(2)

        with c_a:
            if st.button("Generate Pareto Front"):
                pf = get_pareto_front(
                    st.session_state.db,
                    st.session_state.suite,
                    selected_material,
                    tool_choice,
                    coolant_choice
                )

                st.line_chart(pf, x="MRR", y="Ra")
                st.dataframe(pf, use_container_width=True)

        with c_b:
            if st.button("Cross-Material Radar"):
                comp, fig = compare_materials(
                    st.session_state.db,
                    st.session_state.suite,
                    80,
                    0.15,
                    1.0,
                    0.8,
                    tool_choice,
                    coolant_choice
                )

                st.pyplot(fig)
                st.dataframe(comp, use_container_width=True)

    else:
        st.markdown("### EDM Dataset Preview")
        st.info(st.session_state.edm_data_message)

        st.dataframe(
            st.session_state.edm_df.head(30),
            use_container_width=True
        )

        st.markdown("### EDM Model Metrics")

        if st.session_state.edm_suite.metrics:
            metrics_df = pd.DataFrame(st.session_state.edm_suite.metrics).T
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.warning("No EDM metrics available.")

        edm_df = st.session_state.edm_df

        if {
            "peak_current_A",
            "surface_roughness_um"
        }.issubset(edm_df.columns):
            st.markdown("### EDM Current vs Surface Roughness")
            chart_df = edm_df[[
                "peak_current_A",
                "surface_roughness_um"
            ]].dropna()

            st.scatter_chart(
                chart_df,
                x="peak_current_A",
                y="surface_roughness_um"
            )


# ============================================================
# TAB 5: DATA ADMIN
# ============================================================

with tabs[4]:
    st.subheader("Data Management")

    st.markdown("## Conventional Machining Data")

    admin_material = st.selectbox(
        "Select Material for Conventional Data Admin",
        list(st.session_state.db.MATERIAL_PROPERTIES.keys())
    )

    with st.expander("Register New Material Type"):
        with st.form("new_material_form"):
            new_mat_name = st.text_input(
                "Material Name",
                "Aluminum_7075"
            )

            col_p1, col_p2 = st.columns(2)

            with col_p1:
                uts = st.number_input(
                    "Tensile Strength (MPa)",
                    100,
                    3000,
                    500
                )

                v_low = st.number_input(
                    "Speed Min (mpm)",
                    10,
                    500,
                    100
                )

                v_high = st.number_input(
                    "Speed Max (mpm)",
                    10,
                    1000,
                    500
                )

            with col_p2:
                f_low = st.number_input(
                    "Feed Min (mm/rev)",
                    0.01,
                    1.0,
                    0.05
                )

                f_high = st.number_input(
                    "Feed Max (mm/rev)",
                    0.01,
                    2.0,
                    0.5
                )

                d_low = st.number_input(
                    "DoC Min (mm)",
                    0.1,
                    5.0,
                    0.5
                )

                d_high = st.number_input(
                    "DoC Max (mm)",
                    0.1,
                    10.0,
                    5.0
                )

            if st.form_submit_button("Register Material"):
                new_props = {
                    "tensile_strength_MPa": uts,
                    "cutting_speed_range_mpm": (v_low, v_high),
                    "feed_range_mm_rev": (f_low, f_high),
                    "doc_range_mm": (d_low, d_high)
                }

                st.session_state.db.MATERIAL_PROPERTIES[new_mat_name] = new_props

                st.success(
                    f"Material {new_mat_name} registered. "
                    "You can now upload CSV data for it."
                )

                st.rerun()

    uploaded_file = st.file_uploader(
        "Upload Conventional Experimental CSV",
        type=["csv"]
    )

    if uploaded_file and st.button("Ingest Conventional CSV & Retrain"):
        temp = "upload.csv"

        with open(temp, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.db.add_from_csv(temp, admin_material)

        continuous_learning_update(
            st.session_state.db,
            st.session_state.suite,
            [],
            n_trials=5
        )

        st.session_state.conventional_ood = fit_conventional_ood(
            st.session_state.db
        )

        st.success("Conventional database updated and models retrained.")

    st.markdown("### Add Single Conventional Experimental Record")

    with st.form("single_record_form"):
        col1, col2 = st.columns(2)

        with col1:
            vc_rec = st.number_input(
                "Cutting Speed (m/min)",
                value=80.0,
                min_value=0.0,
                max_value=500.0,
                step=0.1
            )

            f_rec = st.number_input(
                "Feed (mm/rev)",
                value=0.15,
                min_value=0.0,
                max_value=0.5,
                step=0.001,
                format="%.4f"
            )

            doc_rec = st.number_input(
                "Depth of Cut (mm)",
                value=1.0,
                min_value=0.0,
                max_value=4.0,
                step=0.01
            )

            rn_rec = st.selectbox(
                "Nose Radius (mm) for record",
                [0.4, 0.8, 1.2],
                index=1
            )

        with col2:
            tool_rec = st.selectbox(
                "Tool Type for record",
                ["Carbide", "CBN", "PCD"],
                index=0
            )

            coolant_rec = st.selectbox(
                "Coolant Type for record",
                ["Dry", "MQL", "Flood"],
                index=1
            )

            ra_rec = st.number_input(
                "Surface Roughness Ra (µm)",
                min_value=0.0,
                value=1.0,
                step=0.01,
                format="%.4f"
            )

            wear_rec = st.number_input(
                "Tool Wear VB (mm)",
                min_value=0.0,
                value=0.1,
                step=0.001,
                format="%.4f"
            )

            force_rec = st.number_input(
                "Cutting Force (N)",
                min_value=0.0,
                value=500.0,
                step=0.1
            )

        submitted = st.form_submit_button(
            "Add Single Conventional Record & Retrain"
        )

        if submitted:
            new_record = {
                "material": admin_material,
                "cutting_speed_mpm": vc_rec,
                "feed_mm_rev": f_rec,
                "depth_of_cut_mm": doc_rec,
                "nose_radius_mm": rn_rec,
                "tool_type": tool_rec,
                "coolant_type": coolant_rec,
                "surface_roughness_Ra_um": ra_rec,
                "tool_wear_VB_mm": wear_rec,
                "cutting_force_N": force_rec
            }

            continuous_learning_update(
                st.session_state.db,
                st.session_state.suite,
                [new_record],
                n_trials=5
            )

            st.session_state.conventional_ood = fit_conventional_ood(
                st.session_state.db
            )

            st.success(
                f"Single record added for {admin_material} and models retrained."
            )

    st.markdown("---")
    st.markdown("## EDM Data")

    edm_upload = st.file_uploader(
        "Upload EDM CSV",
        type=["csv"],
        key="edm_upload"
    )

    if edm_upload and st.button("Ingest EDM CSV & Retrain EDM Model"):
        edm_df_new = pd.read_csv(edm_upload)

        st.session_state.edm_df = edm_df_new
        st.session_state.edm_data_message = "EDM data uploaded from Streamlit file uploader."

        edm_suite = EDMModelSuite()
        edm_suite.train(st.session_state.edm_df)
        st.session_state.edm_suite = edm_suite

        st.success("EDM data updated and EDM models retrained.")

    st.markdown("### Current EDM Data Preview")
    st.dataframe(
        st.session_state.edm_df.head(20),
        use_container_width=True
    )

    st.markdown("### Required EDM CSV Columns")

    required_cols = pd.DataFrame({
        "Column": EDM_NUMERIC_FEATURES + EDM_CATEGORICAL_FEATURES + EDM_TARGETS
    })

    st.dataframe(required_cols, use_container_width=True)
