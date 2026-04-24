import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from machining_logic import (MaterialDatabase, HTMModelSuite, forward_predict, 
                             inverse_optimise, get_pareto_front, recommend_strategy, 
                             compare_materials, continuous_learning_update,prepare_features)

# Page Config
st.set_page_config(page_title="HTM Machining Intelligence", layout="wide")

# Initialize Session State
if 'db' not in st.session_state:
    db = MaterialDatabase()
    for mat in ['Inconel_718', 'Ti6Al4V', 'AISI_4340', 'SS_316L']:
        db.generate_synthetic_data(mat, n_samples=300)
    st.session_state.db = db

if 'suite' not in st.session_state:
    suite = HTMModelSuite()
    df_enc = prepare_features(st.session_state.db.combined_df)
    suite.train_all(df_enc, n_trials=5)
    st.session_state.suite = suite

# --- UI Layout ---
st.title("Intelligent Machining Database (HTM)")

# Sidebar
st.sidebar.header("Basic")
selected_material = st.sidebar.selectbox("Material", list(st.session_state.db.MATERIAL_PROPERTIES.keys()))
tool_choice = st.sidebar.selectbox("Tool Type", ["Carbide", "CBN", "PCD"])
coolant_choice = st.sidebar.selectbox("Coolant", ["Dry", "MQL", "Flood"])

tabs = st.tabs(["Predict", "Optimize", "Strategy Recommender", "Analytics", "Data Admin"])
st.markdown("""
    <style>
    div.stTabs [data-baseweb="tab"] {
        font-size: 20px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

with tabs[0]:
    st.subheader("Forward Prediction")
    c1, c2 = st.columns(2)
    with c1:
        vc = st.slider("Vc (m/min)", 20, 500, 80)
        f = st.slider("f (mm/rev)", 0.05, 0.5, 0.15)
    with c2:
        doc = st.slider("ap (mm)", 0.1, 4.0, 1.0)
        rn = st.selectbox("Nose Radius (mm)", [0.4, 0.8, 1.2])

    if st.button("Predict Outcomes"):
        res = forward_predict(st.session_state.db, st.session_state.suite, selected_material, vc, f, doc, rn, tool_choice, coolant_choice)
        m1, m2, m3 = st.columns(3)
        m1.metric("Surface Roughness (Ra)", f"{res['Ra Model']:.3f} um")
        m2.metric("Tool Wear (VB)", f"{res['Tool Wear Model']:.4f} mm")
        m3.metric("Cutting Force", f"{res['Cutting Force Model']:.1f} N")
        st.info(f"Uncertainty: Ra 95% Confidence Interval: [{res['Ra_Lower_CI']:.3f}, {res['Ra_Upper_CI']:.3f}] um")

with tabs[1]:
    st.subheader("Parameter Search")
    t_ra = st.number_input("Target Ra (um)", 0.1, 8.0, 1.0)
    if st.button("Calculate Optimal Vc, f, ap"):
        opt = inverse_optimise(st.session_state.db, st.session_state.suite, selected_material, t_ra, tool_choice, coolant_choice)
        st.success("Optimization Results")
        st.write(pd.DataFrame({"Parameter": ["Vc", "f", "ap"], "Value": opt}))

with tabs[2]:
    st.subheader("Autonomous Strategy Recommendation")
    req_ra = st.slider("Desired Ra (um)", 0.4, 8.0, 1.0)
    req_mrr = st.number_input("Minimum MRR (optional)", 0.0, 50.0, 0.0)
    if st.button("Find Best Strategy"):
        recs = recommend_strategy(st.session_state.db, st.session_state.suite, selected_material, req_ra, target_MRR=req_mrr)
        st.table(recs)

with tabs[3]:
    st.subheader("Pareto & Material Comparison")
    c_a, c_b = st.columns(2)
    with c_a:
        if st.button("Generate Pareto Front"):
            pf = get_pareto_front(st.session_state.db, st.session_state.suite, selected_material, tool_choice, coolant_choice)
            st.line_chart(pf, x='MRR', y='Ra')
            st.dataframe(pf)
    with c_b:
        if st.button("Cross-Material Radar"):
            comp,fig = compare_materials(st.session_state.db, st.session_state.suite, 80, 0.15, 1.0, 0.8, tool_choice, coolant_choice)
            st.pyplot(fig)
            st.dataframe(comp)

with tabs[4]:
    st.subheader("Data Management")
    
    with st.expander(" Register New Material Type"):
        with st.form("new_material_form"):
            new_mat_name = st.text_input("Material Name", "Aluminum_7075")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                uts = st.number_input("Tensile Strength (MPa)", 100, 3000, 500)
                v_low = st.number_input("Speed Min (mpm)", 10, 500, 100)
                v_high = st.number_input("Speed Max (mpm)", 10, 1000, 500)
            with col_p2:
                f_low = st.number_input("Feed Min (mm/rev)", 0.01, 1.0, 0.05)
                f_high = st.number_input("Feed Max (mm/rev)", 0.01, 2.0, 0.5)
                d_low = st.number_input("DoC Min (mm)", 0.1, 5.0, 0.5)
                d_high = st.number_input("DoC Max (mm)", 0.1, 10.0, 5.0)
            
            if st.form_submit_button("Register Material"):
                new_props = {
                    'tensile_strength_MPa': uts,
                    'cutting_speed_range_mpm': (v_low, v_high),
                    'feed_range_mm_rev': (f_low, f_high),
                    'doc_range_mm': (d_low, d_high)
                }
                st.session_state.db.MATERIAL_PROPERTIES[new_mat_name] = new_props
                st.success(f"Material {new_mat_name} registered! You can now upload CSV data for it.")
                st.rerun()
    uploaded_file = st.file_uploader("Upload Experimental CSV")
    if uploaded_file and st.button("Ingest & Retrain"):
        temp = "upload.csv"
        with open(temp, "wb") as f: f.write(uploaded_file.getbuffer())
        st.session_state.db.add_from_csv(temp, selected_material)
        continuous_learning_update(st.session_state.db, st.session_state.suite, [])
        st.success("Database updated and models retrained.")

                
    st.markdown(" Add Single Experimental Record")
    with st.form("single_record_form"):
        col1, col2 = st.columns(2)
        with col1:
            vc_rec = st.number_input("Cutting Speed (m/min)", value=0.0, min_value=0.0, max_value=500.0, step=0.1)
            f_rec = st.number_input("Feed (mm/rev)", value=0.00, min_value=0.0, max_value=0.5, step=0.001, format="%.4f")
            doc_rec = st.number_input("Depth of Cut (mm)", value=0.00, min_value=0.0, max_value=4.0, step=0.01)
            rn_rec = st.selectbox("Nose Radius (mm) for record", [0.4, 0.8, 1.2], index=1)
        with col2:
            tool_rec = st.selectbox("Tool Type for record", ["Carbide", "CBN", "PCD"], index=0)
            coolant_rec = st.selectbox("Coolant Type for record", ["Dry", "MQL", "Flood"], index=1)
            ra_rec = st.number_input("Surface Roughness Ra (μm)", min_value=0.0, value=1.0, step=0.01, format="%.4f")
            wear_rec = st.number_input("Tool Wear VB (mm)", min_value=0.0, value=0.1, step=0.001, format="%.4f")
            force_rec = st.number_input("Cutting Force (N)", min_value=0.0, value=500.0, step=0.1)

        submitted = st.form_submit_button("Add Single Record & Retrain")
        if submitted:
            new_record = {
                'material': selected_material,
                'cutting_speed_mpm': vc_rec,
                'feed_mm_rev': f_rec,
                'depth_of_cut_mm': doc_rec,
                'nose_radius_mm': rn_rec,
                'tool_type': tool_rec,
                'coolant_type': coolant_rec,
                'surface_roughness_Ra_um': ra_rec,
                'tool_wear_VB_mm': wear_rec,
                'cutting_force_N': force_rec,
                }
            continuous_learning_update(st.session_state.db, st.session_state.suite, [new_record])
            st.success(f"Single record added for {selected_material} and models retrained.")
       