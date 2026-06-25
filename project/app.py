import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go

import os
from datetime import datetime
from sklearn.metrics import (
    r2_score,
    mean_squared_error
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

st.set_page_config(
    page_title="UVVisPredict",
    page_icon="🌿",
    layout="wide"
)

# DATABASE
def init_db():
    conn = sqlite3.connect(
        "uvvis.db",
        check_same_thread=False
    )
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tanggal TEXT,
        scenario TEXT,
        sample TEXT,
        model TEXT,
        wavelength REAL,
        prediksi REAL,
        rmse REAL,
        correlation REAL
    )
""")

    conn.commit()
    return conn
conn = init_db()

# css
st.markdown("""
<style>
.main{
    padding-top:1rem;
}
.stApp{
    background:
    linear-gradient(
    135deg,
    #0F2027,
    #203A43,
    #2C5364);
}
h1,h2,h3,p{
    color:white;
}
.metric-card{
    background:rgba(255,255,255,0.1);
    padding:20px;
    border-radius:15px;
    text-align:center;
}
</style>
""",
unsafe_allow_html=True)

@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).parent
    df = pd.read_excel(
        BASE_DIR / "Latihan UV Vis 21 Juli 2022.xlsx"
    )
    hasil_all = pd.read_excel(
        BASE_DIR / "hasil_model_uvvis.xlsx"
    )
    comparison_s1 = pd.read_excel(
        BASE_DIR / "comparison_s1.xlsx"
    )
    comparison_s2 = pd.read_excel(
        BASE_DIR / "comparison_s2.xlsx"
    )
    return (
        df,
        hasil_all,
        comparison_s1,
        comparison_s2
    )
df, hasil_all, comparison_s1, comparison_s2 = load_data()
drop_cols = [
    'No.',
    'Type',
    'Date/Time',
    'Note'
]

df_clean = df.drop(
    columns=drop_cols,
    errors='ignore'
)

wavelength_cols = [
    col
    for col in df_clean.columns
    if isinstance(col,(int,float))
]

wavelength = np.array(
    wavelength_cols
)

def transform_sample(
    df_clean,
    sample_index
):

    sample_name = df_clean.iloc[
        sample_index
    ]['Name']

    absorbance = df_clean.iloc[
        sample_index
    ][wavelength_cols].values

    transformed = pd.DataFrame({
        'Wavelength': wavelength,
        'Absorbance': absorbance
    })
    return (
        sample_name,
        transformed
    )

tab1, tab2, tab3= st.tabs([
    "Dashboard",
    "Analisis",
    "Riwayat"
])

with tab1:
    st.markdown("""
    <div style='
        text-align:center;
        padding-top:120px;
        padding-bottom:120px;
    '>
    <h1 style='
        font-size:90px;
        color:white;
        margin-bottom:10px;
    '>
    SPECTRA
    </h1>
    <h3 style='
        color:white;
        margin-bottom:30px;
    '>
    UV-Vis Absorbance Prediction System
    </h3>
    <p style='
        font-size:22px;
        width:70%;
        margin:auto;
        line-height:1.8;
        color:white;
    '>
    Gunakan data spektral Spektrofotometri UV-Vis,
    sistem akan bekerja dan membantu prediksi
    data absorbansi UV-Vis secara cepat dan interaktif menggunakan model Machine Learning.
    <br>
    Solusi pintar untuk kemajuan teknologi
    di bidang pertanian.
    </p>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header(
        "Prediksi Nilai Absorbansi"
    )
    uploaded_file = st.file_uploader(
        "Upload Dataset UV-Vis",
        type=["xlsx"]
    )

    if uploaded_file is not None:
        uploaded_df = pd.read_excel(
            uploaded_file
        )
        scenario = st.selectbox(
            "Pilih Skenario",
            [
                "Pilih Skenario",
                "Skenario 1",
                "Skenario 2"
            ]
        )
        comparison_sample = None
        if scenario == "Skenario 1":
            train_samples = [
                "Pilih Sampel",
                "UV 1264",
                "UV 1265",
                "UV 1266"
            ]
            comparison_sample = "UV 1267"
        elif scenario == "Skenario 2":
            train_samples = [
                "Pilih Sampel",
                "UV 1264",
                "UV 1265",
                "UV 1266",
                "UV 1267"
            ]
            comparison_sample = "UV 1268"
        else:
            train_samples = [
                "Pilih Sampel"
            ]
        if comparison_sample:
            st.info(f"Data pembanding: {comparison_sample}")

        sample = st.selectbox("Pilih Sampel", train_samples)

        model_name = st.selectbox(
            "Pilih Model",
            [
                "Pilih Model",
                "Linear Regression",
                "Polynomial Regression",
                "Random Forest"
            ]
        )

        wave_input = st.number_input("Input Wavelength (nm)", min_value=0, value=200, step=5)

        if st.button("Prediksi"):
            if wave_input < 200 or wave_input > 1100:
                st.error(
                    "Panjang gelombang harus berada pada rentang 200–1100 nm."
                )
                st.stop()
            if scenario == "Pilih Skenario":
                st.warning("Silakan pilih skenario terlebih dahulu.")
                st.stop()
            if sample == "Pilih Sampel":
                st.warning("Silakan pilih sampel terlebih dahulu.")
                st.stop()
            if model_name == "Pilih Model":
                st.warning("Silakan pilih model terlebih dahulu.")
                st.stop()
            wavelength_cols = [
                col
                for col
                in uploaded_df.columns

                if isinstance(
                    col,
                    (
                        int,
                        float
                    )
                )
            ]

            idx = uploaded_df[
                uploaded_df[
                    'Name'
                ]
                ==
                sample
            ].index[0]
            absorbance = uploaded_df.iloc[
                idx
            ][
                wavelength_cols
            ].values
            transformed = pd.DataFrame({
                "Wavelength":
                wavelength_cols,
                "Absorbance":
                absorbance
            })
            X = transformed[['Wavelength']]
            y = transformed['Absorbance']

            # split terbaik
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.1,
                random_state=42,
                shuffle=True
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(
                X_train
            )
            X_test_scaled = scaler.transform(
                X_test
            )
            X_full_scaled = scaler.transform(
                X
            )

            # model
            if model_name == "Linear Regression":

                model = LinearRegression()

            elif model_name == "Polynomial Regression":

                model = Pipeline([

                    (
                        'poly',
                        PolynomialFeatures(
                            degree=2
                        )
                    ),

                    (
                        'linear',
                        LinearRegression()
                    )

                ])

            else:

                model = RandomForestRegressor(

                    n_estimators=100,

                    random_state=42

                )

            model.fit(

                X_train_scaled,

                y_train

            )

            wave_scaled = scaler.transform(

                np.array([[
                    wave_input
                ]])

            )

            pred_value = model.predict(

                wave_scaled

            )[0]

            st.metric(

                "Prediksi Absorbansi",

                f"{pred_value:.4f}"

            )

            st.subheader(
                "Interpretasi Hasil Prediksi"
            )

            if pred_value >= 2:

                st.success(
                    f"""
                    Nilai absorbansi sebesar
                    {pred_value:.4f}
                    menunjukkan bahwa sampel
                    memiliki kemampuan
                    penyerapan cahaya yang tinggi
                    pada panjang gelombang
                    {wave_input} nm.
                    """
                )

            elif pred_value >= 1:

                st.info(
                    f"""
                    Nilai absorbansi sebesar
                    {pred_value:.4f}
                    menunjukkan tingkat
                    penyerapan cahaya sedang
                    pada panjang gelombang
                    {wave_input} nm.
                    """
                )

            else:

                st.warning(
                    f"""
                    Nilai absorbansi sebesar
                    {pred_value:.4f}
                    menunjukkan tingkat
                    penyerapan cahaya yang relatif
                    rendah pada panjang gelombang
                    {wave_input} nm.
                    """
                )

            # Prediksi sample terpilih
            pred_full = model.predict(
                X_full_scaled
            )

            # ==========================
            # Data pembanding
            # ==========================

            idx_comp = uploaded_df[

                uploaded_df['Name']
                ==
                comparison_sample

            ].index[0]

            comparison_absorbance = uploaded_df.iloc[
                idx_comp
            ][wavelength_cols].values

            comparison_df = pd.DataFrame({

                "Wavelength":
                wavelength_cols,

                "Absorbance":
                comparison_absorbance

            })

            # Prediksi model terhadap pembanding

            X_comp = comparison_df[
                ['Wavelength']
            ]

            X_comp_scaled = scaler.transform(
                X_comp
            )

            pred_compare = model.predict(
                X_comp_scaled
            )

            # RMSE pembanding

            rmse_compare = np.sqrt(

                mean_squared_error(

                    comparison_absorbance,

                    pred_compare

                )

            )

            # Korelasi pola

            correlation = np.corrcoef(

                np.array(
                    comparison_absorbance,
                    dtype=float
                ),

                np.array(
                    pred_compare,
                    dtype=float
                )

            )[0,1]
            
            col1, col2 = st.columns(2)

            with col1:

                st.metric(

                    "RMSE Pembanding",

                    f"{rmse_compare:.4f}"

                )

            with col2:

                st.metric(

                    "Korelasi Pola",

                    f"{correlation:.4f}"

                )

            fig = go.Figure()

            # Garis aktual (utuh)
            fig.add_trace(
                go.Scatter(
                    x=transformed['Wavelength'],
                    y=transformed['Absorbance'],
                    mode='lines',
                    name='Aktual',
                    line=dict(
                        width=3,
                        dash='solid'
                    )
                )
            )

            # Garis prediksi (putus-putus)
            fig.add_trace(
                go.Scatter(
                    x=transformed['Wavelength'],
                    y=pred_full,
                    mode='lines',
                    name='Prediksi',
                    line=dict(
                        width=3,
                        dash='dash'
                    )
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=transformed['Wavelength'],
                    y=comparison_absorbance,
                    mode='lines',
                    name='Data Pembanding',
                    line=dict(
                        dash='dot'
                    )
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=comparison_df['Wavelength'],
                    y=pred_compare,
                    mode='lines',
                    name='Prediksi Pembanding',
                    line=dict(
                        dash='dashdot'
                    )
                )
            )
            # Titik input wavelength
            fig.add_trace(
                go.Scatter(
                    x=[wave_input],
                    y=[pred_value],
                    mode='markers',
                    name='Input',
                    marker=dict(
                        size=10
                    )
                )
            )

            fig.update_layout(
                title='Aktual vs Prediksi',
                xaxis_title='Wavelength (nm)',
                yaxis_title='Absorbance',
                legend_title='Keterangan'
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader(
                "Kesimpulan Perbandingan"
            )

            if rmse_compare < 0.2 and correlation >= 0.9:

                st.success(

                    f"""
                    Model menunjukkan kesesuaian pola
                    yang sangat baik terhadap
                    data pembanding {comparison_sample}.

                    Nilai RMSE sebesar
                    {rmse_compare:.4f}
                    dan korelasi
                    {correlation:.4f}
                    menunjukkan bahwa model
                    mampu mengikuti pola
                    spektrum UV-Vis dengan baik.
                    """

                )

            elif correlation >= 0.7:

                st.info(

                    f"""
                    Model cukup mengikuti pola
                    data pembanding
                    {comparison_sample}.
                    """

                )

            else:

                st.warning(

                    f"""
                    Model belum mampu mengikuti
                    pola data pembanding
                    {comparison_sample}
                    secara optimal.
                    """

                )
            # simpan riwayat
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO history
                (
                    tanggal,
                    scenario,
                    sample,
                    model,
                    wavelength,
                    prediksi,
                    rmse,
                    correlation
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(datetime.now()),
                scenario,
                sample,
                model_name,
                float(wave_input),
                float(pred_value),
                float(rmse_compare),
                float(correlation)
            ))

            conn.commit()

            st.success(
                "Prediksi berhasil disimpan ke database."
            )

    else:

        st.info(
            "Silakan upload dataset UV-Vis."
        )

with tab3:

    st.header(
        "Riwayat Prediksi"
    )

    query = """
    SELECT
        tanggal,
        scenario,
        sample,
        model,
        wavelength,
        prediksi,
        rmse,
        correlation
    FROM history
    ORDER BY id DESC
    """

    history = pd.read_sql(
        query,
        conn
    )

    if len(history) > 0:

        # st.dataframe(
        #     history,
        #     use_container_width=True
        # )
        if len(history) > 0:
            for _, row in history.iterrows():
                with st.expander(
                    f"📄 {row['tanggal']} | {row['sample']} | {row['model']}"
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(
                            f"**Scenario:** {row['scenario']}"
                        )
                        st.write(
                            f"**Sample:** {row['sample']}"
                        )
                        st.write(
                            f"**Model:** {row['model']}"
                        )
                        st.write(
                            f"**Wavelength:** {row['wavelength']} nm"
                        )
                    with col2:
                        st.write(
                            f"**Prediksi:** {row['prediksi']:.4f}"
                        )
                        st.write(
                            f"**RMSE:** {row['rmse']:.4f}"
                        )
                        st.write(
                            f"**Korelasi:** {row['correlation']:.4f}"
                        )
                        if row['correlation'] >= 0.9:
                            st.success(
                                "Kesesuaian pola sangat baik"
                            )
                        elif row['correlation'] >= 0.7:
                            st.info(
                                "Kesesuaian pola cukup baik"
                            )
                        else:
                            st.warning(
                                "Kesesuaian pola rendah"
                            )
    # else:

    #     st.info(
    #         "Belum ada riwayat prediksi."
    #     )
    
    history = pd.read_sql(
        query,
        conn
    )

    # tombol hapus
    if st.button(
        "🗑 Hapus Riwayat"
    ):

        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM history"
        )

        conn.commit()

        st.success(
            "Riwayat berhasil dihapus"
        )

        st.rerun()

# with tab4:

#     st.header(
#         "ℹ️ Tentang SPECTRA"
#     )

#     st.markdown("""

# ### Spektrofotometri UV-Vis

# Spektrofotometri UV-Vis merupakan metode analisis yang digunakan untuk mengukur kemampuan suatu sampel dalam menyerap cahaya pada panjang gelombang tertentu.

# ---

# ### Machine Learning yang Digunakan

# **Linear Regression**

# Model regresi linier untuk memodelkan hubungan panjang gelombang dan absorbansi.

# **Polynomial Regression**

# Model regresi non-linier yang mampu mengikuti pola spektrum lebih kompleks.

# **Random Forest**

# Model ensemble berbasis decision tree yang mampu menangani hubungan non-linier.

# ---

# ### Dataset

# Data spektrum UV-Vis daun selada.

# Rentang panjang gelombang:

# **200 – 1100 nm**

# ---

# ### Tujuan Sistem

# Membantu analisis data spektral UV-Vis secara cepat dan interaktif sebagai pendukung pengembangan teknologi pertanian berbasis Machine Learning.

# """)