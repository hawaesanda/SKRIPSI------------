import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="UVVisPredict",
    page_icon="🌿",
    layout="wide"
)

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

    df = pd.read_excel(
        "Latihan UV Vis 21 Juli 2022.xlsx"
    )

    hasil_all = pd.read_excel(
        "hasil_model_uvvis.xlsx"
    )

    comparison_s1 = pd.read_excel(
        "comparison_s1.xlsx"
    )

    comparison_s2 = pd.read_excel(
        "comparison_s2.xlsx"
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

        'Wavelength':
        wavelength,

        'Absorbance':
        absorbance

    })

    return (
        sample_name,
        transformed
    )

tab1, tab2, tab3= st.tabs([

    "Dashboard",
    "Analisis",
    "Tentang"
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
    sistem akan bekerja dan memberikan prediksi
    absorbansi berdasarkan model Machine Learning.

    <br><br>

    Solusi pintar untuk kemajuan teknologi
    di bidang pertanian.

    </p>

    </div>
    """, unsafe_allow_html=True)


with tab2:

    st.header(
        "🔮 Prediksi Absorbansi"
    )

    uploaded_file = st.file_uploader(

        "Upload Dataset UV-Vis",

        type=["xlsx"]

    )
    if uploaded_file is None:

        st.info(
            "Silakan upload dataset terlebih dahulu."
        )

        st.stop()

    uploaded_df = pd.read_excel(
        uploaded_file
    )
    st.write(uploaded_df.columns.tolist())

    st.success(
        "Dataset berhasil diupload"
    )

    st.dataframe(
        uploaded_df.head()
    )

    wavelength_cols = [
        col
        for col in uploaded_df.columns
        if isinstance(col,(int,float))
    ]
    sample = st.selectbox(

        "Pilih Sampel",

        uploaded_df['Name']
    )
    idx = uploaded_df[
        uploaded_df['Name']
        ==
        sample
    ].index[0]

    absorbance = uploaded_df.iloc[
        idx
    ][wavelength_cols].values

    transformed = pd.DataFrame({

        "Wavelength": wavelength_cols,

        "Absorbance": absorbance
    })
    X = transformed[['Wavelength']]
    y = transformed['Absorbance']
    
    model_name = st.selectbox(
        "Pilih Model",
        [
            "Linear Regression",
            "Polynomial Regression",
            "Random Forest"
        ]
    )

    wave_input = st.number_input(

        "Input Wavelength",

        min_value=200,

        max_value=1100,

        value=450,

        step=5

    )

    if st.button(
        "Prediksi"
    ):
        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.2,

            random_state=42
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_full_scaled = scaler.transform(
            X
        )

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

        np.array([[wave_input]])

    )

    pred_value = model.predict(
        wave_scaled
    )[0]

    st.metric(

        "Prediksi Absorbansi",

        f"{pred_value:.4f}"

    )

    pred_full = model.predict(
        X_full_scaled
    )

    fig = px.line(

        transformed,

        x='Wavelength',

        y='Absorbance',

        title='Spektrum UV-Vis'

    )

    fig.add_scatter(

        x=transformed['Wavelength'],

        y=pred_full,

        name='Prediksi'

    )

    fig.add_scatter(

        x=[wave_input],

        y=[pred_value],

        mode='markers',

        name='Input'

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab3:

    st.header(
        "ℹ️ Tentang SPECTRA"
    )

    st.markdown("""

### Spektrofotometri UV-Vis

Spektrofotometri UV-Vis merupakan metode analisis yang digunakan untuk mengukur kemampuan suatu sampel dalam menyerap cahaya pada panjang gelombang tertentu.

---

### Machine Learning yang Digunakan

**Linear Regression**

Model regresi linier untuk memodelkan hubungan panjang gelombang dan absorbansi.

**Polynomial Regression**

Model regresi non-linier yang mampu mengikuti pola spektrum lebih kompleks.

**Random Forest**

Model ensemble berbasis decision tree yang mampu menangani hubungan non-linier.

---

### Dataset

Data spektrum UV-Vis daun selada.

Rentang panjang gelombang:

**200 – 1100 nm**

---

### Tujuan Sistem

Membantu analisis data spektral UV-Vis secara cepat dan interaktif sebagai pendukung pengembangan teknologi pertanian berbasis Machine Learning.

""")