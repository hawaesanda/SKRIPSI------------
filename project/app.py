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

tab1, tab2, tab3, tab4 = st.tabs([

    "🏠 Dashboard",
    "📂 Dataset",
    "📊 Hasil Penelitian",
    "🔮 Prediksi"

])

with tab1:

    st.title(
        "🌿 UVVisPredict"
    )

    st.markdown(
        """
        Analisis Spektrum UV-Vis
        Daun Selada Menggunakan
        Machine Learning
        """
    )

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Jumlah Sampel",
        len(df_clean)
    )

    c2.metric(
        "Jumlah Wavelength",
        len(wavelength)
    )

    c3.metric(
        "Jumlah Model",
        3
    )

    fig = px.line()

    for i in range(len(df_clean)):

        sample_name,data = transform_sample(
            df_clean,
            i
        )

        fig.add_scatter(

            x=data['Wavelength'],

            y=data['Absorbance'],

            name=sample_name

        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab2:

    st.header(
        "Dataset Asli"
    )

    st.dataframe(df)

    st.header(
        "Dataset Cleaning"
    )

    st.dataframe(
        df_clean
    )

    st.header(
        "Dataset Transformasi"
    )

    sample = st.selectbox(

        "Pilih Sampel",

        df_clean[
            'Name'
        ].tolist()

    )

    idx = df_clean[
        df_clean['Name']
        ==
        sample
    ].index[0]

    _, transformed = transform_sample(
        df_clean,
        idx
    )

    st.dataframe(
        transformed
    )

with tab3:

    st.header(
        "Hasil Penelitian"
    )

    scenario = st.selectbox(

        "Scenario",

        hasil_all[
            'Scenario'
        ].unique()

    )

    filtered = hasil_all[

        hasil_all[
            'Scenario'
        ]
        ==
        scenario

    ]

    st.dataframe(
        filtered
    )

    fig = px.bar(

        filtered,

        x='Model',

        y='R2',

        color='Split',

        barmode='group',

        title='Perbandingan R²'

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.bar(

        filtered,

        x='Model',

        y='RMSE',

        color='Split',

        barmode='group',

        title='Perbandingan RMSE'

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Model Terbaik Skenario 1"
    )

    st.dataframe(
        comparison_s1
    )

    st.subheader(
        "Model Terbaik Skenario 2"
    )

    st.dataframe(
        comparison_s2
    )

with tab4:

    st.header(
        "Prediksi Absorbansi"
    )

    sample = st.selectbox(

        "Sampel",

        df_clean[
            'Name'
        ].tolist()

    )

    model_name = st.selectbox(

        "Model",

        [

            "Linear Regression",

            "Polynomial Regression",

            "Random Forest"

        ]

    )

    wave_input = st.number_input(

        "Wavelength",

        min_value=200,

        max_value=1100,

        value=450,

        step=5

    )

    idx = df_clean[
        df_clean['Name']
        ==
        sample
    ].index[0]

    _, transformed = transform_sample(
        df_clean,
        idx
    )

    X = transformed[
        ['Wavelength']
    ]

    y = transformed[
        'Absorbance'
    ]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        shuffle=True

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

        x=wavelength,

        y=y,

        labels={

            'x':'Wavelength',

            'y':'Absorbance'

        }

    )

    fig.add_scatter(

        x=wavelength,

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

