import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

file = "Latihan UV Vis 21 Juli 2022.xlsx"

df = pd.read_excel(file)

print("Ukuran Awal :", df.shape)

df.head()

drop_cols = [
    'No.',
    'Type',
    'Date/Time',
    'Note'
]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

print("Ukuran Setelah Cleaning :", df.shape)

df.head()

sample_1264 = df.iloc[0,1:].values.astype(float)
sample_1265 = df.iloc[1,1:].values.astype(float)
sample_1266 = df.iloc[2,1:].values.astype(float)
sample_1267 = df.iloc[3,1:].values.astype(float)
sample_1268 = df.iloc[4,1:].values.astype(float)

wavelength = df.columns[1:].astype(float)

def evaluate_model(model_name,
                   y_test,
                   pred_test):

    r2 = r2_score(
        y_test,
        pred_test
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred_test
        )
    )

    print(f"\n=== {model_name} ===")
    print("R² :", r2)
    print("RMSE :", rmse)

    return [
        model_name,
        r2,
        rmse
    ]

def run_scenario(
        X,
        y,
        wavelength,
        scenario_name,
        test_size=0.3,
        show_plot=True):

    split_label = f"{int((1-test_size)*100)}:{int(test_size*100)}"

    print("\n")
    print("="*60)
    print(f"{scenario_name} | Split {split_label}")
    print("="*60)

    print("Shape X :", X.shape)
    print("Shape y :", y.shape)

    # Split data sesuai rasio test_size

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False
    )

    wave_train, wave_test = train_test_split(
        wavelength,
        test_size=test_size,
        shuffle=False
    )

    # StandardScaler

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    results = []

    # ====================================
    # Visualisasi
    # ====================================
    print("Masuk ke visualisasi")
    def plot_result(
        wavelength_test,
        y_test,
        y_pred,
        model_name,
        scenario_name,
        split_label):

        plt.figure(figsize=(12,6))

        plt.plot(
            wavelength_test,
            y_test,
            label='Actual',
            linewidth=3
        )

        plt.plot(
            wavelength_test,
            y_pred, 
            label='Prediction',
            linestyle='--'
        )

        plt.title(
            f'{scenario_name}\n{model_name} | Split {split_label}'
        )

        plt.xlabel('Data Testing')
        plt.ylabel('Absorbance')

        plt.legend()
        plt.grid(True)

        plt.show()

    # ====================================
    # Linear Regression
    # ====================================

    linear = LinearRegression()

    linear.fit(
        X_train_scaled,
        y_train
    )

    pred_linear = linear.predict(
        X_test_scaled
    )

    if show_plot:
        plot_result(
            wave_test,
            y_test,
            pred_linear,
            "Linear Regression",
            scenario_name,
            split_label
        )
    results.append(

        evaluate_model(
            "Linear Regression",
            y_test,
            pred_linear
        )

    )

    # ====================================
    # Polynomial Regression
    # ====================================

    poly = Pipeline([

        (
            'poly',
            PolynomialFeatures(
                degree=2,
                include_bias=False
            )
        ),

        (
            'linear',
            LinearRegression()
        )

    ])

    poly.fit(
        X_train_scaled,
        y_train
    )

    pred_poly = poly.predict(
        X_test_scaled
    )

    if show_plot:

        plot_result(
            wave_test,
            y_test,
            pred_poly,
            "Polynomial Regression",
            scenario_name,
            split_label
        )

    results.append(

        evaluate_model(
            "Polynomial Regression",
            y_test,
            pred_poly
        )

    )

    # ====================================
    # Random Forest
    # ====================================

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rf.fit(
        X_train_scaled,
        y_train
    )

    pred_rf = rf.predict(
        X_test_scaled
    )

    if show_plot:

        plot_result(
            wave_test,
            y_test,
            pred_rf,
            "Random Forest",
            scenario_name,
            split_label
        )
        
    results.append(

        evaluate_model(
            "Random Forest",
            y_test,
            pred_rf
        )

    )

    # ====================================
    # Tabel Hasil
    # ====================================

    hasil = pd.DataFrame(

        results,

        columns=[
            "Model",
            "R2",
            "RMSE"
        ]

    )
    hasil.insert(0, "Split", split_label)
    hasil.insert(0, "Scenario", scenario_name)

    print("\n")
    print(hasil)

    return hasil

X1 = np.column_stack([

    sample_1264,
    sample_1265,
    sample_1266

])

y1 = sample_1267

split_list = [0.4, 0.3, 0.2, 0.1]

hasil_s1_all = []

for ts in split_list:
    hasil_tmp = run_scenario(
        X1,
        y1,
        wavelength,
        "SKENARIO 1 : (1264,1265,1266) -> 1267",
        test_size=ts,
        show_plot=True
    )
    hasil_s1_all.append(hasil_tmp)

hasil_s1_compare = pd.concat(hasil_s1_all, ignore_index=True)

print("\nPerbandingan Split - Skenario 1")
display(hasil_s1_compare.sort_values(["Split", "Model"]).reset_index(drop=True))

X2 = np.column_stack([

    sample_1264,
    sample_1265,
    sample_1266,
    sample_1267

])

y2 = sample_1268

split_list = [0.4, 0.3, 0.2, 0.1]

hasil_s2_all = []

for ts in split_list:
    hasil_tmp = run_scenario(
        X2,
        y2,
        wavelength,
        "SKENARIO 2 : (1264,1265,1266,1267) -> 1268",
        test_size=ts,
        show_plot=False
    )
    hasil_s2_all.append(hasil_tmp)

hasil_s2_compare = pd.concat(hasil_s2_all, ignore_index=True)

print("\nPerbandingan Split - Skenario 2")
display(hasil_s2_compare.sort_values(["Split", "Model"]).reset_index(drop=True))