import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Input, Dot, Activation, BatchNormalization, Conv1D, Flatten
import xgboost as xgb
from tensorflow.keras.regularizers import l2
!pip install -q streamlit
import streamlit as st

# Load dataset
df = pd.read_csv("/content/TASK-ML-INTERN.csv")

# Data Preprocessing
print("Dataset Overview:")
print(df.info())
print("Missing Values:", df.isnull().sum().sum())

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df = df[numeric_cols]

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Normalize Data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(df.iloc[:, :-1])
y = scaler_y.fit_transform(df.iloc[:, -1].values.reshape(-1, 1))

# **Spectral Band Visualization**
plt.figure(figsize=(12, 5))
plt.plot(df.iloc[:, :-1].mean(axis=0))
plt.title("Average Spectral Reflectance")
plt.xlabel("Wavelength Bands")
plt.ylabel("Reflectance")
plt.show()

# **Heatmap for spectral data**
plt.figure(figsize=(10, 6))
sns.heatmap(df.iloc[:, :-1].corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Spectral Feature Correlation")
plt.show()

# **Dimensionality Reduction: PCA & t-SNE**
pca = PCA(n_components=15)
X_pca = pca.fit_transform(X)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.ravel(), cmap='viridis')
plt.title("t-SNE Visualization of Spectral Data")
plt.colorbar()
plt.show()

# **Train-Test Split**
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# **XGBoost Model**
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=7, subsample=0.9, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# **Random Forest Model for Comparison**
rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train.ravel())
y_pred_rf = rf_model.predict(X_test)

# **Model Evaluation**
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")
    
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name} - Actual vs. Predicted")
    plt.show()

evaluate_model(y_test, y_pred_xgb, "XGBoost")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# **Feature Importance (XGBoost)**
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, max_num_features=10)
plt.show()

# **LSTM Model - Reshape Input**
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# **LSTM with CNN layers**
input_layer = Input(shape=(X_train.shape[1], 1))
x = Conv1D(64, kernel_size=3, activation='relu')(input_layer)
x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.0005)))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = LSTM(64, return_sequences=False, kernel_regularizer=l2(0.0005))(x)
x = Dropout(0.4)(x)
out = Dense(1, activation='linear')(x)

# **Compile & Train LSTM**
model = Model(inputs=input_layer, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

history = model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test),
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

# **Plot Loss Curve**
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Epoch vs Loss')
plt.show()

# **LSTM Predictions & Evaluation**
y_pred_lstm = model.predict(X_test_lstm)
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm)
y_test_inv = scaler_y.inverse_transform(y_test)
evaluate_model(y_test_inv, y_pred_lstm, "LSTM")

# **Streamlit App for User Upload & Predictions**
def predict_new_data(uploaded_file):
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        new_data_scaled = scaler_X.transform(new_data)
        new_data_pca = pca.transform(new_data_scaled)
        new_data_lstm = new_data_pca.reshape((new_data_pca.shape[0], new_data_pca.shape[1], 1))
        predictions = model.predict(new_data_lstm)
        return scaler_y.inverse_transform(predictions)

st.title("Hyperspectral Imaging Prediction App")
uploaded_file = st.file_uploader("Upload Spectral Data CSV", type=["csv"])
if uploaded_file:
    predictions = predict_new_data(uploaded_file)
    st.write("Predicted DON Concentration:", predictions)

