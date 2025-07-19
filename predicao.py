import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

# 2. Criar features a partir do dataframe
def criar_features(df, target_col, n_lags=7, ma_window=5):
    df_feat = df.copy()
    for col in df.columns:
        # Cria lags (valor dos dia anterior)
        for lag in range(1, n_lags + 1):
            df_feat[f"{col}_lag{lag}"] = df_feat[col].shift(lag)

        # Cria média móvel com janela fixa
        df_feat[f"{col}_ma{ma_window}"] = df_feat[col].rolling(window=ma_window).mean()

        # Cria o retorno percentual de 1 dia (variação do preço de um dia para outro)
        df_feat[f"{col}_ret1"] = df_feat[col].pct_change()

    df_feat = df_feat.dropna()
    return df_feat

# 3. Treinar e prever
def predicao_regressao(df, target_col, dias_a_frente=1, modelo_nome='LinearRegression', prever_retorno=True,
                     modelo_params=None, n_lags=7, ma_window=5):
    # Gera as features baseadas na série temporal
    df_feat = criar_features(df, target_col, n_lags=n_lags, ma_window=ma_window)

    # Define a variável alvo: retorno percentual ou valor futuro
    if prever_retorno:
        df_feat["target"] = df_feat[target_col].pct_change(periods=dias_a_frente).shift(-dias_a_frente)
    else:
        df_feat["target"] = df_feat[target_col].shift(-dias_a_frente)

    df_feat = df_feat.dropna()

    # Define X (preditores) e y (alvo)
    X = df_feat.drop(columns=[target_col, 'target'])
    y = df_feat['target']

    # Separa em conjunto de treino (75%) e teste (25%)
    split = int(0.75 * len(df_feat))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Normalização
    scaler_X = StandardScaler().fit(X_train)
    X_train_s = scaler_X.transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    modelos = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(C=1.0, epsilon=0.01, kernel='rbf', gamma='auto'),
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance', metric='euclidean'),
        'MLP': MLPRegressor(max_iter=20000, random_state=42, hidden_layer_sizes=(50, 25))
    }

    # Seleciona o modelo
    modelo = modelos.get(modelo_nome, LinearRegression())

    # Atualiza o modelo com os parâmetros passados
    if modelo_params:
        modelo.set_params(**modelo_params)

    # Treinamento e predição
    if modelo_nome in ['SVR', 'MLP']:
        modelo.fit(X_train_s, y_train)
        y_pred = modelo.predict(X_test_s)
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

    # Avaliação da predição com métricas de regressão
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return y_test, y_pred, {'MAE': round(mae, 4), 'MSE': round(mse, 4), 'R2': round(r2, 4)}
