import yfinance as yf
import pandas as pd

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
