import yfinance as yf
import pandas as pd

def montar_dataframe(alvo, preditores, inicio, fim):
    tickers = [alvo] + preditores
    dfs = []
    
    for t in tickers:
        try:
            df = yf.download(t, start=inicio, end=fim, auto_adjust=True, progress=False)
            if df.empty:
                continue
            if hasattr(df.columns, 'levels'):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            if "Close" not in df.columns:
                continue
            df = df.reset_index()[["Date","Close"]].rename(columns={"Close": t})
            df["Date"] = pd.to_datetime(df["Date"])
            dfs.append(df)
            
        except Exception as e:
            print(f"Erro ao baixar {t}: {str(e)}")
            continue
    
    if not dfs:
        available_tickers = ', '.join([alvo] + preditores)
        raise ValueError(f"Nenhum dado disponível para os tickers informados: {available_tickers}. ")

    # merge em sequência
    df_final = dfs[0]
    for df in dfs[1:]:
        df_final = pd.merge(df_final, df, on="Date", how="inner")

    df_final = df_final.sort_values("Date").set_index("Date")
    return df_final
