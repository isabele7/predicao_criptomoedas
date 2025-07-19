import streamlit as st
import pandas as pd
from dados import montar_dataframe

st.set_page_config(page_title="Sistema de Predição de Criptomoedas", layout="wide")
st.title("Sistema de predição de criptomoedas")

with st.sidebar:
    st.header("Configurações do Sistema")
    st.subheader("Tipo de Análise")
    tipo_analise = st.radio("Escolha o tipo:", ["Regressão (Valor)", "Classificação (Tendência)"])
    st.subheader("Seleção de Dados")
    # Criptomoedas alvo 
    moeda_alvo = st.selectbox(
        "Criptomoeda Alvo", 
        ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'LINK-USD'],
        help="Selecione a criptomoeda que deseja prever"
    )
    # Ativos auxiliares 
    auxiliares = st.multiselect(
        "Ativos Auxiliares (Variáveis Preditoras)", 
        [
            # Outras criptomoedas
            'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'LINK-USD',
            # Ações importantes
            'SPY', 'QQQ', 'TSLA', 'AAPL', 'MSFT',
            # Índices globais
            '^GSPC', '^IXIC', '^DJI', '^BVSP', '^FTSE',
            # Commodities e moedas
            'GC=F', 'SI=F', 'DX-Y.NYB', 'CL=F',
            # Stablecoins
            'USDT-USD', 'USDC-USD'
        ],
        help="Selecione os ativos que serão usados como variáveis preditoras"
    )
    
    st.subheader("Período dos Dados")
    col1, col2 = st.columns(2)
    with col1:
        inicio = st.date_input("Data Início", value=pd.to_datetime("2023-01-01"))
    with col2:
        fim = st.date_input("Data Fim", value=pd.to_datetime("2024-06-01"))

    st.subheader("Configurações do Modelo")

    dias_pred = st.selectbox(
        "Horizonte de Predição", 
        [1, 3, 5, 7, 10, 15], 
        index=2,
        help="Quantos dias à frente prever"
    )

    st.subheader("Configurações de Janelamento")
    n_lags = st.slider("Número de Lags", 3, 20, 7, help="Quantos dias anteriores usar como features")
    ma_window = st.slider("Janela da Média Móvel", 3, 15, 5, help="Tamanho da janela para média móvel")
