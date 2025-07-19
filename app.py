import streamlit as st
import pandas as pd
from dados import montar_dataframe
from predicao import predizer_regressao, predizer_classificacao

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

    st.subheader("Algoritmos")
    if "Regressão" in tipo_analise:
        modelos = st.multiselect(
            "Algoritmos de Regressão", 
            ['LinearRegression', 'RandomForest', 'SVR', 'KNN', 'MLP'],
            default=['LinearRegression', 'RandomForest', 'KNN'],
            help="Selecione os algoritmos para comparação"
        )
    else:
        modelos = st.multiselect(
            "Algoritmos de Classificação", 
            ['LogisticRegression', 'RandomForest', 'SVC', 'KNN', 'MLP'],
            default=['LogisticRegression', 'RandomForest', 'KNN'],
            help="Selecione os algoritmos para comparação"
        )
    
    st.subheader("Parametrização")
    with st.expander("Configurações dos Algoritmos"):
        if 'RandomForest' in modelos:
            st.write("**Random Forest**")
            n_trees = st.slider("Número de árvores", 10, 200, 100)
            max_depth = st.selectbox("Profundidade máxima", [None, 5, 10, 15, 20], index=0)
        
        if 'KNN' in modelos:
            st.write("**K-Nearest Neighbors**")
            k_neighbors = st.slider("Número de vizinhos (K)", 1, 15, 3)
        
        if 'MLP' in modelos:
            st.write("**Rede Neural (MLP)**")
            hidden_layers = st.selectbox("Camadas ocultas", ["(50,)", "(50, 25)", "(100, 50)", "(100, 50, 25)"], index=1)
            max_iter = st.slider("Máx. iterações", 1000, 20000, 10000, step=1000)

st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    executar = st.button("Executar Predição", use_container_width=True, type="primary")

if executar:
    params = {}
    if 'RandomForest' in modelos:
        params['RandomForest'] = {
            'n_estimators': locals().get('n_trees', 100),
            'max_depth': locals().get('max_depth', None)
        }
    if 'KNN' in modelos:
        params['KNN'] = {'n_neighbors': locals().get('k_neighbors', 3)}
    if 'MLP' in modelos:
        hidden_size = eval(locals().get('hidden_layers', '(50, 25)'))
        params['MLP'] = {
            'hidden_layer_sizes': hidden_size,
            'max_iter': locals().get('max_iter', 10000)
        }
    
    with st.spinner("Carregando dados..."):
        df = montar_dataframe(moeda_alvo, auxiliares, inicio.isoformat(), fim.isoformat())
        st.success(f"Dados carregados: {len(df)} registros de {df.index.min().strftime('%d/%m/%Y')} até {df.index.max().strftime('%d/%m/%Y')}")
    
    # Executar predição
    if "Regressão" in tipo_analise:
        with st.spinner("Executando modelos de regressão..."):
            try:
                resultados, metricas, idx_test = predizer_regressao(
                    df=df,
                    target_col=moeda_alvo,
                    dias_previsao=dias_pred,
                    modelos_escolhidos=modelos,
                    n_lags=n_lags,
                    ma_window=ma_window,
                    params=params
                )   
                st.success("Predição de regressão concluída!")
