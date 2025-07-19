import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
                
                tab1, tab2, tab3, tab4 = st.tabs(["Gráficos", "Métricas", "Comparação", "Exportar"])
                
                with tab1:
                    st.subheader("Predições vs Valores Reais")
                    df_plot = resultados.loc[idx_test]
                    cols = ['Real'] + [c for c in df_plot.columns if any(m in c for m in modelos)]
                    
                    fig = go.Figure()
                    
                    # Linha real
                    fig.add_trace(go.Scatter(
                        x=df_plot.index,
                        y=df_plot['Real'],
                        mode='lines',
                        name='Valor Real',
                        line=dict(color='black', width=3),
                        hovertemplate='<b>Real</b><br>Data: %{x}<br>Valor: $%{y:.2f}<extra></extra>'
                    ))
                    # Predições
                    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
                    for i, col in enumerate([c for c in cols if c != 'Real']):
                        fig.add_trace(go.Scatter(
                            x=df_plot.index,
                            y=df_plot[col],
                            mode='lines',
                            name=f'Predição {col}',
                            line=dict(color=colors[i % len(colors)], width=2, dash='dot'),
                            hovertemplate=f'<b>{col}</b><br>Data: %{{x}}<br>Valor: $%{{y:.2f}}<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title=f"Predições para {moeda_alvo} - Horizonte de {dias_pred} dias",
                        xaxis_title="Data",
                        yaxis_title="Preço (USD)",
                        hovermode='x unified',
                        height=600,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Análise de Erros")
                    fig_erro = go.Figure()
                    
                    for col in [c for c in cols if c != 'Real']:
                        erro = df_plot[col] - df_plot['Real']
                        fig_erro.add_trace(go.Scatter(
                            x=df_plot.index,
                            y=erro,
                            mode='lines',
                            name=f'Erro {col}',
                            hovertemplate=f'<b>Erro {col}</b><br>Data: %{{x}}<br>Erro: $%{{y:.2f}}<extra></extra>'
                        ))
                    
                    fig_erro.add_hline(y=0, line_dash="dash", line_color="black")
                    
                    fig_erro.update_layout(
                        title="Erro de Predição por Modelo",
                        xaxis_title="Data",
                        yaxis_title="Erro (Predição - Real)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_erro, use_container_width=True)
                
                with tab2:
                    st.subheader("Métricas de desempenho")

                    rows = []
                    for m, dias in metricas.items():
                        for d, mets in dias.items():
                            rows.append({"Modelo": m, "Horizonte": f"{d} dias", **mets})
                    
                    df_metricas = pd.DataFrame(rows)
                    st.dataframe(df_metricas, use_container_width=True)

                    if len(df_metricas) > 1:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            fig_mae = px.bar(df_metricas, x='Modelo', y='MAE', 
                                           title='Mean Absolute Error (MAE)')
                            st.plotly_chart(fig_mae, use_container_width=True)
                        
                        with col2:
                            fig_r2 = px.bar(df_metricas, x='Modelo', y='R2', 
                                          title='R² Score')
                            st.plotly_chart(fig_r2, use_container_width=True)
                        
                        with col3:
                            if 'Acurácia_Direcional' in df_metricas.columns:
                                fig_dir = px.bar(df_metricas, x='Modelo', y='Acurácia_Direcional', 
                                               title='Acurácia Direcional')
                                st.plotly_chart(fig_dir, use_container_width=True)
                
                with tab3:
                    st.subheader("Comparação detalhada")

                    modelo_comparar = st.selectbox("Selecione um modelo para análise:", modelos)
                    
                    if modelo_comparar:
                        df_modelo = df_plot[['Real', modelo_comparar]].dropna()

                        fig_scatter = px.scatter(
                            x=df_modelo['Real'], 
                            y=df_modelo[modelo_comparar],
                            title=f'Real vs Predito - {modelo_comparar}',
                            labels={'x': 'Valor Real', 'y': 'Valor Predito'}
                        )
                        fig_scatter.add_shape(
                            type="line",
                            x0=df_modelo['Real'].min(),
                            x1=df_modelo['Real'].max(),
                            y0=df_modelo['Real'].min(),
                            y1=df_modelo['Real'].max(),
                            line=dict(color="red", dash="dash")
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                with tab4:
                    st.subheader("Exportar resultados")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Baixar predições (CSV)",
                            resultados.to_csv().encode(),
                            "predicoes_regressao.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        st.download_button(
                            "Baixar métricas (CSV)",
                            df_metricas.to_csv(index=False).encode(),
                            "metricas_regressao.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    st.subheader("Resumo")
                    melhor_modelo = df_metricas.loc[df_metricas['R2'].idxmax(), 'Modelo']
                    melhor_r2 = df_metricas['R2'].max()
                    
                    st.info(f"""
                    **Análise Concluída!**
                    **Melhor modelo**: {melhor_modelo} (R² = {melhor_r2:.4f})
                    **Ativos analisados**: {moeda_alvo} + {len(auxiliares)} auxiliares
                    **Horizonte**: {dias_pred} dias à frente
                    **Período**: {inicio.strftime('%d/%m/%Y')} a {fim.strftime('%d/%m/%Y')}
                    """)
                    
            except Exception as e:
                st.error(f"Erro na predição de regressão: {str(e)}")
    
    else:  
        with st.spinner("Executando modelos de classificação..."):
            try:
                resultados, metricas, idx_test = predizer_classificacao(
                    df=df,
                    target_col=moeda_alvo,
                    dias_previsao=dias_pred,
                    modelos_escolhidos=modelos,
                    n_lags=n_lags,
                    ma_window=ma_window,
                    params=params
                )
                
                st.success("Predição de classificação concluída!")
                tab1, tab2, tab3, tab4 = st.tabs(["Gráficos", "Métricas", "Matriz Confusão", "Exportar"])
                
                with tab1:
                    st.subheader("Classificações vs Tendência Real")
                    
                    df_plot = resultados.loc[idx_test]
                    cols = ['Classe_Real'] + [c for c in df_plot.columns if c.startswith('Classe_')]

                    fig = go.Figure()
                    
                    # Classe real
                    fig.add_trace(go.Scatter(x=df_plot.index,y=df_plot['Classe_Real'],mode='lines+markers',
                        name='Tendência Real',line=dict(color='black', width=3),marker=dict(size=8)
                    ))
                    
                    # Predições
                    colors = ['blue', 'red', 'green', 'orange', 'purple']
                    for i, col in enumerate([c for c in cols if c != 'Classe_Real']):
                        modelo_name = col.replace('Classe_', '')
                        fig.add_trace(go.Scatter(x=df_plot.index,y=df_plot[col],
                                                 mode='markers',name=f'Predição {modelo_name}',
                                                 marker=dict(color=colors[i % len(colors)], 
                                                 size=6, opacity=0.7)
                        ))
                    
                    fig.update_layout(
                        title=f"Classificação de Tendência para {moeda_alvo}",
                        xaxis_title="Data",
                        yaxis_title="Tendência (0=Queda, 1=Alta)",
                        height=600,
                        yaxis=dict(tickmode='linear', tick0=0, dtick=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.subheader("Métricas de classificação")
                    rows = []
                    for m, dias in metricas.items():
                        for d, mets in dias.items():
                            mets_clean = {k: v for k, v in mets.items() if k != 'matriz_confusao'}
                            rows.append({"Modelo": m, "Horizonte": f"{d} dias", **mets_clean})
                    
                    df_metricas = pd.DataFrame(rows)
                    st.dataframe(df_metricas, use_container_width=True)
                    
                    if len(df_metricas) > 1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_acc = px.bar(df_metricas, x='Modelo', y='Acurácia', 
                                           title='Acurácia por Modelo')
                            st.plotly_chart(fig_acc, use_container_width=True)
                        
                        with col2:
                            fig_f1 = px.bar(df_metricas, x='Modelo', y='F1', 
                                          title='F1 Score por Modelo')
                            st.plotly_chart(fig_f1, use_container_width=True)
                
                with tab3:
                    st.subheader("Matriz de Confusão")
                    
                    modelo_selecionado = st.selectbox("Selecione o modelo:", modelos)
                    
                    if modelo_selecionado in metricas and dias_pred in metricas[modelo_selecionado]:
                        if 'matriz_confusao' in metricas[modelo_selecionado][dias_pred]:
                            cm = metricas[modelo_selecionado][dias_pred]['matriz_confusao']
                            fig = px.imshow(
                                cm, 
                                text_auto=True, 
                                aspect="auto",
                                title=f"Matriz de Confusão - {modelo_selecionado}",
                                labels=dict(x="Predito", y="Real", color="Contagem"),
                                x=['Queda (0)', 'Alta (1)'],
                                y=['Queda (0)', 'Alta (1)']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Matriz de confusão não disponível para este modelo")
                
                with tab4:
                    st.subheader("Exportar Resultados")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Baixar Classificações (CSV)",
                            resultados.to_csv().encode(),
                            "classificacoes.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        st.download_button(
                            "Baixar Métricas (CSV)",
                            df_metricas.to_csv(index=False).encode(),
                            "metricas_classificacao.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    st.subheader("Resumo")
                    melhor_modelo = df_metricas.loc[df_metricas['Acurácia'].idxmax(), 'Modelo']
                    melhor_acc = df_metricas['Acurácia'].max()
                    
                    st.info(f"""
                    **Análise de classificação concluída!**
                    
                    **Melhor modelo**: {melhor_modelo} (Acurácia = {melhor_acc:.4f})
                    **Ativos analisados**: {moeda_alvo} + {len(auxiliares)} auxiliares  
                    **Horizonte**: {dias_pred} dias à frente
                    **Período**: {inicio.strftime('%d/%m/%Y')} a {fim.strftime('%d/%m/%Y')}
                    """)
                    
            except Exception as e:
                st.error(f"Erro na predição de classificação: {str(e)}")
