import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yfinance as yf
import streamlit as st

# Configuração da página
st.set_page_config(
    page_title='Projeto 6', 
    page_icon='📈', 
    layout='wide'
)

# Descrição do projeto
st.header('Projeto 6')
st.markdown('''
    **Simulação de Monte Carlo para um ativo financeiro usando o 
    Modelo Geométrico de Movimento Browniano.**
''')

# Configuração da barra lateral
st.sidebar.header('Menu de opções')

@st.cache_data
def get_data(ticker, start_date, end_date):
    return yf.Ticker(ticker).history(start=start_date, end=end_date)


# Seleção do ticker
ticker = st.sidebar.text_input('Ticker do ativo', value='ITUB4.SA')

# Quantidade de dias para download e modelagem
modeling_days = st.sidebar.number_input('Dias para modelagem', min_value=5, value=100)

# Quantidade de dias para simulação
T = st.sidebar.number_input('Dias para simulação', min_value=1, value=50)

# Quantidade de simulações
n_simulations = st.sidebar.number_input('Número de simulações', min_value=1, value=300)

# Botão para executar a simulação
if st.sidebar.button('▶️ Simular'):
    # Coleta as cotações dos últimos modeling_days dias
    end_date = datetime.today()
    start_date = end_date - timedelta(days=modeling_days)
    df = df = get_data(ticker, start_date, end_date)

    ### Modela os parâmetros do GBM
    close_prices = df['Close'].tail(modeling_days)
    returns = close_prices.pct_change()

    # Preço atual (último preço de fechamento)
    S0 = close_prices.iloc[-1]

    # Taxa de retorno esperada (μ)
    mu = np.mean(returns)

    # Volatilidade (σ)
    sigma = np.std(returns)

    ### Realiza as simulações
    simulations = np.zeros((n_simulations, T))
    simulations[:, 0] = S0
    for t in range(1, T):
        Z = np.random.standard_normal(n_simulations)
        simulations[:, t] = simulations[:, t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * Z)

    ### Exibe as estatísticas da simulação
    st.markdown('''
        <h6 style="font-size: 19px; text-align: center; margin-top: 15px">
            Preço final da simulação em relação ao último preço de fechamento
        </h6>'''
        , unsafe_allow_html=True
    )

    final_prices = simulations[:, -1]
    mean_final_price = np.mean(final_prices)
    min_final_price = np.min(final_prices)
    max_final_price = np.max(final_prices)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label='Preço médio final', 
            value=round(mean_final_price, 2), 
            delta=round(mean_final_price - close_prices[-1], 2)
        )
    
    with col2:
        st.metric(
            label='Preço mínimo final', 
            value=round(min_final_price, 2), 
            delta=round(min_final_price - close_prices[-1], 2)
        )

    with col3:
        st.metric(
            label='Preço máximo final', 
            value=round(max_final_price, 2), 
            delta=round(max_final_price - close_prices[-1], 2)
        )

    ### Plota os preços das simulações
    colors = cm.rainbow(np.linspace(0, 1, n_simulations))
    plt.figure(figsize=(10, 6))
    plt.plot(close_prices.values, label='Cotações anteriores', color='blue', lw=1.5)

    # Plota cada simulação
    for i in range(n_simulations):
        plt.plot(np.arange(len(close_prices), len(close_prices) + T), simulations[i, :], color=colors[i], alpha=0.7, lw=1)

    # Calcula e plota o caminho médio das simulações
    mean_simulation = np.mean(simulations, axis=0)
    plt.plot(np.arange(len(close_prices), len(close_prices) + T), mean_simulation, color='black', label='Caminho médio', lw=1, linestyle='--')

    plt.title('Simulações de Monte Carlo')
    plt.xlabel('Período')
    plt.ylabel('Preço de fechamento')
    plt.legend(loc='upper left')
    plt.grid(True)
    st.pyplot(plt)

    ### Plota o histograma das simulações
    plt.figure(figsize=(10, 6))
    plt.hist(final_prices, bins=50, color='blue', alpha=0.6, edgecolor='black')
    plt.title('Histograma dos preços finais das Simulações de Monte Carlo')
    plt.xlabel('Preço Final')
    plt.ylabel('Frequência')
    plt.grid(axis='y', alpha=0.75)
    plt.axvline(np.mean(final_prices), color='red', linestyle='dashed', linewidth=2, label='Preço médio final')
    plt.legend()
    st.pyplot(plt)

st.sidebar.markdown('''
    <p style="margin-top: 30px; text-align: center">
        Projetos Python para o Mercado Financeiro<br>
        <a href="https://www.instagram.com/cientistadabolsa">@cientistadabolsa</a>
    </p>
''', unsafe_allow_html=True)