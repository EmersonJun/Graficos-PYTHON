"""
Dashboard e Relatório em Python para Regressão Linear e Não Linear
Arquivo: dashboard_regressao.py
Como usar (no terminal):
  1) Instale dependências: 
     pip install pandas numpy matplotlib scikit-learn statsmodels scipy plotly streamlit openpyxl
     pip install seaborn
  2) Coloque 'tabelinha.xlsx' na mesma pasta
  3) Execute relatório (opcional): python dashboard_regressao.py --report
  4) Execute o dashboard interativo: streamlit run dashboard_regressao.py

O script carrega os dados, faz análise estatística (reaproveitando parte do seu código),
executa regressão linear e não linear (polinomial e ajuste exponencial), gera gráficos
interativos com plotly e monta um dashboard com streamlit.

NOTA: o Streamlit espera que o arquivo contenha chamadas da API do Streamlit ao rodar com
`streamlit run`. Para gerar um relatório estático, use a flag --report.
"""

python dashboard_regressao.py --report

