import argparse
import math
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# ---------------------------- UTILITÁRIOS ----------------------------

def carregar_dados(caminho='tabelinha.xlsx', colunas_necessarias=None):
    if colunas_necessarias is None:
        colunas_necessarias = ["Ano Modelo", "Tamanho do motor", "Preço médio brl"]
    df = pd.read_excel(caminho)
    df.columns = [c.strip() for c in df.columns]
    existentes = [c for c in colunas_necessarias if c in df.columns]
    if not existentes:
        raise ValueError(f"Nenhuma das colunas necessárias encontradas. Colunas do arquivo: {df.columns.tolist()}")
    df_limpo = df.copy()
    for c in existentes:
        df_limpo[c] = pd.to_numeric(df_limpo[c], errors='coerce')
    df_limpo = df_limpo.dropna(subset=existentes).reset_index(drop=True)
    return df_limpo, existentes

def resumo_estatistico(df, colunas):
    resultados = {}
    for col in colunas:
        s = df[col]
        resultados[col] = {
            'media': float(s.mean()),
            'mediana': float(s.median()),
            'moda': float(s.mode().iat[0]) if not s.mode().empty else np.nan,
            'variancia': float(s.var()),
            'desvio_padrao': float(s.std()),
            'assimetria': float(s.skew()),
            'curtose': float(s.kurtosis()),
            'min': float(s.min()),
            'max': float(s.max()),
            'n': int(s.count())
        }
    return resultados

def ajusta_regressao_linear_simples(x, y):
    x_resh = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_resh, y)
    y_pred = model.predict(x_resh)
    r2 = r2_score(y, y_pred)
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    return model, y_pred, r2, rmse

def ajusta_regressao_linear_multivariada(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    return model, y_pred, r2, rmse

def ajusta_regressao_polinomial(x, y, grau=2):
    model = make_pipeline(PolynomialFeatures(degree=grau, include_bias=False), LinearRegression())
    x_resh = x.reshape(-1, 1)
    model.fit(x_resh, y)
    y_pred = model.predict(x_resh)
    r2 = r2_score(y, y_pred)
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    return model, y_pred, r2, rmse

def modelo_exponencial(x, a, b):
    return a * np.exp(b * x)

def ajusta_exponencial(x, y, p0=None):
    if p0 is None:
        p0 = [y.max() if y.max() > 0 else 1.0, 0.01]
    try:
        popt, pcov = curve_fit(modelo_exponencial, x, y, p0=p0, maxfev=10000)
        y_pred = modelo_exponencial(x, *popt)
        r2 = r2_score(y, y_pred)
        rmse = math.sqrt(mean_squared_error(y, y_pred))
        return popt, y_pred, r2, rmse
    except Exception:
        return None, None, None, None

def resumo_modelo_linear_statsmodels(X, y):
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    return model

# ---------------------------- RELATÓRIO ----------------------------

def gerar_relatorio_texto(df, colunas, resultados_stats, modelos_resumo):
    buf = StringIO()
    print("RELATÓRIO - ANÁLISE E MODELOS", file=buf)
    print("="*60, file=buf)
    print(f"Registros: {len(df)}\n", file=buf)

    print("== Estatísticas descritivas ==\n", file=buf)
    for col in colunas:
        r = resultados_stats[col]
        print(f"{col} - n={r['n']}, média={r['media']:.3f}, mediana={r['mediana']:.3f}, dp={r['desvio_padrao']:.3f}, assimetria={r['assimetria']:.3f}", file=buf)
    print("\n== Modelos ajustados ==\n", file=buf)
    for nome, info in modelos_resumo.items():
        print(f"Modelo: {nome}", file=buf)
        for k, v in info.items():
            if isinstance(v, (int, float, str)):
                print(f"  {k}: {v}", file=buf)
        print("\n", file=buf)
    return buf.getvalue()

# ---------------------------- DASHBOARD STREAMLIT ----------------------------

def streamlit_app(df, colunas):
    st.set_page_config(page_title='Dashboard de Regressões', layout='wide')
    st.title('📊 Dashboard: Análise de Regressões')

    st.sidebar.header('Configurações')
    target = st.sidebar.selectbox('Escolha a variável alvo (y)', colunas, index=len(colunas)-1)
    features = st.sidebar.multiselect('Escolha variáveis independentes (X)', [c for c in colunas if c != target], default=[c for c in colunas if c != target])
    grau_poly = st.sidebar.slider('Grau do polinômio (para regressão polinomial univariada)', 2, 5, 2)

    stats = resumo_estatistico(df, colunas)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Estatísticas Descritivas",
        "📈 Gráficos por Variável",
        "🔗 Correlações & Comparações",
        "📋 Resumo Final"
    ])

    # TAB 1
    with tab1:
        st.subheader("Estatísticas Resumidas")
        st.write(pd.DataFrame.from_dict(stats, orient='index'))

    # TAB 2
    with tab2:
        st.subheader("Distribuições e Outliers")
        for col in colunas:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Histograma - {col}"), use_container_width=True)
            with c2:
                st.plotly_chart(px.box(df, y=col, title=f"Boxplot - {col}"), use_container_width=True)

    # TAB 3
    with tab3:
        st.subheader("Correlação entre Variáveis")
        fig_corr = px.imshow(df[colunas].corr(), text_auto='.3f', aspect='auto', title='Matriz de Correlação')
        st.plotly_chart(fig_corr, use_container_width=True)

        if len(features) >= 1:
            xcol = features[0]
            ycol = target
            x = df[xcol].values
            y = df[ycol].values

            model_lin, y_pred_lin, r2_lin, rmse_lin = ajusta_regressao_linear_simples(x, y)
            model_poly, y_pred_poly, r2_poly, rmse_poly = ajusta_regressao_polinomial(x, y, grau=grau_poly)
            if np.all(y > 0):
                _, y_pred_exp, r2_exp, rmse_exp = ajusta_exponencial(x, y)
            else:
                r2_exp, rmse_exp, y_pred_exp = (None, None, None)

            df_plot = pd.DataFrame({xcol: x, ycol: y, 'lin_pred': y_pred_lin, 'poly_pred': y_pred_poly})
            if y_pred_exp is not None:
                df_plot['exp_pred'] = y_pred_exp

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot[xcol], y=df_plot[ycol], mode='markers', name='Observado'))
            fig.add_trace(go.Scatter(x=df_plot[xcol], y=df_plot['lin_pred'], mode='lines', name=f'Linear (R²={r2_lin:.3f})'))
            fig.add_trace(go.Scatter(x=df_plot[xcol], y=df_plot['poly_pred'], mode='lines', name=f'Polinomial {grau_poly} (R²={r2_poly:.3f})'))
            if 'exp_pred' in df_plot:
                fig.add_trace(go.Scatter(x=df_plot[xcol], y=df_plot['exp_pred'], mode='lines', name=f'Exponencial (R²={r2_exp:.3f})'))

            fig.update_layout(title=f"Ajustes de Regressão: {ycol} vs {xcol}", xaxis_title=xcol, yaxis_title=ycol)
            st.plotly_chart(fig, use_container_width=True)

            st.write("**Métricas de Ajuste (Univariadas)**")
            met_df = pd.DataFrame({
                'Modelo': ['Linear', f'Polinomial grau {grau_poly}', 'Exponencial'],
                'R²': [r2_lin, r2_poly, r2_exp if r2_exp is not None else np.nan],
                'RMSE': [rmse_lin, rmse_poly, rmse_exp if rmse_exp is not None else np.nan]
            })
            st.table(met_df)

        if len(features) >= 2:
            X = df[features].values
            y = df[target].values
            model_mv, y_pred_mv, r2_mv, rmse_mv = ajusta_regressao_linear_multivariada(X, y)
            st.subheader("Regressão Linear Multivariada")
            st.write(pd.DataFrame({'feature': features, 'coef': model_mv.coef_}))
            st.write(f"Intercepto: {model_mv.intercept_:.3f}")
            st.write(f"R²: {r2_mv:.3f} | RMSE: {rmse_mv:.3f}")

        # TAB 4
    with tab4:
        st.subheader("Resumo Final e Exportação")

        # Estatísticas descritivas
        st.write("📊 Estatísticas Descritivas")
        st.write(pd.DataFrame.from_dict(stats, orient='index'))

        # Métricas dos modelos
        st.write("📈 Métricas dos Modelos")
        modelos_resumo = {}
        if 'r2_lin' in locals():
            modelos_resumo['Linear Univariada'] = {'R2': r2_lin, 'RMSE': rmse_lin}
        if 'r2_poly' in locals():
            modelos_resumo[f'Polinomial grau {grau_poly}'] = {'R2': r2_poly, 'RMSE': rmse_poly}
        if 'r2_exp' in locals() and r2_exp is not None:
            modelos_resumo['Exponencial'] = {'R2': r2_exp, 'RMSE': rmse_exp}
        if 'r2_mv' in locals():
            modelos_resumo['Multivariada'] = {'R2': r2_mv, 'RMSE': rmse_mv}

        if modelos_resumo:
            st.write(pd.DataFrame(modelos_resumo).T)

        # Coeficientes multivariados
        if 'model_mv' in locals():
            st.write("📌 Coeficientes do Modelo Multivariado")
            st.write(pd.DataFrame({'feature': features, 'coef': model_mv.coef_}))
            st.write(f"Intercepto: {model_mv.intercept_:.3f}")

        # Matriz de covariância
        st.write("🔗 Matriz de Covariância")
        st.dataframe(df[colunas].cov())

        # Relatório completo em texto
        texto = gerar_relatorio_texto(df, colunas, stats, modelos_resumo)
        st.subheader("📑 Prévia do Relatório")
        st.text(texto)

        # Exportação
        if st.button("Exportar relatório .txt"):
            st.download_button("Download relatório", texto, file_name="relatorio_regressao.txt")

# ---------------------------- CLI ----------------------------

def gerar_report_cli(df, colunas):
    xcol = colunas[1] if len(colunas) > 1 else colunas[0]
    ycol = colunas[-1]
    x = df[xcol].values
    y = df[ycol].values

    model_lin, y_pred_lin, r2_lin, rmse_lin = ajusta_regressao_linear_simples(x, y)
    model_poly, y_pred_poly, r2_poly, rmse_poly = ajusta_regressao_polinomial(x, y, grau=2)
    popt_exp, y_pred_exp, r2_exp, rmse_exp = (None, None, None, None)
    if np.all(y > 0):
        popt_exp, y_pred_exp, r2_exp, rmse_exp = ajusta_exponencial(x, y)

    modelos_resumo = {
        'Linear univariada': {'R2': r2_lin, 'RMSE': rmse_lin},
        'Polinomial grau 2': {'R2': r2_poly, 'RMSE': rmse_poly}
    }
    if r2_exp is not None:
        modelos_resumo['Exponencial'] = {'R2': r2_exp, 'RMSE': rmse_exp}

    stats = resumo_estatistico(df, colunas)
    texto = gerar_relatorio_texto(df, colunas, stats, modelos_resumo)

    with open('relatorio_regressao.txt', 'w', encoding='utf-8') as f:
        f.write(texto)
    print('Relatório salvo em relatorio_regressao.txt')

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.scatter(x, y, label='Observado')
        idx_sort = np.argsort(x)
        plt.plot(x[idx_sort], y_pred_lin[idx_sort], '--', label=f'Linear (R2={r2_lin:.3f})')
        plt.plot(x[idx_sort], y_pred_poly[idx_sort], '-', label=f'Polinomial (R2={r2_poly:.3f})')
        if y_pred_exp is not None:
            plt.plot(x[idx_sort], y_pred_exp[idx_sort], '-.', label=f'Exponencial (R2={r2_exp:.3f})')
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        plt.legend()
        plt.title('Ajustes - Comparação')
        plt.grid(True)
        plt.savefig('ajustes_comparacao.png', dpi=300, bbox_inches='tight')
        print('Gráfico salvo em ajustes_comparacao.png')
    except Exception as e:
        print('Não foi possível salvar o gráfico:', e)

# ---------------------------- MAIN ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true', help='Gerar relatório estático e sair')
    parser.add_argument('--file', type=str, default='tabelinha.xlsx', help='Arquivo Excel com dados')
    args = parser.parse_args()

    df, colunas = carregar_dados(args.file)

    if args.report:
        gerar_report_cli(df, colunas)
    else:
        streamlit_app(df, colunas)

if __name__ == '__main__':
    main()
