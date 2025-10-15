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
        print(
            f"{col} - n={r['n']}, média={r['media']:.3f}, mediana={r['mediana']:.3f}, "
            f"moda={r['moda']:.3f}, variância={r['variancia']:.3f}, dp={r['desvio_padrao']:.3f}, "
            f"assimetria={r['assimetria']:.3f}, curtose={r['curtose']:.3f}, min={r['min']:.3f}, max={r['max']:.3f}",
            file=buf
        )
    print("\n== Modelos ajustados ==\n", file=buf)
    for nome, info in modelos_resumo.items():
        print(f"Modelo: {nome}", file=buf)
        for k, v in info.items():
            if isinstance(v, (int, float, str)):
                print(f"  {k}: {v}", file=buf)
        print("\n", file=buf)

    print("== Matriz de Covariância ==\n", file=buf)
    print(df[colunas].cov(), file=buf)

    print("\n== Matriz de Correlação ==\n", file=buf)
    print(df[colunas].corr(), file=buf)

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

    # ==========================
    # CALCULA MODELOS ANTES DAS ABAS
    # ==========================
    r2_lin = rmse_lin = r2_poly = rmse_poly = r2_exp = rmse_exp = r2_mv = rmse_mv = None
    model_mv = None

    if len(features) >= 1:
        xcol = features[0]
        ycol = target
        x = df[xcol].values
        y = df[ycol].values

        model_lin, y_pred_lin, r2_lin, rmse_lin = ajusta_regressao_linear_simples(x, y)
        model_poly, y_pred_poly, r2_poly, rmse_poly = ajusta_regressao_polinomial(x, y, grau=grau_poly)
        if np.all(y > 0):
            _, y_pred_exp, r2_exp, rmse_exp = ajusta_exponencial(x, y)

    if len(features) >= 2:
        X = df[features].values
        y = df[target].values
        model_mv, y_pred_mv, r2_mv, rmse_mv = ajusta_regressao_linear_multivariada(X, y)

    # ==========================
    # ABAS
    # ==========================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Estatísticas Descritivas",
        "📈 Gráficos por Variável",
        "🔗 Correlações & Modelos",
        "📋 Resumo Final"
    ])

    # TAB 1
    with tab1:
        st.subheader("📊 Estatísticas Descritivas e Gráficos")

        # ===== TABELA DE ESTATÍSTICAS =====
        df_stats = pd.DataFrame.from_dict(stats, orient='index')
        st.write(df_stats)

        # ===== GRÁFICOS ESTATÍSTICOS =====
        st.subheader("📈 Gráficos Estatísticos")

        # Gráfico de medidas centrais
        st.plotly_chart(
            px.bar(df_stats[['media', 'mediana', 'moda']],
                   barmode='group',
                   title="Média, Mediana e Moda por Variável"),
            use_container_width=True
        )

        # Gráfico de variância
        st.plotly_chart(
            px.bar(df_stats[['variancia']],
                   title="Variância por Variável"),
            use_container_width=True
        )

        # Gráfico de desvio padrão
        st.plotly_chart(
            px.bar(df_stats[['desvio_padrao']],
                   title="Desvio Padrão por Variável"),
            use_container_width=True
        )

        # Gráfico de mínimos e máximos
        st.plotly_chart(
            px.bar(df_stats[['min', 'max']],
                   barmode='group',
                   title="Valores Mínimos e Máximos por Variável"),
            use_container_width=True
        )

        # Gráfico de contagem (n)
        st.plotly_chart(
            px.bar(df_stats[['n']],
                   title="Número de Registros (N) por Variável"),
            use_container_width=True
        )

        # Matriz de correlação
        st.subheader("🔗 Matriz de Correlação entre Variáveis")
        corr = df[colunas].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                             title="Correlação entre Variáveis")
        st.plotly_chart(fig_corr, use_container_width=True)
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
            df_plot = pd.DataFrame({xcol: x, ycol: y, 'lin_pred': y_pred_lin, 'poly_pred': y_pred_poly})
            if r2_exp is not None:
                df_plot['exp_pred'] = y_pred_exp

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot[xcol], y=df_plot[ycol], mode='markers', name='Observado'))
            fig.add_trace(go.Scatter(x=df_plot[xcol], y=df_plot['lin_pred'], mode='lines', name=f'Linear (R²={r2_lin:.3f})'))
            fig.add_trace(go.Scatter(x=df_plot[xcol], y=df_plot['poly_pred'], mode='lines', name=f'Polinomial {grau_poly} (R²={r2_poly:.3f})'))
            if r2_exp is not None:
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

        if model_mv is not None:
            st.subheader("Regressão Linear Multivariada")
            st.write(pd.DataFrame({'feature': features, 'coef': model_mv.coef_}))
            st.write(f"Intercepto: {model_mv.intercept_:.3f}")
            st.write(f"R²: {r2_mv:.3f} | RMSE: {rmse_mv:.3f}")

    # TAB 4
    with tab4:
        st.subheader("Resumo Final e Exportação")

        st.write("📊 Estatísticas Descritivas")
        st.write(pd.DataFrame.from_dict(stats, orient='index'))

        modelos_resumo = {}
        if r2_lin is not None:
            modelos_resumo['Linear Univariada'] = {'R2': r2_lin, 'RMSE': rmse_lin}
        if r2_poly is not None:
            modelos_resumo[f'Polinomial grau {grau_poly}'] = {'R2': r2_poly, 'RMSE': rmse_poly}
        if r2_exp is not None:
            modelos_resumo['Exponencial'] = {'R2': r2_exp, 'RMSE': rmse_exp}
        if r2_mv is not None:
            modelos_resumo['Multivariada'] = {'R2': r2_mv, 'RMSE': rmse_mv}

        if modelos_resumo:
            st.write("📈 Métricas dos Modelos")
            st.write(pd.DataFrame(modelos_resumo).T)

        if model_mv is not None:
            st.write("📌 Coeficientes do Modelo Multivariado")
            st.write(pd.DataFrame({'feature': features, 'coef': model_mv.coef_}))
            st.write(f"Intercepto: {model_mv.intercept_:.3f}")

        st.write("🔗 Matriz de Covariância")
        st.dataframe(df[colunas].cov())

        st.write("🔗 Matriz de Correlação")
        st.dataframe(df[colunas].corr())

        texto = gerar_relatorio_texto(df, colunas, stats, modelos_resumo)
        st.subheader("📑 Prévia do Relatório")
        st.text(texto)

        st.download_button("📥 Download relatório .txt", texto, file_name="relatorio_regressao.txt")

# ---------------------------- MAIN ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true', help='Gerar relatório estático e sair')
    parser.add_argument('--file', type=str, default='tabelinha.xlsx', help='Arquivo Excel com dados')
    args = parser.parse_args()

    df, colunas = carregar_dados(args.file)

    if args.report:
        stats = resumo_estatistico(df, colunas)
        gerar_report_cli(df, colunas)
    else:
        streamlit_app(df, colunas)

if __name__ == '__main__':
    main()
