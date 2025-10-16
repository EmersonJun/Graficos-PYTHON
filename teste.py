import pandas as pd
import matplotlib.pyplot as plt

# === 1) LER A PLANILHA EXISTENTE ===
# Substitua pelo nome do seu arquivo
df = pd.read_excel("tabelinha2.xlsx")

# Veja como o pandas entendeu os dados
print(df.head())

# === 2) CALCULAR ESTATÍSTICAS ===
# Estatísticas para Ano Modelo
media = df["Ano Modelo"].mean()
mediana = df["Ano Modelo"].median()
moda = df["Ano Modelo"].mode()[0]  # Pega o primeiro valor da moda
variancia = df["Ano Modelo"].var()
desvio = df["Ano Modelo"].std()
assimetria = df["Ano Modelo"].skew()

# Estatísticas para Tamanho do motor
media2 = df["Tamanho do motor"].mean()
mediana2 = df["Tamanho do motor"].median()
moda2 = df["Tamanho do motor"].mode()[0]
variancia2 = df["Tamanho do motor"].var()
desvio2 = df["Tamanho do motor"].std()
assimetria2 = df["Tamanho do motor"].skew()

# Estatísticas para Preço médio brl
media3 = df["Preço médio brl"].mean()
mediana3 = df["Preço médio brl"].median()
moda3 = df["Preço médio brl"].mode()[0]
variancia3 = df["Preço médio brl"].var()
desvio3 = df["Preço médio brl"].std()
assimetria3 = df["Preço médio brl"].skew()

# Covariância entre as variáveis numéricas
cov_ano_motor = df["Ano Modelo"].cov(df["Tamanho do motor"])
cov_ano_preco = df["Ano Modelo"].cov(df["Preço médio brl"])
cov_motor_preco = df["Tamanho do motor"].cov(df["Preço médio brl"])


print("ANÁLISE ESTATÍSTICA COMPLETA")
print("="*50)

print("\n📊 ANO POR MODELO")
print(f"Média: {media:.3f}")
print(f"Mediana: {mediana:.3f}")
print(f"Moda: {moda:.3f}")
print(f"Variância: {variancia:.3f}")
print(f"Desvio padrão: {desvio:.3f}")
print(f"Assimetria: {assimetria:.3f}")

print("\n🔧 TAMANHO DO MOTOR")
print(f"Média: {media2:.3f}")
print(f"Mediana: {mediana2:.3f}")
print(f"Moda: {moda2:.3f}")
print(f"Variância: {variancia2:.3f}")
print(f"Desvio padrão: {desvio2:.3f}")
print(f"Assimetria: {assimetria2:.3f}")

print("\n💰 PREÇO EM BRL")
print(f"Média: {media3:.3f}")
print(f"Mediana: {mediana3:.3f}")
print(f"Moda: {moda3:.3f}")
print(f"Variância: {variancia3:.3f}")
print(f"Desvio padrão: {desvio3:.3f}")
print(f"Assimetria: {assimetria3:.3f}")

print("\n📈 COVARIÂNCIAS")
print(f"Ano Modelo vs Tamanho do Motor: {cov_ano_motor:.3f}")
print(f"Ano Modelo vs Preço BRL: {cov_ano_preco:.3f}")
print(f"Tamanho do Motor vs Preço BRL: {cov_motor_preco:.3f}")

# === 3) PLOTAR GRÁFICO ===
# plt.plot(df["Preço médio brl"], df["Ano Modelo "], marker="o")
# plt.title("Gráfico a partir do Excel")
# plt.xlabel("Preço médio brl")
# plt.ylabel("Ano Modelo ")
# plt.grid(True)
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# === FUNÇÕES AUXILIARES PARA ESTATÍSTICAS ===
def calcular_assimetria(dados):
    """Calcula assimetria (skewness) manualmente"""
    media = np.mean(dados)
    desvio = np.std(dados, ddof=1)
    n = len(dados)
    
    if desvio == 0:
        return 0
    
    skewness = np.sum(((dados - media) / desvio) ** 3) / n
    return skewness

def calcular_curtose(dados):
    """Calcula curtose (kurtosis) manualmente"""
    media = np.mean(dados)
    desvio = np.std(dados, ddof=1)
    n = len(dados)
    
    if desvio == 0:
        return 0
    
    kurtosis = np.sum(((dados - media) / desvio) ** 4) / n - 3
    return kurtosis

def teste_shapiro_simples(dados):
    """Teste simples de normalidade baseado em assimetria e curtose"""
    assimetria = calcular_assimetria(dados)
    curtose = calcular_curtose(dados)
    
    # Critérios simples para normalidade
    assimetria_ok = abs(assimetria) < 2
    curtose_ok = abs(curtose) < 7
    
    return assimetria_ok and curtose_ok

def densidade_estimada(dados, x_pontos):
    """Estima densidade usando método do histograma suavizado"""
    hist, bin_edges = np.histogram(dados, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Interpolação linear simples
    densidade = np.interp(x_pontos, bin_centers, hist)
    return np.maximum(densidade, 0)

def qq_plot_simples(dados, ax):
    """Q-Q plot simples contra distribuição normal"""
    dados_ord = np.sort(dados)
    n = len(dados_ord)
    
    # Quantis teóricos da normal padrão (aproximação)
    quantis_teoricos = []
    for i in range(1, n + 1):
        p = (i - 0.5) / n
        # Aproximação inversa da normal padrão
        if p < 0.5:
            quantil = -np.sqrt(-2 * np.log(p))
        else:
            quantil = np.sqrt(-2 * np.log(1 - p))
        quantis_teoricos.append(quantil)
    
    quantis_teoricos = np.array(quantis_teoricos)
    
    # Padronizar dados observados
    dados_padronizados = (dados_ord - np.mean(dados)) / np.std(dados, ddof=1)
    
    ax.scatter(quantis_teoricos, dados_padronizados, alpha=0.6, s=20)
    
    # Linha de referência
    min_val = min(np.min(quantis_teoricos), np.min(dados_padronizados))
    max_val = max(np.max(quantis_teoricos), np.max(dados_padronizados))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax.set_xlabel('Quantis Teóricos (Normal)')
    ax.set_ylabel('Quantis Observados')
    ax.grid(True, alpha=0.3)

# === 1) CARREGAR DADOS ===
try:
    df = pd.read_excel("tabelinha2.xlsx")
    print("✅ Arquivo carregado com sucesso!")
    print(f"📊 Dados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Verificar e limpar colunas
    colunas_necessarias = ["Ano Modelo", "Tamanho do motor", "Preço médio brl"]
    colunas_existentes = [col for col in colunas_necessarias if col in df.columns]
    
    if not colunas_existentes:
        print("❌ Nenhuma coluna encontrada. Colunas disponíveis:")
        print(df.columns.tolist())
        exit()
    
    # Limpar dados
    df_limpo = df.copy()
    for col in colunas_existentes:
        df_limpo[col] = pd.to_numeric(df_limpo[col], errors='coerce')
    df_limpo = df_limpo.dropna(subset=colunas_existentes)
    
    print(f"📋 Dados limpos: {df_limpo.shape[0]} linhas")
    print(f"🔍 Colunas analisadas: {colunas_existentes}")
    
except Exception as e:
    print(f"❌ Erro: {e}")
    print("\n💡 DICAS PARA RESOLVER:")
    print("1. Verifique se o arquivo 'tabelinha2.xlsx' está na mesma pasta")
    print("2. Instale o openpyxl: pip install openpyxl")
    print("3. Verifique se o arquivo não está aberto em outro programa")
    exit()

# === 2) FUNÇÃO PARA ANÁLISE ESTATÍSTICA COMPLETA ===
def analise_completa(serie, nome_var):
    """Análise estatística completa de uma variável"""
    
    print(f"\n{'='*60}")
    print(f"📊 ANÁLISE COMPLETA: {nome_var.upper()}")
    print(f"{'='*60}")
    
    # === MEDIDAS DE TENDÊNCIA CENTRAL ===
    media = serie.mean()
    mediana = serie.median()
    try:
        moda = serie.mode()[0] if not serie.mode().empty else np.nan
    except:
        moda = np.nan
    
    print(f"\n🎯 MEDIDAS DE TENDÊNCIA CENTRAL:")
    print(f"   Média: {media:.3f}")
    print(f"   Mediana: {mediana:.3f}")
    print(f"   Moda: {moda:.3f}")
    
    # === MEDIDAS DE DISPERSÃO ===
    variancia = serie.var()
    desvio_padrao = serie.std()
    amplitude = serie.max() - serie.min()
    coef_variacao = (desvio_padrao / media) * 100 if media != 0 else np.nan
    desvio_medio = np.mean(np.abs(serie - media))
    
    print(f"\n📏 MEDIDAS DE DISPERSÃO:")
    print(f"   Variância: {variancia:.3f}")
    print(f"   Desvio Padrão: {desvio_padrao:.3f}")
    print(f"   Amplitude: {amplitude:.3f}")
    print(f"   Coeficiente de Variação: {coef_variacao:.3f}%")
    print(f"   Desvio Médio: {desvio_medio:.3f}")
    
    # === PERCENTIS ===
    percentis = [5, 10, 25, 50, 75, 90, 95]
    print(f"\n📈 PERCENTIS:")
    for p in percentis:
        valor = np.percentile(serie, p)
        print(f"   P{p}: {valor:.3f}")
    
    # === MEDIDAS DE FORMA ===
    assimetria = calcular_assimetria(serie.values)
    curtose = calcular_curtose(serie.values)
    
    print(f"\n📐 MEDIDAS DE FORMA:")
    print(f"   Assimetria (Skewness): {assimetria:.3f}")
    
    if abs(assimetria) < 0.5:
        tipo_assim = "Aproximadamente simétrica"
    elif assimetria > 0:
        tipo_assim = "Assimétrica positiva (cauda à direita)"
    else:
        tipo_assim = "Assimétrica negativa (cauda à esquerda)"
    print(f"   → {tipo_assim}")
    
    print(f"   Curtose (Kurtosis): {curtose:.3f}")
    if curtose > 0:
        tipo_curt = "Leptocúrtica (mais pontiaguda que a normal)"
    elif curtose < 0:
        tipo_curt = "Platicúrtica (mais achatada que a normal)"
    else:
        tipo_curt = "Mesocúrtica (similar à distribuição normal)"
    print(f"   → {tipo_curt}")
    
    # === OUTROS INDICADORES ===
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    outliers = serie[(serie < limite_inf) | (serie > limite_sup)]
    
    print(f"\n🔍 ANÁLISE DE OUTLIERS:")
    print(f"   Q1 (25%): {q1:.3f}")
    print(f"   Q3 (75%): {q3:.3f}")
    print(f"   IQR: {iqr:.3f}")
    print(f"   Limite Inferior: {limite_inf:.3f}")
    print(f"   Limite Superior: {limite_sup:.3f}")
    print(f"   Outliers encontrados: {len(outliers)}")
    if len(outliers) > 0:
        print(f"   Valores outliers: {outliers.values[:5]}")  # Mostrar até 5
    
    return {
        'media': media, 'mediana': mediana, 'moda': moda,
        'variancia': variancia, 'desvio_padrao': desvio_padrao,
        'assimetria': assimetria, 'curtose': curtose,
        'q1': q1, 'q3': q3, 'iqr': iqr, 'outliers': len(outliers)
    }

# === 3) ANALISAR CADA VARIÁVEL ===
resultados = {}
nomes_variaveis = {
    "Ano Modelo": "Ano do Modelo",
    "Tamanho do motor": "Tamanho do Motor",
    "Preço médio brl": "Preço em BRL"
}

for col in colunas_existentes:
    nome = nomes_variaveis.get(col, col)
    resultados[col] = analise_completa(df_limpo[col], nome)

# === 4) CRIAR GRÁFICOS AVANÇADOS ===
print(f"\n{'='*60}")
print("📊 GERANDO GRÁFICOS AVANÇADOS")
print(f"{'='*60}")

# Definir cores para cada variável
cores = ['steelblue', 'darkgreen', 'darkorange', 'purple', 'brown', 'pink']

# Configurar subplot principal
fig = plt.figure(figsize=(20, 6*len(colunas_existentes)))
fig.suptitle('ANÁLISE ESTATÍSTICA COMPLETA - GRÁFICOS AVANÇADOS', 
             fontsize=16, fontweight='bold')

for idx, col in enumerate(colunas_existentes):
    dados = df_limpo[col]
    nome = nomes_variaveis.get(col, col)
    cor = cores[idx % len(cores)]
    
    # === GRÁFICO 1: HISTOGRAMA COM DENSIDADE ===
    plt.subplot(len(colunas_existentes), 4, idx*4 + 1)
    
    # Histograma
    n, bins, patches = plt.hist(dados, bins=25, density=True, alpha=0.7, 
                               color=cor, edgecolor='black', linewidth=0.5)
    
    # Curva de densidade estimada
    x_smooth = np.linspace(dados.min(), dados.max(), 200)
    y_smooth = densidade_estimada(dados.values, x_smooth)
    plt.plot(x_smooth, y_smooth, 'red', linewidth=2, label='Densidade estimada')
    
    # Adicionar linhas estatísticas
    plt.axvline(dados.mean(), color='green', linestyle='--', linewidth=2, 
                label=f'Média: {dados.mean():.2f}')
    plt.axvline(dados.median(), color='orange', linestyle='--', linewidth=2, 
                label=f'Mediana: {dados.median():.2f}')
    
    plt.title(f'Histograma + Densidade - {nome}', fontsize=11, fontweight='bold')
    plt.xlabel(nome)
    plt.ylabel('Densidade')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # === GRÁFICO 2: BOX PLOT ===
    plt.subplot(len(colunas_existentes), 4, idx*4 + 2)
    
    box_plot = plt.boxplot(dados, patch_artist=True, labels=[nome])
    box_plot['boxes'][0].set_facecolor(cor)
    box_plot['boxes'][0].set_alpha(0.7)
    
    # Adicionar estatísticas
    q1, q2, q3 = dados.quantile(0.25), dados.median(), dados.quantile(0.75)
    stats_text = f'Q1: {q1:.2f}\nMediana: {q2:.2f}\nQ3: {q3:.2f}\nIQR: {q3-q1:.2f}'
    plt.text(1.15, q2, stats_text, fontsize=8, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.title(f'Box Plot - {nome}', fontsize=11, fontweight='bold')
    plt.ylabel(nome)
    plt.grid(True, alpha=0.3)
    
    # === GRÁFICO 3: Q-Q PLOT (NORMALIDADE) ===
    ax3 = plt.subplot(len(colunas_existentes), 4, idx*4 + 3)
    qq_plot_simples(dados.values, ax3)
    plt.title(f'Q-Q Plot (Normalidade) - {nome}', fontsize=11, fontweight='bold')
    
    # === GRÁFICO 4: PERCENTIS ===
    plt.subplot(len(colunas_existentes), 4, idx*4 + 4)
    percentis = range(1, 100, 2)
    valores_percentis = [np.percentile(dados, p) for p in percentis]
    
    plt.plot(percentis, valores_percentis, 'o-', color=cor, linewidth=2, markersize=3)
    
    # Destacar quartis
    for q_val, q_name, q_color in [(25, 'Q1', 'red'), (50, 'Q2', 'green'), (75, 'Q3', 'blue')]:
        plt.axvline(q_val, color=q_color, linestyle='--', alpha=0.7, linewidth=1)
        plt.axhline(np.percentile(dados, q_val), color=q_color, linestyle='--', alpha=0.7, linewidth=1)
        plt.plot(q_val, np.percentile(dados, q_val), 'o', color=q_color, markersize=8)
        plt.text(q_val+1, np.percentile(dados, q_val), 
                f'{q_name}\n{np.percentile(dados, q_val):.1f}', 
                fontsize=7, color=q_color, fontweight='bold')
    
    plt.title(f'Gráfico de Percentis - {nome}', fontsize=11, fontweight='bold')
    plt.xlabel('Percentil (%)')
    plt.ylabel('Valor')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(hspace=0.4, wspace=0.3)

plt.show()

# === 5) GRÁFICOS DE CORRELAÇÃO E ANÁLISE COMPARATIVA ===
if len(colunas_existentes) >= 2:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ANÁLISE COMPARATIVA E CORRELAÇÕES', fontsize=16, fontweight='bold')
    
    # === MATRIZ DE CORRELAÇÃO ===
    ax1 = axes[0, 0]
    corr_matrix = df_limpo[colunas_existentes].corr()
    
    # Criar mapa de calor manual
    im = ax1.imshow(corr_matrix, cmap='RdYlBu', vmin=-1, vmax=1, aspect='auto')
    
    # Configurar ticks e labels
    ax1.set_xticks(range(len(colunas_existentes)))
    ax1.set_yticks(range(len(colunas_existentes)))
    ax1.set_xticklabels([nomes_variaveis.get(col, col) for col in colunas_existentes], 
                       rotation=45, ha='right')
    ax1.set_yticklabels([nomes_variaveis.get(col, col) for col in colunas_existentes])
    
    # Adicionar valores na matriz
    for i in range(len(colunas_existentes)):
        for j in range(len(colunas_existentes)):
            valor = corr_matrix.iloc[i, j]
            cor_texto = 'white' if abs(valor) > 0.5 else 'black'
            ax1.text(j, i, f'{valor:.3f}', ha='center', va='center', 
                    fontweight='bold', color=cor_texto, fontsize=10)
    
    ax1.set_title('Matriz de Correlação', fontsize=14, fontweight='bold')
    
    # Colorbar manual
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Coeficiente de Correlação', rotation=270, labelpad=15)
    
    # === SCATTER PLOT COM TENDÊNCIA ===
    if len(colunas_existentes) >= 2:
        ax2 = axes[0, 1]
        x = df_limpo[colunas_existentes[0]]
        y = df_limpo[colunas_existentes[1]]
        
        ax2.scatter(x, y, alpha=0.6, s=50, color='steelblue', edgecolors='navy', linewidth=0.5)
        
        # Linha de tendência
        coef = np.polyfit(x, y, 1)
        linha_tendencia = np.poly1d(coef)
        x_linha = np.linspace(x.min(), x.max(), 100)
        ax2.plot(x_linha, linha_tendencia(x_linha), "r--", linewidth=2, 
                label=f'y = {coef[0]:.3f}x + {coef[1]:.3f}')
        
        # Estatísticas
        corr = x.corr(y)
        ax2.text(0.05, 0.95, f'Correlação: {corr:.3f}\nR²: {corr**2:.3f}', 
                transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                fontsize=11, fontweight='bold', verticalalignment='top')
        
        ax2.set_xlabel(nomes_variaveis.get(colunas_existentes[0], colunas_existentes[0]))
        ax2.set_ylabel(nomes_variaveis.get(colunas_existentes[1], colunas_existentes[1]))
        ax2.set_title('Gráfico de Dispersão com Tendência', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
    
    # === GRÁFICO DE ASSIMETRIA E CURTOSE ===
    ax3 = axes[1, 0]
    nomes = [nomes_variaveis.get(col, col) for col in colunas_existentes]
    assimetrias = [resultados[col]['assimetria'] for col in colunas_existentes]
    curtoses = [resultados[col]['curtose'] for col in colunas_existentes]
    
    x_pos = np.arange(len(nomes))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, assimetrias, width, label='Assimetria', 
                   alpha=0.8, color='skyblue', edgecolor='navy')
    bars2 = ax3.bar(x_pos + width/2, curtoses, width, label='Curtose', 
                   alpha=0.8, color='lightcoral', edgecolor='darkred')
    
    ax3.set_xlabel('Variáveis')
    ax3.set_ylabel('Valores')
    ax3.set_title('Assimetria e Curtose por Variável', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(nomes, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Adicionar valores nas barras
    for bar, val in zip(bars1, assimetrias):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., 
                height + 0.05 if height >= 0 else height - 0.1,
                f'{val:.2f}', ha='center', 
                va='bottom' if height >= 0 else 'top', fontsize=9, fontweight='bold')
    
    for bar, val in zip(bars2, curtoses):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., 
                height + 0.05 if height >= 0 else height - 0.1,
                f'{val:.2f}', ha='center', 
                va='bottom' if height >= 0 else 'top', fontsize=9, fontweight='bold')
    
    # === COMPARAÇÃO DE DISTRIBUIÇÕES NORMALIZADAS ===
    ax4 = axes[1, 1]
    
    for i, col in enumerate(colunas_existentes):
        dados_norm = (df_limpo[col] - df_limpo[col].mean()) / df_limpo[col].std()
        
        # Histograma normalizado
        ax4.hist(dados_norm, bins=20, alpha=0.5, label=nomes_variaveis.get(col, col), 
                density=True, color=cores[i], edgecolor='black', linewidth=0.5)
    
    # Distribuição normal teórica
    x_norm = np.linspace(-3.5, 3.5, 100)
    y_norm = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_norm**2)
    ax4.plot(x_norm, y_norm, 'k--', linewidth=3, label='Normal Padrão', alpha=0.8)
    
    ax4.set_xlabel('Valores Padronizados (Z-score)')
    ax4.set_ylabel('Densidade')
    ax4.set_title('Comparação com Distribuição Normal', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-3.5, 3.5)
    
    plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()


# === 6) TABELA RESUMO FINAL ===
print(f"\n{'='*90}")
print("📋 TABELA RESUMO FINAL - TODAS AS ESTATÍSTICAS")
print(f"{'='*90}")

resumo_final = []
for col in colunas_existentes:
    nome = nomes_variaveis.get(col, col)
    stats_var = resultados[col]
    
    resumo_final.append({
        'Variável': nome,
        'Média': f"{stats_var['media']:.3f}",
        'Mediana': f"{stats_var['mediana']:.3f}",
        'Desvio Padrão': f"{stats_var['desvio_padrao']:.3f}",
        'Assimetria': f"{stats_var['assimetria']:.3f}",
        'Curtose': f"{stats_var['curtose']:.3f}",
        'Q1': f"{stats_var['q1']:.3f}",
        'Q3': f"{stats_var['q3']:.3f}",
        'Outliers': stats_var['outliers']
    })

tabela_final = pd.DataFrame(resumo_final)
print(tabela_final.to_string(index=False))

# === 7) TESTES DE NORMALIDADE SIMPLES ===
print(f"\n🔬 ANÁLISE DE NORMALIDADE (MÉTODO SIMPLIFICADO):")
print("-" * 60)

for col in colunas_existentes:
    nome = nomes_variaveis.get(col, col)
    dados = df_limpo[col].values
    
    # Critérios simples de normalidade
    assimetria = calcular_assimetria(dados)
    curtose = calcular_curtose(dados)
    
    # Teste baseado em regras práticas
    assimetria_normal = abs(assimetria) < 2  # Critério mais relaxado
    curtose_normal = abs(curtose) < 7        # Critério mais relaxado
    
    print(f"\n📊 {nome}:")
    print(f"   Assimetria: {assimetria:.4f} ({'✅ Aceitável' if assimetria_normal else '❌ Problemática'})")
    print(f"   Curtose: {curtose:.4f} ({'✅ Aceitável' if curtose_normal else '❌ Problemática'})")
    
    if assimetria_normal and curtose_normal:
        conclusao = "✅ Possivelmente próxima da distribuição normal"
    elif assimetria_normal or curtose_normal:
        conclusao = "⚠️  Parcialmente compatível com normalidade"
    else:
        conclusao = "❌ Provavelmente não segue distribuição normal"
    
    print(f"   → {conclusao}")

# === 8) RELATÓRIO DE COVARIÂNCIAS ===
if len(colunas_existentes) >= 2:
    print(f"\n📈 MATRIZ DE COVARIÂNCIAS:")
    print("-" * 60)
    
    cov_matrix = df_limpo[colunas_existentes].cov()
    print(cov_matrix.round(3))
    
    print(f"\n🔗 COVARIÂNCIAS E CORRELAÇÕES DETALHADAS:")
    for i, col1 in enumerate(colunas_existentes):
        for j, col2 in enumerate(colunas_existentes):
            if i < j:
                cov_val = df_limpo[col1].cov(df_limpo[col2])
                corr_val = df_limpo[col1].corr(df_limpo[col2])
                nome1 = nomes_variaveis.get(col1, col1)
                nome2 = nomes_variaveis.get(col2, col2)
                
                print(f"\n   📊 {nome1} ↔ {nome2}:")
                print(f"      • Covariância: {cov_val:.3f}")
                print(f"      • Correlação: {corr_val:.3f}")
                
                if abs(corr_val) > 0.7:
                    forca = "Forte"
                elif abs(corr_val) > 0.3:
                    forca = "Moderada"
                else:
                    forca = "Fraca"
                
                direcao = "positiva" if corr_val > 0 else "negativa"
                print(f"      • Interpretação: Correlação {forca.lower()} {direcao}")
                
                # Interpretação prática
                if abs(corr_val) > 0.7:
                    print(f"      • 💡 As variáveis estão fortemente relacionadas")
                elif abs(corr_val) > 0.3:
                    print(f"      • 💡 Existe relação moderada entre as variáveis")
                else:
                    print(f"      • 💡 As variáveis têm pouca relação linear")

print(f"\n{'='*80}")
print("✅ ANÁLISE COMPLETA FINALIZADA!")
print(f"📊 {len(colunas_existentes)} variáveis analisadas")
print(f"📈 Gráficos gerados: Histogramas, Box plots, Q-Q plots, Percentis, Correlações")
print(f"📋 Estatísticas calculadas: Tendência central, Dispersão, Forma, Outliers")
print(f"🔬 Análises realizadas: Normalidade, Correlações, Covariâncias")
print(f"📊 Total de registros processados: {len(df_limpo)}")
print(f"\n💡 BIBLIOTECAS USADAS: pandas, matplotlib, numpy (nativas do Python)")
print(f"{'='*80}")



