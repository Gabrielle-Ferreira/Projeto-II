"""
Projeto BNDES - Controle de Acurácia em 4 Dimensões (VERSÃO FINAL)
"""

# ============================================================
# BIBLIOTECAS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import pearsonr, spearmanr, shapiro
import warnings
import os
import sys

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
sns.set_style("whitegrid")

print("="*80)
print("PROJETO BNDES - SISTEMA DE ACURÁCIA EM 4 DIMENSÕES")
print("="*80)

# ============================================================
# CAMINHO DO ARQUIVO
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_CSV = os.path.join(BASE_DIR, "..", "Data", "bndes_dre.csv")

print("\nDiretório atual:", os.getcwd())
print("Caminho do CSV:", CAMINHO_CSV)
print("Arquivo existe?", os.path.exists(CAMINHO_CSV))

# ============================================================
# 1. ACURÁCIA DOS DADOS (VERSÃO FINAL ROBUSTA)
# ============================================================

def carregar_e_limpar_dados(caminho_arquivo):
    print(f"\nCarregando arquivo: {caminho_arquivo}")
    
    # Preview do arquivo (agora com UTF-8)
    with open(caminho_arquivo, 'r', encoding='utf-8-sig') as f:
        print("\n--- Preview do arquivo ---")
        for _ in range(5):
            print(next(f).strip())

    df = pd.read_csv(
        caminho_arquivo,
        encoding='utf-8-sig',   
        sep=';',
        skiprows=1,
        engine='python',
        on_bad_lines='skip'
    )
    
    # Remove colunas totalmente vazias
    df = df.dropna(axis=1, how='all')
    
    # Remove colunas "Unnamed"
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Corrige nomes das colunas
    df.columns = [str(col).strip() for col in df.columns]
    
    # Remove linhas vazias
    df = df.dropna(how='all')

    for col in df.columns[1:]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"\n✓ Arquivo carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    print("\nColunas finais:")
    print(df.columns.tolist())
    
    print("\nPreview limpo:")
    print(df.head())
    
    return df


def validar_integridade(df):
    print("\n--- Validação de Integridade ---")
    
    print(f"✓ Valores nulos: {df.isnull().sum().sum()}")
    print(f"✓ Linhas duplicadas: {df.duplicated().sum()}")
    
    return df


def detectar_outliers(df):
    print("\n--- Detecção de Outliers ---")
    
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    
    for col in colunas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        
        if len(outliers) > 0:
            print(f"{col}: {len(outliers)} outliers")


# ============================================================
# 2. ACURÁCIA DA ANÁLISE
# ============================================================

def calcular_correlacoes(df):
    print("\n--- Correlações ---")
    
    df_num = df.select_dtypes(include=[np.number]).dropna()
    
    if df_num.shape[1] < 2:
        print("Dados insuficientes")
        return
    
    corr_pearson = df_num.corr(method='pearson')
    corr_spearman = df_num.corr(method='spearman')
    
    print("\n✓ Correlação de Pearson:")
    print(corr_pearson)
    
    print("\n✓ Correlação de Spearman:")
    print(corr_spearman)
    
    return corr_spearman


# ============================================================
# 3. REPRODUTIBILIDADE
# ============================================================

def gerar_relatorio_reprodutibilidade():
    print("\n--- Ambiente ---")
    
    print(f"Python: {sys.version}")
    print(f"pandas: {pd.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"scipy: {scipy.__version__}")
    print(f"matplotlib: {plt.matplotlib.__version__}")
    print(f"seaborn: {sns.__version__}")
    
    np.random.seed(42)


# ============================================================
# EXECUÇÃO
# ============================================================

try:
    df = carregar_e_limpar_dados(CAMINHO_CSV)
    
    validar_integridade(df)
    
    detectar_outliers(df)
    
    calcular_correlacoes(df)
    
    gerar_relatorio_reprodutibilidade()
    
    gerar_matriz_correlacao(df)
    
    print("\nANÁLISE CONCLUÍDA!")
    
except FileNotFoundError:
    print(f"\n Arquivo não encontrado: {CAMINHO_CSV}")
    
except Exception as e:
    print(f"\n ERRO: {e}")