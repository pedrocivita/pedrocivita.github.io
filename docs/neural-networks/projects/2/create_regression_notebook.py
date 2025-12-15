"""
Script para criar o notebook de regressão completo.
Execute: python create_regression_notebook.py
"""

import json

# Estrutura do notebook
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Helper function to create cells
def add_markdown_cell(content):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split("\n")
    })

def add_code_cell(content):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.split("\n")
    })

# ==============================================================================
# CÉLULAS DO NOTEBOOK
# ==============================================================================

# Título e Introdução
add_markdown_cell("""# Projeto de Regressão: Previsão de Demanda de Bicicletas Compartilhadas

**Disciplina:** Redes Neurais e Deep Learning
**Projeto:** Regression Task com Multi-Layer Perceptron (MLP)
**Deadline:** 19.out.2025

---

## Objetivo

Este projeto implementa uma rede neural MLP do zero para resolver um problema de regressão real: **prever a demanda de bicicletas compartilhadas** com base em condições climáticas, temporais e sazonais. O objetivo é aprofundar o conhecimento sobre redes neurais através de implementação completa, desde preparação de dados até avaliação de resultados.""")

# Seção 1: Seleção do Dataset
add_markdown_cell("""## 1. Seleção do Dataset

### Dataset Escolhido: **Bike Sharing Demand Dataset**

- **Fonte:** [UCI Machine Learning Repository - Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
- **Também disponível em:** [Kaggle - Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)
- **Tamanho:** 17,379 amostras (dia a dia durante 2 anos: 2011-2012)
- **Features:** 16 variáveis (temporais, climáticas, sazonais)
- **Target:** Contagem total de aluguéis de bicicletas (variável contínua)

### Por que este dataset?

1. **Relevância Real:** Sistemas de bike-sharing são amplamente usados em cidades inteligentes. Prever demanda ajuda em:
   - Rebalanceamento de bicicletas entre estações
   - Planejamento de manutenção
   - Otimização de recursos operacionais

2. **Complexidade Adequada:**
   - Múltiplas features com diferentes tipos (numéricas, categóricas)
   - Relações não-lineares entre clima e demanda
   - Sazonalidade e tendências temporais

3. **Dataset Não-Trivial:**
   - Evita datasets clássicos como Boston Housing ou California Housing
   - Mais de 17k amostras (>1,000 requerido)
   - Mais de 5 features relevantes

4. **Possibilidade de Competição:**
   - Dataset usado em competição do Kaggle (oportunidade de bonus)
   - Benchmark público disponível""")

# Seção 2: Importação de Bibliotecas
add_markdown_cell("""## 2. Importação de Bibliotecas""")

add_code_cell("""# Manipulação de dados
import numpy as np
import pandas as pd

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Utilitários
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuração de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline

# Seed para reprodutibilidade
np.random.seed(42)

print("Bibliotecas importadas com sucesso!")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")""")

# Seção 3: Carregamento dos Dados
add_markdown_cell("""## 3. Carregamento e Exploração Inicial dos Dados

### Download do Dataset

**IMPORTANTE:** Antes de executar o notebook, baixe o dataset:

1. Acesse: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
2. Baixe o arquivo `Bike-Sharing-Dataset.zip`
3. Extraia e coloque o arquivo `hour.csv` neste diretório

**Alternativa:** Use o dataset do Kaggle (requer Kaggle API):
```bash
kaggle competitions download -c bike-sharing-demand
```""")

add_code_cell("""# Carregamento do dataset
# Nota: Usando dados por hora para maior volume de amostras

try:
    df = pd.read_csv('hour.csv')
    print("✓ Dataset carregado com sucesso!")
except FileNotFoundError:
    print("❌ Arquivo 'hour.csv' não encontrado!")
    print("Por favor, baixe o dataset de: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset")
    raise

# Visualização inicial
print(f"\\nDimensões do dataset: {df.shape}")
print(f"Amostras: {df.shape[0]:,} | Features: {df.shape[1]}")
print("\\n" + "="*80)
df.head(10)""")

# Seção 4: Explicação do Dataset
add_markdown_cell("""## 4. Explicação Detalhada do Dataset

### Contexto do Problema

O dataset contém registros horários de um sistema de bike-sharing em Washington D.C. durante 2011 e 2012. Sistemas de compartilhamento de bicicletas automatizam o processo de aluguel/devolução através de quiosques distribuídos pela cidade.

### Descrição das Features

#### Variáveis Temporais:
- **instant**: Índice do registro (não será usado como feature)
- **dteday**: Data (formato YYYY-MM-DD)
- **season**: Estação do ano (1: primavera, 2: verão, 3: outono, 4: inverno)
- **yr**: Ano (0: 2011, 1: 2012)
- **mnth**: Mês (1 a 12)
- **hr**: Hora do dia (0 a 23)
- **holiday**: Dia é feriado (0: não, 1: sim)
- **weekday**: Dia da semana (0: domingo ... 6: sábado)
- **workingday**: Dia útil (0: fim de semana/feriado, 1: dia útil)

#### Variáveis Climáticas:
- **weathersit**: Condição climática
  - 1: Claro, poucas nuvens, parcialmente nublado
  - 2: Neblina + nublado, neblina + nuvens quebradas
  - 3: Neve leve, chuva leve + trovoada
  - 4: Chuva forte + granizo + trovoada + neblina, neve + neblina
- **temp**: Temperatura normalizada em Celsius (dividida por 41°C máx)
- **atemp**: Sensação térmica normalizada (dividida por 50°C máx)
- **hum**: Umidade relativa normalizada (dividida por 100)
- **windspeed**: Velocidade do vento normalizada (dividida por 67 km/h)

#### Variáveis Target:
- **casual**: Contagem de usuários casuais (não-registrados)
- **registered**: Contagem de usuários registrados
- **cnt**: **VARIÁVEL TARGET** - Contagem total (casual + registered)

### Conhecimento de Domínio

**Fatores que influenciam demanda de bikes:**

1. **Hora do dia**: Picos em horários de commute (8h-9h, 17h-18h)
2. **Clima**: Dias ensolarados têm mais demanda que chuvosos
3. **Temperatura**: Temperaturas moderadas favorecem uso de bicicletas
4. **Dia da semana**: Padrão diferente entre dias úteis e fins de semana
5. **Sazonalidade**: Mais uso em primavera/verão vs. inverno""")

add_code_cell("""# Informações detalhadas sobre o dataset
print("INFORMAÇÕES DO DATASET")
print("="*80)
df.info()

print("\\n" + "="*80)
print("ESTATÍSTICAS DESCRITIVAS")
print("="*80)
df.describe().round(2)""")

# Verificação de problemas nos dados
add_markdown_cell("""### Identificação de Potenciais Problemas""")

add_code_cell("""# Verificação de valores ausentes
print("VALORES AUSENTES")
print("="*80)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Coluna': missing.index,
    'Valores Ausentes': missing.values,
    'Percentual (%)': missing_pct.values
})
missing_df = missing_df[missing_df['Valores Ausentes'] > 0].sort_values('Valores Ausentes', ascending=False)

if len(missing_df) == 0:
    print("✓ Nenhum valor ausente encontrado!")
else:
    print(missing_df.to_string(index=False))

# Verificação de duplicatas
print("\\n" + "="*80)
print("DUPLICATAS")
print("="*80)
duplicates = df.duplicated().sum()
print(f"Linhas duplicadas: {duplicates}")
if duplicates == 0:
    print("✓ Nenhuma duplicata encontrada!")

# Verificação de outliers na variável target
print("\\n" + "="*80)
print("ANÁLISE DE OUTLIERS NA VARIÁVEL TARGET (cnt)")
print("="*80)
Q1 = df['cnt'].quantile(0.25)
Q3 = df['cnt'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['cnt'] < lower_bound) | (df['cnt'] > upper_bound)]
print(f"Q1 (25%): {Q1:.0f}")
print(f"Q3 (75%): {Q3:.0f}")
print(f"IQR: {IQR:.0f}")
print(f"Limite inferior: {lower_bound:.0f}")
print(f"Limite superior: {upper_bound:.0f}")
print(f"Outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
print(f"\\nNota: Outliers podem representar eventos especiais (ex: feriados, eventos climáticos extremos)")
print(f"Decisão: Manter outliers pois representam variação real do sistema")""")

# Visualizações exploratórias
add_markdown_cell("""### Visualizações Exploratórias""")

add_code_cell("""# Distribuição da variável target
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histograma
axes[0].hist(df['cnt'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Contagem de Aluguéis', fontsize=12)
axes[0].set_ylabel('Frequência', fontsize=12)
axes[0].set_title('Distribuição da Variável Target (cnt)', fontsize=14, fontweight='bold')
axes[0].axvline(df['cnt'].mean(), color='red', linestyle='--', label=f'Média: {df["cnt"].mean():.0f}')
axes[0].axvline(df['cnt'].median(), color='green', linestyle='--', label=f'Mediana: {df["cnt"].median():.0f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Boxplot
axes[1].boxplot(df['cnt'], vert=True)
axes[1].set_ylabel('Contagem de Aluguéis', fontsize=12)
axes[1].set_title('Boxplot da Variável Target', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Assimetria (Skewness): {df['cnt'].skew():.2f}")
print(f"Curtose (Kurtosis): {df['cnt'].kurtosis():.2f}")
print("\\nInterpretação: Distribuição positivamente assimétrica (cauda à direita)")
print("Muitas horas com baixa demanda, poucas horas com alta demanda")""")

add_code_cell("""# Correlação entre features numéricas
numeric_cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação - Features Numéricas', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("CORRELAÇÕES COM A VARIÁVEL TARGET (cnt):")
print("="*80)
target_corr = correlation_matrix['cnt'].sort_values(ascending=False)
for feature, corr in target_corr.items():
    if feature != 'cnt':
        print(f"{feature:15s}: {corr:+.3f}")""")

# Salvar o notebook
output_path = r"c:\Users\pedro\OneDrive\Documents\Insper\9_semestre\redes_neurais_e_deep_learning\pedrocivita.github.io\site\projects\2\main\regression.ipynb"

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✓ Notebook criado com sucesso em: {output_path}")
print(f"✓ Total de células: {len(notebook['cells'])}")
