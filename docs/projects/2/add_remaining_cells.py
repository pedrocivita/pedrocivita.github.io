"""
Script para adicionar células restantes ao notebook de regressão.
Execute APÓS create_regression_notebook.py
"""

import json

# Carregar notebook existente
notebook_path = r"c:\Users\pedro\OneDrive\Documents\Insper\9_semestre\redes_neurais_e_deep_learning\pedrocivita.github.io\site\projects\2\main\regression.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Helper functions
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

print(f"Células existentes: {len(notebook['cells'])}")
print("Adicionando células restantes...")

# ==============================================================================
# CÉLULAS ADICIONAIS
# ==============================================================================

# Análise temporal e sazonal
add_code_cell("""# Análise temporal e sazonal
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Demanda por hora do dia
hourly_avg = df.groupby('hr')['cnt'].mean()
axes[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=6)
axes[0, 0].set_xlabel('Hora do Dia', fontsize=11)
axes[0, 0].set_ylabel('Média de Aluguéis', fontsize=11)
axes[0, 0].set_title('Demanda Média por Hora do Dia', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(range(0, 24, 2))

# 2. Demanda por estação
season_names = {1: 'Primavera', 2: 'Verão', 3: 'Outono', 4: 'Inverno'}
season_avg = df.groupby('season')['cnt'].mean()
axes[0, 1].bar([season_names[i] for i in season_avg.index], season_avg.values,
               color=['lightgreen', 'yellow', 'orange', 'lightblue'], edgecolor='black')
axes[0, 1].set_ylabel('Média de Aluguéis', fontsize=11)
axes[0, 1].set_title('Demanda Média por Estação', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Demanda por condição climática
weather_names = {1: 'Claro', 2: 'Nublado', 3: 'Chuva Leve', 4: 'Chuva Forte'}
weather_avg = df.groupby('weathersit')['cnt'].mean()
axes[1, 0].bar([weather_names.get(i, str(i)) for i in weather_avg.index], weather_avg.values,
               color=['gold', 'lightgray', 'lightblue', 'darkblue'], edgecolor='black')
axes[1, 0].set_ylabel('Média de Aluguéis', fontsize=11)
axes[1, 0].set_title('Demanda Média por Condição Climática', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].tick_params(axis='x', rotation=15)

# 4. Temperatura vs Demanda
axes[1, 1].scatter(df['temp'], df['cnt'], alpha=0.3, s=10)
axes[1, 1].set_xlabel('Temperatura Normalizada', fontsize=11)
axes[1, 1].set_ylabel('Contagem de Aluguéis', fontsize=11)
axes[1, 1].set_title('Relação entre Temperatura e Demanda', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Linha de tendência
z = np.polyfit(df['temp'], df['cnt'], 2)
p = np.poly1d(z)
temp_range = np.linspace(df['temp'].min(), df['temp'].max(), 100)
axes[1, 1].plot(temp_range, p(temp_range), "r-", linewidth=2, label='Tendência (polinomial)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()""")

# Seção 5: Limpeza e Normalização
add_markdown_cell("""## 5. Limpeza e Normalização dos Dados

### Estratégia de Pré-processamento:

1. **Remoção de colunas irrelevantes**:
   - `instant`: índice sequencial sem valor preditivo
   - `dteday`: data já representada por outras features temporais
   - `casual` e `registered`: componentes do target (evitar data leakage)

2. **Tratamento de valores ausentes**:
   - Dataset não possui valores ausentes (verificado anteriormente)

3. **Tratamento de outliers**:
   - Mantidos pois representam variação real do sistema
   - Outliers podem ser eventos especiais (feriados, clima extremo)

4. **Encoding de variáveis categóricas**:
   - Variáveis ordinais (`season`, `weathersit`): manter valores numéricos (já possuem ordem natural)
   - Variáveis cíclicas (`hr`, `mnth`, `weekday`): aplicar transformação seno/cosseno para capturar ciclicidade

5. **Normalização**:
   - Z-score standardization (média=0, desvio=1) para todas as features
   - Razão: MLP converge mais rápido com features na mesma escala
   - Target também será normalizado para facilitar treinamento""")

add_code_cell("""print("ETAPA 1: PREPARAÇÃO INICIAL")
print("="*80)

# Criar cópia para preservar dados originais
df_processed = df.copy()

# Remover colunas irrelevantes
columns_to_drop = ['instant', 'dteday', 'casual', 'registered']
df_processed = df_processed.drop(columns=columns_to_drop)

print(f"Colunas removidas: {columns_to_drop}")
print(f"Shape após remoção: {df_processed.shape}")
print(f"\\nColunas restantes: {list(df_processed.columns)}")""")

add_code_cell("""print("ETAPA 2: FEATURE ENGINEERING - VARIÁVEIS CÍCLICAS")
print("="*80)
print("Transformando variáveis cíclicas (hora, mês, dia da semana) em sin/cos")
print("Razão: Capturar a natureza cíclica (ex: hora 23 está próxima da hora 0)\\n")

# Função para criar features cíclicas
def encode_cyclical_feature(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    return data

# Aplicar transformação cíclica
df_processed = encode_cyclical_feature(df_processed, 'hr', 24)
df_processed = encode_cyclical_feature(df_processed, 'mnth', 12)
df_processed = encode_cyclical_feature(df_processed, 'weekday', 7)

print(f"Features cíclicas criadas:")
print(f"  - hr_sin, hr_cos (hora do dia)")
print(f"  - mnth_sin, mnth_cos (mês)")
print(f"  - weekday_sin, weekday_cos (dia da semana)")

# Remover features originais (mantemos apenas sin/cos)
df_processed = df_processed.drop(columns=['hr', 'mnth', 'weekday'])

print(f"\\nFeatures originais removidas (substituídas por sin/cos)")
print(f"Shape após feature engineering: {df_processed.shape}")""")

add_code_cell("""print("ETAPA 3: SEPARAÇÃO DE FEATURES E TARGET")
print("="*80)

# Separar features (X) e target (y)
X = df_processed.drop(columns=['cnt']).values
y = df_processed['cnt'].values.reshape(-1, 1)

feature_names = df_processed.drop(columns=['cnt']).columns.tolist()

print(f"Features (X): shape = {X.shape}")
print(f"Target (y): shape = {y.shape}")
print(f"\\nLista de features ({len(feature_names)}):")
for i, name in enumerate(feature_names, 1):
    print(f"  {i:2d}. {name}")""")

add_code_cell("""print("ETAPA 4: NORMALIZAÇÃO (Z-SCORE STANDARDIZATION)")
print("="*80)
print("Normalizando features e target usando z-score: (x - μ) / σ")
print("Razão: MLP converge mais rápido com features na mesma escala\\n")

# Salvar estatísticas ANTES da normalização (para desnormalização posterior)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
y_mean = y.mean()
y_std = y.std()

print(f"Estatísticas do Target (ANTES da normalização):")
print(f"  Média: {y_mean:.2f}")
print(f"  Desvio padrão: {y_std:.2f}")
print(f"  Min: {y.min():.2f}")
print(f"  Max: {y.max():.2f}")

# Normalizar features
X_normalized = (X - X_mean) / (X_std + 1e-8)  # +epsilon para evitar divisão por zero

# Normalizar target
y_normalized = (y - y_mean) / y_std

print(f"\\nFeatures normalizadas: shape = {X_normalized.shape}")
print(f"Target normalizado: shape = {y_normalized.shape}")

print(f"\\nEstatísticas do Target (DEPOIS da normalização):")
print(f"  Média: {y_normalized.mean():.6f} (≈ 0)")
print(f"  Desvio padrão: {y_normalized.std():.6f} (≈ 1)")
print(f"  Min: {y_normalized.min():.2f}")
print(f"  Max: {y_normalized.max():.2f}")""")

add_code_cell("""# Visualização: Antes vs Depois da normalização
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Antes
axes[0].hist(y, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Contagem de Aluguéis', fontsize=12)
axes[0].set_ylabel('Frequência', fontsize=12)
axes[0].set_title('Target ANTES da Normalização', fontsize=14, fontweight='bold')
axes[0].axvline(y_mean, color='red', linestyle='--', linewidth=2, label=f'Média: {y_mean:.1f}')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Depois
axes[1].hist(y_normalized, bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[1].set_xlabel('Contagem Normalizada', fontsize=12)
axes[1].set_ylabel('Frequência', fontsize=12)
axes[1].set_title('Target DEPOIS da Normalização', fontsize=14, fontweight='bold')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Média: 0.0')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()""")

add_markdown_cell("""### Resumo do Pré-processamento

✓ **Colunas removidas:** `instant`, `dteday`, `casual`, `registered`
✓ **Feature engineering:** Variáveis cíclicas transformadas em sin/cos
✓ **Valores ausentes:** Nenhum encontrado
✓ **Outliers:** Mantidos (representam variação real)
✓ **Normalização:** Z-score standardization aplicada
✓ **Features finais:** 15 variáveis
✓ **Amostras:** 17,379""")

# Continua no próximo bloco...
print("Células adicionadas até seção 5 (normalização)...")

# Salvar parcialmente
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Total de células agora: {len(notebook['cells'])}")
