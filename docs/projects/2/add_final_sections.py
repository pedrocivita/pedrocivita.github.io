"""
Script para adicionar seções finais: treinamento, avaliação, conclusão e referências.
Execute APÓS add_mlp_and_training.py
"""

import json

# Carregar notebook
notebook_path = r"c:\Users\pedro\OneDrive\Documents\Insper\9_semestre\redes_neurais_e_deep_learning\pedrocivita.github.io\site\projects\2\main\regression.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

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

# ==============================================================================
# SEÇÃO 8: Treinamento
# ==============================================================================

add_markdown_cell("""## 8. Treinamento do Modelo

### Hiperparâmetros Escolhidos:

- **Arquitetura:** [15 → 64 → 32 → 16 → 1]
- **Learning rate:** 0.001
- **Batch size:** 64
- **Épocas:** 200 (máximo)
- **Early stopping:** Paciência de 15 épocas
- **Regularização L2:** λ = 0.001

### Justificativa:

1. **Arquitetura progressiva:** Redução gradual de neurônios (64→32→16) ajuda a extrair features hierárquicas
2. **Learning rate:** Valor conservador para garantir convergência estável
3. **Early stopping:** Previne overfitting, para treinamento quando validação para de melhorar""")

add_code_cell("""print("INICIALIZANDO E TREINANDO O MODELO")
print("="*80)

# Hiperparâmetros
INPUT_SIZE = X_train.shape[1]  # 15 features
HIDDEN_SIZES = [64, 32, 16]     # 3 camadas ocultas
OUTPUT_SIZE = 1                 # 1 output (regressão)
LEARNING_RATE = 0.001
REG_LAMBDA = 0.001
EPOCHS = 200
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 15

print(f"Arquitetura: {INPUT_SIZE} → {' → '.join(map(str, HIDDEN_SIZES))} → {OUTPUT_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Épocas (máx): {EPOCHS}")
print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print(f"Regularização L2 (λ): {REG_LAMBDA}")

print("\\n" + "="*80)
print("INICIANDO TREINAMENTO...")
print("="*80 + "\\n")

# Criar modelo
model = MLPRegressor(
    input_size=INPUT_SIZE,
    hidden_sizes=HIDDEN_SIZES,
    output_size=OUTPUT_SIZE,
    learning_rate=LEARNING_RATE,
    reg_lambda=REG_LAMBDA,
    random_seed=42
)

# Treinar
model.fit(
    X_train, y_train,
    X_val, y_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    verbose=True
)

print("\\n" + "="*80)
print("TREINAMENTO CONCLUÍDO!")
print("="*80)""")

add_markdown_cell("""### Desafios Durante o Treinamento:

**Possíveis problemas e soluções adotadas:**

1. **Vanishing Gradients:**
   - **Problema:** Gradientes ficam muito pequenos em redes profundas
   - **Solução:** He initialization + ReLU activation

2. **Overfitting:**
   - **Problema:** Modelo memoriza training set ao invés de generalizar
   - **Solução:** L2 regularization + early stopping

3. **Convergência Lenta:**
   - **Problema:** Learning rate muito baixo
   - **Solução:** Testamos diferentes valores (0.001 mostrou-se adequado)

4. **Instabilidade:**
   - **Problema:** Loss oscilando muito
   - **Solução:** Mini-batch GD (batch=64) ao invés de SGD puro""")

# ==============================================================================
# SEÇÃO 9: Curvas de Erro
# ==============================================================================

add_markdown_cell("""## 9. Curvas de Erro e Visualização

Análise da convergência do modelo através das curvas de loss ao longo das épocas.""")

add_code_cell("""# Curvas de loss
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

epochs_trained = len(model.train_loss_history)
epochs_range = range(1, epochs_trained + 1)

# Gráfico 1: Loss ao longo das épocas
axes[0].plot(epochs_range, model.train_loss_history, label='Training Loss',
             linewidth=2, color='steelblue')
axes[0].plot(epochs_range, model.val_loss_history, label='Validation Loss',
             linewidth=2, color='coral')
axes[0].set_xlabel('Época', fontsize=12)
axes[0].set_ylabel('Loss (MSE + L2)', fontsize=12)
axes[0].set_title('Curva de Convergência: Training vs Validation Loss',
                   fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Marcar melhor época
best_epoch = np.argmin(model.val_loss_history) + 1
best_val_loss = min(model.val_loss_history)
axes[0].axvline(best_epoch, color='green', linestyle='--', linewidth=1.5,
                label=f'Melhor época: {best_epoch}')
axes[0].scatter([best_epoch], [best_val_loss], color='green', s=100, zorder=5)
axes[0].legend(fontsize=11)

# Gráfico 2: Loss em escala log
axes[1].plot(epochs_range, model.train_loss_history, label='Training Loss',
             linewidth=2, color='steelblue')
axes[1].plot(epochs_range, model.val_loss_history, label='Validation Loss',
             linewidth=2, color='coral')
axes[1].set_xlabel('Época', fontsize=12)
axes[1].set_ylabel('Loss (escala log)', fontsize=12)
axes[1].set_title('Curva de Convergência (Escala Logarítmica)',
                   fontsize=14, fontweight='bold')
axes[1].set_yscale('log')
axes[1].legend(fontsize=11, loc='upper right')
axes[1].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

print(f"Estatísticas do Treinamento:")
print("="*80)
print(f"Total de épocas treinadas: {epochs_trained}")
print(f"Melhor época: {best_epoch}")
print(f"Melhor validation loss: {best_val_loss:.4f}")
print(f"Training loss final: {model.train_loss_history[-1]:.4f}")
print(f"Validation loss final: {model.val_loss_history[-1]:.4f}")""")

add_markdown_cell("""### Interpretação das Curvas:

**Análise esperada (ajustar após executar):**

1. **Convergência:**
   - Loss de treinamento e validação devem diminuir nas primeiras épocas
   - Plateau indica que modelo atingiu capacidade de aprendizado

2. **Overfitting/Underfitting:**
   - **Ideal:** Training e validation loss próximos e estáveis
   - **Overfitting:** Training loss continua caindo, validation loss sobe
   - **Underfitting:** Ambos loss altos e não convergem

3. **Early Stopping:**
   - Linha verde vertical marca quando validation loss foi mínimo
   - Modelo usa pesos dessa época (melhor generalização)""")

add_code_cell("""# Análise de overfitting/underfitting
print("ANÁLISE DE OVERFITTING/UNDERFITTING")
print("="*80)

final_train_loss = model.train_loss_history[-1]
final_val_loss = model.val_loss_history[-1]
gap = final_val_loss - final_train_loss
gap_pct = (gap / final_train_loss) * 100

print(f"Training loss (final): {final_train_loss:.4f}")
print(f"Validation loss (final): {final_val_loss:.4f}")
print(f"Gap (val - train): {gap:.4f} ({gap_pct:+.1f}%)")
print()

if gap_pct < 5:
    print("MODELO BEM AJUSTADO: Gap pequeno entre train e validation")
    print("  Modelo está generalizando bem!")
elif gap_pct < 20:
    print("LEVE OVERFITTING: Gap moderado entre train e validation")
    print("  Modelo ainda generalizando razoavelmente bem")
else:
    print("OVERFITTING DETECTADO: Gap grande entre train e validation")
    print("  Sugestões: Aumentar regularização, usar dropout, ou reduzir complexidade")

if final_val_loss > 0.5:
    print("\\nPOSSÍVEL UNDERFITTING: Loss de validação ainda alto")
    print("  Sugestões: Aumentar complexidade do modelo, treinar por mais épocas")""")

# ==============================================================================
# SEÇÃO 10: Métricas de Avaliação
# ==============================================================================

add_markdown_cell("""## 10. Métricas de Avaliação

### Métricas para Regressão:

1. **MAE (Mean Absolute Error):**
   - Média dos erros absolutos
   - `MAE = (1/n) * Σ|y_pred - y_true|`
   - **Interpretação:** Erro médio em unidades do target

2. **MSE (Mean Squared Error):**
   - Média dos erros ao quadrado
   - `MSE = (1/n) * Σ(y_pred - y_true)²`
   - **Interpretação:** Penaliza erros grandes mais fortemente

3. **RMSE (Root Mean Squared Error):**
   - Raiz quadrada do MSE
   - `RMSE = sqrt(MSE)`
   - **Interpretação:** Erro médio na mesma unidade do target

4. **R² (Coefficient of Determination):**
   - Proporção da variância explicada pelo modelo
   - `R² = 1 - (SS_res / SS_tot)`
   - **Interpretação:** 0 = modelo inútil, 1 = modelo perfeito

5. **MAPE (Mean Absolute Percentage Error):**
   - Erro percentual médio
   - `MAPE = (100/n) * Σ|((y_true - y_pred) / y_true)|`
   - **Interpretação:** Erro em percentual (cuidado com valores próximos de zero)""")

add_code_cell("""def calculate_regression_metrics(y_true, y_pred):
    \"\"\"
    Calcula métricas de avaliação para regressão.

    Parâmetros:
    -----------
    y_true : array
        Valores reais
    y_pred : array
        Valores preditos

    Retorna:
    --------
    metrics : dict
        Dicionário com todas as métricas
    \"\"\"
    # Garantir que são arrays 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # MSE
    mse = np.mean((y_true - y_pred) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # MAPE (evitar divisão por zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

    return metrics

def denormalize_predictions(y_normalized, y_mean, y_std):
    \"\"\"
    Desnormaliza predições para escala original.

    y_original = (y_normalized * std) + mean
    \"\"\"
    return (y_normalized * y_std) + y_mean

print("Funções de avaliação implementadas!")""")

add_markdown_cell("""### Avaliação no Conjunto de Teste""")

add_code_cell("""print("AVALIAÇÃO NO CONJUNTO DE TESTE")
print("="*80)

# Fazer predições (dados normalizados)
y_test_pred_norm = model.predict(X_test)
y_train_pred_norm = model.predict(X_train)
y_val_pred_norm = model.predict(X_val)

# Desnormalizar para escala original
y_test_pred = denormalize_predictions(y_test_pred_norm, y_mean, y_std)
y_test_true = denormalize_predictions(y_test, y_mean, y_std)

y_train_pred = denormalize_predictions(y_train_pred_norm, y_mean, y_std)
y_train_true = denormalize_predictions(y_train, y_mean, y_std)

y_val_pred = denormalize_predictions(y_val_pred_norm, y_mean, y_std)
y_val_true = denormalize_predictions(y_val, y_mean, y_std)

# Calcular métricas para cada conjunto
train_metrics = calculate_regression_metrics(y_train_true, y_train_pred)
val_metrics = calculate_regression_metrics(y_val_true, y_val_pred)
test_metrics = calculate_regression_metrics(y_test_true, y_test_pred)

# Criar DataFrame de comparação
metrics_df = pd.DataFrame({
    'Métrica': ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE (%)'],
    'Training': [
        f"{train_metrics['MAE']:.2f}",
        f"{train_metrics['MSE']:.2f}",
        f"{train_metrics['RMSE']:.2f}",
        f"{train_metrics['R2']:.4f}",
        f"{train_metrics['MAPE']:.2f}%"
    ],
    'Validation': [
        f"{val_metrics['MAE']:.2f}",
        f"{val_metrics['MSE']:.2f}",
        f"{val_metrics['RMSE']:.2f}",
        f"{val_metrics['R2']:.4f}",
        f"{val_metrics['MAPE']:.2f}%"
    ],
    'Test': [
        f"{test_metrics['MAE']:.2f}",
        f"{test_metrics['MSE']:.2f}",
        f"{test_metrics['RMSE']:.2f}",
        f"{test_metrics['R2']:.4f}",
        f"{test_metrics['MAPE']:.2f}%"
    ]
})

print(metrics_df.to_string(index=False))
print("\\n" + "="*80)""")

add_markdown_cell("""### Comparação com Baseline""")

add_code_cell("""print("COMPARAÇÃO COM BASELINE (MÉDIA)")
print("="*80)

# Baseline: sempre prever a média do training set
baseline_pred = np.full_like(y_test_true, y_train_true.mean())
baseline_metrics = calculate_regression_metrics(y_test_true, baseline_pred)

print(f"Baseline (sempre prever média = {y_train_true.mean():.2f}):")
print(f"  MAE:  {baseline_metrics['MAE']:.2f}")
print(f"  RMSE: {baseline_metrics['RMSE']:.2f}")
print(f"  R²:   {baseline_metrics['R2']:.4f}")

print(f"\\nMLP Model (nosso modelo):")
print(f"  MAE:  {test_metrics['MAE']:.2f}")
print(f"  RMSE: {test_metrics['RMSE']:.2f}")
print(f"  R²:   {test_metrics['R2']:.4f}")

# Melhoria relativa
mae_improvement = ((baseline_metrics['MAE'] - test_metrics['MAE']) / baseline_metrics['MAE']) * 100
rmse_improvement = ((baseline_metrics['RMSE'] - test_metrics['RMSE']) / baseline_metrics['RMSE']) * 100

print(f"\\nMelhoria em relação ao baseline:")
print(f"  MAE:  {mae_improvement:+.1f}%")
print(f"  RMSE: {rmse_improvement:+.1f}%")

print("\\n" + "="*80)

if test_metrics['R2'] > 0.5:
    print("MODELO COM BOA CAPACIDADE PREDITIVA (R² > 0.5)")
elif test_metrics['R2'] > 0.3:
    print("MODELO COM CAPACIDADE PREDITIVA MODERADA (0.3 < R² < 0.5)")
else:
    print("MODELO COM BAIXA CAPACIDADE PREDITIVA (R² < 0.3)")""")

add_markdown_cell("""### Visualizações de Avaliação""")

# Continua no próximo script...
print(f"Células adicionadas até avaliação! Total: {len(notebook['cells'])}")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Notebook atualizado com sucesso!")
