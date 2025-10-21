"""
Cria um notebook de regressão simples e direto ao ponto.
Sem formalidades excessivas.
"""

import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.8.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split("\n")
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.split("\n")
    })

# ==============================================================================
# NOTEBOOK SIMPLIFICADO
# ==============================================================================

add_md("""# Projeto de Regressão - MLP

**Objetivo:** Prever demanda de bicicletas compartilhadas usando uma rede neural MLP implementada do zero.

**Dataset:** Bike Sharing (UCI) - 17k amostras com dados de clima e horário

**O que vamos fazer:**
1. Carregar e explorar os dados
2. Limpar e preparar os dados
3. Implementar MLP do zero (só NumPy)
4. Treinar o modelo
5. Avaliar resultados""")

add_md("""## 1. Setup e Dataset

Primeiro, baixe o dataset:
- Link: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
- Arquivo: `hour.csv`
- Ou execute: `python download_dataset.py`""")

add_code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
print("Pronto!")""")

add_code("""# Carregar dados
df = pd.read_csv('hour.csv')
print(f"Dataset: {df.shape[0]} linhas, {df.shape[1]} colunas")
df.head()""")

add_md("""## 2. Exploração Rápida

Vamos ver o que temos e entender os dados.""")

add_code("""# Info básica
print("Informações do Dataset:")
print("="*60)
df.info()
print("\\nEstatísticas:")
df.describe()""")

add_code("""# Variável target: cnt (contagem de aluguéis)
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.hist(df['cnt'], bins=50, edgecolor='black')
plt.title('Distribuição do Target')
plt.xlabel('Número de aluguéis')

plt.subplot(1, 3, 2)
df.groupby('hr')['cnt'].mean().plot(marker='o')
plt.title('Demanda por Hora do Dia')
plt.xlabel('Hora')
plt.ylabel('Média de aluguéis')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(df['temp'], df['cnt'], alpha=0.2, s=5)
plt.title('Temperatura vs Demanda')
plt.xlabel('Temperatura (normalizada)')
plt.ylabel('Aluguéis')

plt.tight_layout()
plt.show()

print(f"Target (cnt): min={df['cnt'].min()}, max={df['cnt'].max()}, média={df['cnt'].mean():.1f}")""")

add_md("""## 3. Preparação dos Dados

Vamos limpar e preparar os dados para o modelo.""")

add_code("""# Remover colunas que não vamos usar
df_clean = df.drop(columns=['instant', 'dteday', 'casual', 'registered'])

# Transformar variáveis cíclicas (hora, mês, dia da semana) em sin/cos
# Isso ajuda o modelo a entender que hora 23 está perto da hora 0
def add_cyclical(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    return data

df_clean = add_cyclical(df_clean, 'hr', 24)
df_clean = add_cyclical(df_clean, 'mnth', 12)
df_clean = add_cyclical(df_clean, 'weekday', 7)
df_clean = df_clean.drop(columns=['hr', 'mnth', 'weekday'])

print(f"Features após preparação: {df_clean.shape[1]} colunas")
print(f"Lista de features: {list(df_clean.columns)}")""")

add_code("""# Separar X (features) e y (target)
X = df_clean.drop(columns=['cnt']).values
y = df_clean['cnt'].values.reshape(-1, 1)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Normalizar (z-score)
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
y_mean, y_std = y.mean(), y.std()

X = (X - X_mean) / (X_std + 1e-8)
y = (y - y_mean) / y_std

print(f"\\nDados normalizados!")
print(f"y: média={y.mean():.3f}, std={y.std():.3f}")""")

add_code("""# Dividir em train/val/test (70/15/15)
n = X.shape[0]
idx = np.arange(n)
np.random.shuffle(idx)

train_size = int(0.70 * n)
val_size = int(0.15 * n)

train_idx = idx[:train_size]
val_idx = idx[train_size:train_size + val_size]
test_idx = idx[train_size + val_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")""")

add_md("""## 4. Implementação do MLP

Rede neural implementada do zero com NumPy.

**Arquitetura:** Input(15) → Hidden(64) → Hidden(32) → Hidden(16) → Output(1)""")

add_code("""class MLP:
    def __init__(self, layers, lr=0.001, reg=0.001):
        self.lr = lr
        self.reg = reg
        self.weights = []
        self.biases = []

        # Inicializar pesos (He initialization)
        for i in range(len(layers) - 1):
            W = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(W)
            self.biases.append(b)

        self.train_loss = []
        self.val_loss = []

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.cache = {'A': [X], 'Z': []}
        A = X

        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = self.relu(Z)
            self.cache['Z'].append(Z)
            self.cache['A'].append(A)

        # Output layer (linear)
        Z = A @ self.weights[-1] + self.biases[-1]
        self.cache['Z'].append(Z)
        self.cache['A'].append(Z)

        return Z

    def loss(self, y_true, y_pred):
        mse = np.mean((y_pred - y_true) ** 2)
        l2 = sum(np.sum(W ** 2) for W in self.weights)
        return mse + (self.reg / (2 * len(y_true))) * l2

    def backward(self, y_true):
        m = len(y_true)
        y_pred = self.cache['A'][-1]
        dA = (2.0 / m) * (y_pred - y_true)

        grads_W = []
        grads_b = []

        for i in reversed(range(len(self.weights))):
            A_prev = self.cache['A'][i]

            dW = A_prev.T @ dA
            db = np.sum(dA, axis=0, keepdims=True)

            dW += (self.reg / m) * self.weights[i]

            grads_W.insert(0, dW)
            grads_b.insert(0, db)

            if i > 0:
                dA = (dA @ self.weights[i].T) * self.relu_grad(self.cache['Z'][i-1])

        # Update
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_W[i]
            self.biases[i] -= self.lr * grads_b[i]

    def fit(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=64, patience=15):
        best_loss = float('inf')
        patience_count = 0

        for epoch in range(epochs):
            # Mini-batch training
            idx = np.arange(len(X_train))
            np.random.shuffle(idx)

            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                batch_idx = idx[start:end]

                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                y_pred = self.forward(X_batch)
                self.backward(y_batch)

            # Calcular loss
            train_pred = self.forward(X_train)
            val_pred = self.forward(X_val)

            train_loss = self.loss(y_train, train_pred)
            val_loss = self.loss(y_val, val_pred)

            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_count = 0
                self.best_weights = [W.copy() for W in self.weights]
                self.best_biases = [b.copy() for b in self.biases]
            else:
                patience_count += 1

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_loss:.4f}")

            if patience_count >= patience:
                print(f"\\nEarly stopping! Melhor val loss: {best_loss:.4f}")
                break

        # Restaurar melhores pesos
        self.weights = self.best_weights
        self.biases = self.best_biases

    def predict(self, X):
        A = X
        for i in range(len(self.weights) - 1):
            A = self.relu(A @ self.weights[i] + self.biases[i])
        return A @ self.weights[-1] + self.biases[-1]

print("Classe MLP implementada!")""")

add_md("""## 5. Treinamento

Vamos treinar o modelo. Isso pode levar 2-3 minutos.""")

add_code("""# Criar e treinar modelo
model = MLP(
    layers=[15, 64, 32, 16, 1],
    lr=0.001,
    reg=0.001
)

print("Treinando...")
model.fit(X_train, y_train, X_val, y_val, epochs=200, batch_size=64, patience=15)
print("\\nTreinamento concluído!")""")

add_code("""# Curvas de loss
plt.figure(figsize=(10, 4))
plt.plot(model.train_loss, label='Train', linewidth=2)
plt.plot(model.val_loss, label='Validation', linewidth=2)
plt.xlabel('Época')
plt.ylabel('Loss')
plt.title('Convergência do Modelo')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Loss final - Train: {model.train_loss[-1]:.4f} | Val: {model.val_loss[-1]:.4f}")""")

add_md("""## 6. Avaliação

Vamos ver como o modelo se saiu no conjunto de teste.""")

add_code("""# Predições (desnormalizar para valores reais)
y_pred_test = model.predict(X_test) * y_std + y_mean
y_true_test = y_test * y_std + y_mean

y_pred_train = model.predict(X_train) * y_std + y_mean
y_true_train = y_train * y_std + y_mean

# Métricas
def calc_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

train_metrics = calc_metrics(y_true_train, y_pred_train)
test_metrics = calc_metrics(y_true_test, y_pred_test)

print("RESULTADOS:")
print("="*60)
print(f"{'Métrica':<10} {'Train':<15} {'Test':<15}")
print("-"*60)
print(f"{'MAE':<10} {train_metrics['MAE']:<15.2f} {test_metrics['MAE']:<15.2f}")
print(f"{'RMSE':<10} {train_metrics['RMSE']:<15.2f} {test_metrics['RMSE']:<15.2f}")
print(f"{'R²':<10} {train_metrics['R2']:<15.4f} {test_metrics['R2']:<15.4f}")
print("="*60)

# Baseline (sempre prever a média)
baseline_pred = np.full_like(y_true_test, y_true_train.mean())
baseline_metrics = calc_metrics(y_true_test, baseline_pred)

print(f"\\nBaseline (prever média): RMSE = {baseline_metrics['RMSE']:.2f}")
print(f"Nosso modelo: RMSE = {test_metrics['RMSE']:.2f}")
print(f"Melhoria: {((baseline_metrics['RMSE'] - test_metrics['RMSE']) / baseline_metrics['RMSE'] * 100):.1f}%")""")

add_code("""# Visualizações
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 1. Predito vs Real
axes[0].scatter(y_true_test, y_pred_test, alpha=0.4, s=10)
lim = [y_true_test.min(), y_true_test.max()]
axes[0].plot(lim, lim, 'r--', linewidth=2)
axes[0].set_xlabel('Real')
axes[0].set_ylabel('Predito')
axes[0].set_title(f'Predições vs Real (R²={test_metrics["R2"]:.3f})')
axes[0].grid(True, alpha=0.3)

# 2. Resíduos
residuals = (y_true_test - y_pred_test).flatten()
axes[1].scatter(y_pred_test, residuals, alpha=0.4, s=10)
axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predito')
axes[1].set_ylabel('Resíduo')
axes[1].set_title('Análise de Resíduos')
axes[1].grid(True, alpha=0.3)

# 3. Distribuição dos resíduos
axes[2].hist(residuals, bins=50, edgecolor='black')
axes[2].axvline(0, color='r', linestyle='--', linewidth=2)
axes[2].set_xlabel('Resíduo')
axes[2].set_ylabel('Frequência')
axes[2].set_title(f'Distribuição dos Resíduos (média={residuals.mean():.2f})')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\\nEstatísticas dos resíduos:")
print(f"Média: {residuals.mean():.2f} (ideal: ~0)")
print(f"Desvio padrão: {residuals.std():.2f}")
print(f"50% dos erros < {np.percentile(np.abs(residuals), 50):.1f} bikes")
print(f"95% dos erros < {np.percentile(np.abs(residuals), 95):.1f} bikes")""")

add_md("""## 7. Conclusão

### O que fizemos:
- Implementamos um MLP do zero usando apenas NumPy
- Treinamos para prever demanda de bikes
- Obtivemos resultados melhores que o baseline

### Limitações:
- MLP não captura bem padrões temporais (melhor seria LSTM)
- Features podem ser melhoradas (adicionar lags, interações)
- Hiperparâmetros não foram otimizados

### Melhorias possíveis:
- Testar diferentes arquiteturas
- Adicionar dropout
- Usar Adam optimizer
- Fazer grid search de hiperparâmetros

### Referências:
- Dataset: UCI ML Repository (Bike Sharing)
- Implementação: NumPy, conceitos de Deep Learning (Goodfellow et al.)""")

# Salvar
output = r"c:\Users\pedro\OneDrive\Documents\Insper\9_semestre\redes_neurais_e_deep_learning\pedrocivita.github.io\site\projects\2\main\regression.ipynb"
with open(output, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✓ Notebook simples criado!")
print(f"✓ Total: {len(notebook['cells'])} células")
print(f"✓ Arquivo: {output}")
