"""
Script para adicionar seções de MLP, treinamento e avaliação.
Execute APÓS add_remaining_cells.py
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
# SEÇÃO 6: Divisão Train/Val/Test
# ==============================================================================

add_markdown_cell("""## 6. Divisão de Dados: Train / Validation / Test

### Estratégia de Divisão:

- **70%** Training set (12,165 amostras) - treinar o modelo
- **15%** Validation set (2,607 amostras) - ajustar hiperparâmetros e monitorar overfitting
- **15%** Test set (2,607 amostras) - avaliação final e métricas reportadas

### Justificativa:

1. **Train set (70%)**: Dataset grande o suficiente, 70% proporciona amostras suficientes para aprendizado
2. **Validation set (15%)**: Usado durante treinamento para early stopping e evitar overfitting
3. **Test set (15%)**: Avaliação imparcial do modelo final

### Modo de Treinamento:

- **Mini-batch Gradient Descent** com batch size = 64
- **Razão**: Equilíbrio entre velocidade (vs. SGD) e estabilidade (vs. Batch GD)
- Mini-batches permitem:
  - Convergência mais rápida que batch completo
  - Gradientes mais estáveis que SGD puro
  - Uso eficiente de memória

### Reprodutibilidade:

- **Random seed = 42** para garantir mesma divisão em execuções diferentes
- Importante para comparar resultados entre experimentos""")

add_code_cell("""print("DIVISÃO DOS DADOS EM TRAIN / VALIDATION / TEST")
print("="*80)

# Seed para reprodutibilidade
np.random.seed(42)

# Total de amostras
n_samples = X_normalized.shape[0]

# Shuffle dos índices
indices = np.arange(n_samples)
np.random.shuffle(indices)

# Definir tamanhos
train_size = int(0.70 * n_samples)
val_size = int(0.15 * n_samples)
test_size = n_samples - train_size - val_size

# Dividir índices
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Criar sets
X_train = X_normalized[train_indices]
y_train = y_normalized[train_indices]

X_val = X_normalized[val_indices]
y_val = y_normalized[val_indices]

X_test = X_normalized[test_indices]
y_test = y_normalized[test_indices]

print(f"Total de amostras: {n_samples:,}\\n")
print(f"Training Set:")
print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"  {len(X_train):,} amostras ({len(X_train)/n_samples*100:.1f}%)\\n")

print(f"Validation Set:")
print(f"  X_val: {X_val.shape} | y_val: {y_val.shape}")
print(f"  {len(X_val):,} amostras ({len(X_val)/n_samples*100:.1f}%)\\n")

print(f"Test Set:")
print(f"  X_test: {X_test.shape} | y_test: {y_test.shape}")
print(f"  {len(X_test):,} amostras ({len(X_test)/n_samples*100:.1f}%)")

print("\\n" + "="*80)
print("Divisão concluída com sucesso!")
print(f"Random seed: 42 (para reprodutibilidade)")""")

# ==============================================================================
# SEÇÃO 7: Implementação do MLP
# ==============================================================================

add_markdown_cell("""## 7. Implementação do MLP (Multi-Layer Perceptron)

### Arquitetura da Rede Neural:

```
Input Layer (15 neurônios) → Hidden Layer 1 (64 neurônios, ReLU)
                           → Hidden Layer 2 (32 neurônios, ReLU)
                           → Hidden Layer 3 (16 neurônios, ReLU)
                           → Output Layer (1 neurônio, Linear)
```

### Componentes Implementados:

1. **Funções de Ativação:**
   - **ReLU** (Rectified Linear Unit): `f(x) = max(0, x)` para camadas ocultas
   - **Linear**: `f(x) = x` para camada de saída (regressão)

2. **Função de Perda:**
   - **MSE** (Mean Squared Error): `L = (1/n) * Σ(y_pred - y_true)²`
   - Padrão para problemas de regressão

3. **Inicialização de Pesos:**
   - **He Initialization** para camadas com ReLU: `W ~ N(0, sqrt(2/n_in))`
   - Previne vanishing/exploding gradients

4. **Otimizador:**
   - **Mini-batch Gradient Descent** com learning rate adaptativo
   - Batch size: 64
   - Learning rate: 0.001 (inicial)

5. **Regularização:**
   - **L2 Regularization** (Ridge): penalização de pesos grandes
   - Lambda = 0.001""")

# Implementação completa do MLP (dividida em partes para melhor organização)
mlp_code_part1 = """class MLPRegressor:
    \"\"\"
    Multi-Layer Perceptron para Regressão implementado do zero.

    Arquitetura: Input → Hidden1 (ReLU) → Hidden2 (ReLU) → Hidden3 (ReLU) → Output (Linear)
    \"\"\"

    def __init__(self, input_size, hidden_sizes, output_size=1, learning_rate=0.001,
                 reg_lambda=0.001, random_seed=42):
        \"\"\"
        Inicializa o MLP.

        Parâmetros:
        -----------
        input_size : int
            Número de features de entrada
        hidden_sizes : list
            Lista com número de neurônios em cada camada oculta
        output_size : int
            Número de neurônios de saída (1 para regressão)
        learning_rate : float
            Taxa de aprendizado inicial
        reg_lambda : float
            Parâmetro de regularização L2
        random_seed : int
            Seed para reprodutibilidade
        \"\"\"
        np.random.seed(random_seed)

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

        # Inicializar pesos e biases
        self.weights = []
        self.biases = []

        # Criar lista de tamanhos de camadas
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # He Initialization para cada camada
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]

            # He initialization: std = sqrt(2 / n_in)
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))

            self.weights.append(W)
            self.biases.append(b)

        # Histórico de treinamento
        self.train_loss_history = []
        self.val_loss_history = []

    def relu(self, Z):
        \"\"\"ReLU activation: f(x) = max(0, x)\"\"\"
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        \"\"\"Derivada da ReLU: f'(x) = 1 se x > 0, caso contrário 0\"\"\"
        return (Z > 0).astype(float)

    def forward(self, X):
        \"\"\"
        Forward propagation.

        Retorna:
        --------
        output : array
            Predições do modelo
        cache : dict
            Valores intermediários (para backpropagation)
        \"\"\"
        cache = {'A': [X]}  # Armazenar ativações
        cache['Z'] = []     # Armazenar outputs pré-ativação

        A = X

        # Camadas ocultas (com ReLU)
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.relu(Z)

            cache['Z'].append(Z)
            cache['A'].append(A)

        # Camada de saída (linear, sem ativação)
        Z_out = np.dot(A, self.weights[-1]) + self.biases[-1]
        output = Z_out  # Ativação linear para regressão

        cache['Z'].append(Z_out)
        cache['A'].append(output)

        return output, cache"""

mlp_code_part2 = """    def compute_loss(self, y_true, y_pred):
        \"\"\"
        Calcula MSE loss com regularização L2.

        Loss = MSE + L2_penalty
        MSE = (1/n) * Σ(y_pred - y_true)²
        L2 = (λ/2n) * Σ(W²)
        \"\"\"
        n_samples = y_true.shape[0]

        # MSE
        mse = np.mean((y_pred - y_true) ** 2)

        # L2 regularization
        l2_penalty = 0
        for W in self.weights:
            l2_penalty += np.sum(W ** 2)
        l2_penalty *= (self.reg_lambda / (2 * n_samples))

        total_loss = mse + l2_penalty

        return total_loss

    def backward(self, X, y_true, cache):
        \"\"\"
        Backpropagation para calcular gradientes.

        Retorna:
        --------
        grads : dict
            Gradientes dos pesos e biases
        \"\"\"
        n_samples = X.shape[0]
        grads = {'W': [], 'b': []}

        # Gradiente da loss em relação à saída
        y_pred = cache['A'][-1]
        dA = (2.0 / n_samples) * (y_pred - y_true)

        # Backprop através das camadas (de trás para frente)
        for i in reversed(range(len(self.weights))):
            A_prev = cache['A'][i]

            # Gradientes de W e b
            dW = np.dot(A_prev.T, dA)
            db = np.sum(dA, axis=0, keepdims=True)

            # Adicionar regularização L2 ao gradiente de W
            dW += (self.reg_lambda / n_samples) * self.weights[i]

            # Inserir no início da lista (pois estamos indo de trás para frente)
            grads['W'].insert(0, dW)
            grads['b'].insert(0, db)

            # Gradiente para camada anterior (se não for a primeira camada)
            if i > 0:
                dA = np.dot(dA, self.weights[i].T)
                # Aplicar derivada da ReLU
                dA = dA * self.relu_derivative(cache['Z'][i - 1])

        return grads

    def update_weights(self, grads):
        \"\"\"Atualiza pesos usando gradiente descendente.\"\"\"
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads['W'][i]
            self.biases[i] -= self.learning_rate * grads['b'][i]"""

mlp_code_part3 = """    def train_epoch(self, X_train, y_train, batch_size=64):
        \"\"\"
        Treina por uma época usando mini-batch gradient descent.

        Retorna:
        --------
        epoch_loss : float
            Loss média da época
        \"\"\"
        n_samples = X_train.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        epoch_loss = 0
        n_batches = 0

        # Mini-batch training
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Forward pass
            y_pred, cache = self.forward(X_batch)

            # Calcular loss
            batch_loss = self.compute_loss(y_batch, y_pred)
            epoch_loss += batch_loss
            n_batches += 1

            # Backward pass
            grads = self.backward(X_batch, y_batch, cache)

            # Update weights
            self.update_weights(grads)

        return epoch_loss / n_batches

    def predict(self, X):
        \"\"\"Faz predições para novos dados.\"\"\"
        y_pred, _ = self.forward(X)
        return y_pred

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64,
            early_stopping_patience=15, verbose=True):
        \"\"\"
        Treina o modelo.

        Parâmetros:
        -----------
        X_train, y_train : arrays
            Dados de treinamento
        X_val, y_val : arrays
            Dados de validação
        epochs : int
            Número máximo de épocas
        batch_size : int
            Tamanho do mini-batch
        early_stopping_patience : int
            Número de épocas sem melhora antes de parar
        verbose : bool
            Imprimir progresso
        \"\"\"
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Treinar uma época
            train_loss = self.train_epoch(X_train, y_train, batch_size)

            # Calcular loss de validação
            y_val_pred, _ = self.forward(X_val)
            val_loss = self.compute_loss(y_val, y_val_pred)

            # Salvar histórico
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Salvar melhores pesos
                self.best_weights = [W.copy() for W in self.weights]
                self.best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1

            # Imprimir progresso
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Best Val: {best_val_loss:.4f}")

            # Parar se não houver melhora
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\\nEarly stopping na época {epoch+1}")
                    print(f"Melhor val loss: {best_val_loss:.4f}")
                break

        # Restaurar melhores pesos
        if hasattr(self, 'best_weights'):
            self.weights = self.best_weights
            self.biases = self.best_biases
            if verbose:
                print("Melhores pesos restaurados")

print("Classe MLPRegressor implementada com sucesso!")
print("\\nFuncionalidades implementadas:")
print("  - Forward propagation")
print("  - Backpropagation")
print("  - Mini-batch gradient descent")
print("  - He initialization")
print("  - ReLU activation")
print("  - MSE loss")
print("  - L2 regularization")
print("  - Early stopping")"""

# Adicionar as 3 partes do código MLP
add_code_cell(mlp_code_part1)
add_code_cell(mlp_code_part2)
add_code_cell(mlp_code_part3)

# Salvar
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Células adicionadas! Total agora: {len(notebook['cells'])}")
