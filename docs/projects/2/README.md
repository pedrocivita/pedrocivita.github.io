# Projeto de Regressão: Previsão de Demanda de Bicicletas Compartilhadas

**Disciplina:** Redes Neurais e Deep Learning
**Projeto:** Regression Task com Multi-Layer Perceptron (MLP)
**Deadline:** 19.out.2025

---

## 📋 Descrição

Este projeto implementa uma rede neural **Multi-Layer Perceptron (MLP) do zero** usando apenas **NumPy** para resolver um problema de regressão real-world: prever a demanda de bicicletas compartilhadas com base em condições climáticas, temporais e sazonais.

## 🎯 Objetivos

- Implementar MLP completo do zero (forward/backward propagation)
- Aplicar técnicas de feature engineering para dados temporais
- Treinar modelo com regularização L2 e early stopping
- Avaliar performance com múltiplas métricas de regressão
- Analisar resultados e comparar com baseline

## 📊 Dataset

**Bike Sharing Demand Dataset** do UCI Machine Learning Repository

- **Fonte:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
- **Alternativa:** [Kaggle Competition](https://www.kaggle.com/c/bike-sharing-demand)
- **Tamanho:** 17,379 amostras (registros horários de 2011-2012)
- **Features:** 16 variáveis (temporais, climáticas, sazonais)
- **Target:** Contagem total de aluguéis de bicicletas (variável contínua)

### Por que este dataset?

✅ Relevância prática (sistemas de bike-sharing em cidades inteligentes)
✅ Complexidade adequada (relações não-lineares, sazonalidade)
✅ Evita datasets clássicos (Boston/California Housing)
✅ Possibilidade de participação em competição Kaggle (bonus)

## 🚀 Como Executar

### Pré-requisitos

```bash
# Bibliotecas necessárias
pip install numpy pandas matplotlib seaborn jupyter
```

### Passo 1: Download do Dataset

**Opção 1: UCI Repository (Recomendado)**

1. Acesse: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
2. Baixe `Bike-Sharing-Dataset.zip`
3. Extraia e copie o arquivo `hour.csv` para esta pasta

**Opção 2: Kaggle API**

```bash
# Instale o Kaggle CLI
pip install kaggle

# Configure suas credenciais Kaggle
# Depois execute:
kaggle competitions download -c bike-sharing-demand
```

### Passo 2: Executar o Notebook

```bash
# Inicie o Jupyter Notebook
jupyter notebook regression.ipynb

# Ou use Jupyter Lab
jupyter lab regression.ipynb
```

### Passo 3: Executar Células

Execute as células sequencialmente de cima para baixo. O notebook está organizado em seções numeradas.

## 📁 Estrutura do Projeto

```
regression.ipynb          # Notebook principal (49 células)
README.md                 # Este arquivo
hour.csv                  # Dataset (baixar separadamente)
create_regression_notebook.py   # Script auxiliar (geração do notebook)
add_remaining_cells.py    # Script auxiliar
add_mlp_and_training.py   # Script auxiliar
add_final_sections.py     # Script auxiliar
add_conclusion.py         # Script auxiliar
```

## 📖 Conteúdo do Notebook

### Seções Principais

1. **Seleção e Explicação do Dataset**
   - Justificativa da escolha
   - Descrição detalhada das features
   - Conhecimento de domínio

2. **Análise Exploratória**
   - Estatísticas descritivas
   - Detecção de valores ausentes e outliers
   - Visualizações (distribuições, correlações, padrões temporais)

3. **Pré-processamento**
   - Limpeza de dados
   - Feature engineering (variáveis cíclicas com sin/cos)
   - Normalização (z-score standardization)
   - Divisão Train/Val/Test (70/15/15)

4. **Implementação do MLP**
   - Arquitetura: [15 → 64 → 32 → 16 → 1]
   - Forward propagation
   - Backpropagation
   - He initialization
   - ReLU activation
   - MSE loss + L2 regularization

5. **Treinamento**
   - Mini-batch gradient descent (batch=64)
   - Early stopping (patience=15)
   - Curvas de convergência
   - Análise de overfitting/underfitting

6. **Avaliação**
   - Métricas: MAE, MSE, RMSE, R², MAPE
   - Comparação com baseline (média)
   - Visualizações (predito vs real, resíduos)
   - Análise estatística dos erros

7. **Discussão e Conclusão**
   - Pontos fortes e limitações
   - Melhorias futuras
   - Aprendizados principais

## 🔧 Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| **Arquitetura** | [15 → 64 → 32 → 16 → 1] | Redução progressiva para hierarquia de features |
| **Learning Rate** | 0.001 | Convergência estável sem oscilações |
| **Batch Size** | 64 | Equilíbrio entre velocidade e estabilidade |
| **Épocas (máx)** | 200 | Suficiente para convergência |
| **Early Stopping** | 15 épocas | Previne overfitting |
| **L2 Lambda** | 0.001 | Regularização moderada |

## 📊 Métricas de Avaliação

- **MAE (Mean Absolute Error)**: Erro médio em número de aluguéis
- **MSE (Mean Squared Error)**: Penaliza erros grandes
- **RMSE (Root MSE)**: Erro na mesma unidade do target
- **R² (Coef. Determinação)**: Variância explicada pelo modelo
- **MAPE (Mean Absolute % Error)**: Erro percentual médio

## 🎓 Conceitos Implementados

### Redes Neurais
- [x] Forward propagation
- [x] Backpropagation
- [x] Gradient descent (mini-batch)
- [x] He initialization
- [x] ReLU activation
- [x] Linear output (regressão)

### Regularização
- [x] L2 regularization (Ridge)
- [x] Early stopping
- [x] Train/val/test split

### Feature Engineering
- [x] Transformação cíclica (sin/cos)
- [x] Normalização (z-score)
- [x] Remoção de features irrelevantes

### Avaliação
- [x] Múltiplas métricas de regressão
- [x] Análise de resíduos
- [x] Comparação com baseline
- [x] Visualizações de performance

## ⚠️ Requisitos do Projeto

✅ Dataset real com >1,000 amostras e >5 features
✅ Evita datasets clássicos (Boston/California Housing)
✅ MLP implementado do zero (NumPy apenas)
✅ Todas as etapas explicadas e justificadas
✅ Train/val/test split adequado
✅ Early stopping implementado
✅ Múltiplas métricas de avaliação
✅ Análise de curvas de erro
✅ Comparação com baseline
✅ Conclusão e limitações discutidas
✅ Referências citadas

## 📚 Referências Principais

1. **Dataset:** Fanaee-T & Gama (2013) - UCI ML Repository
2. **Deep Learning:** Goodfellow, Bengio & Courville (2016)
3. **He Initialization:** He et al. (2015) - ICCV
4. **Material do Curso:**
   - https://caioboa.github.io/DeepLearningPages/
   - https://insper.github.io/ann-dl/versions/2025.2/projects/regression/

## 🤝 Ferramentas de IA Utilizadas

- **Claude Code (Anthropic)**: Assistência na estruturação do código e documentação
- Todas as implementações foram **compreendidas e validadas manualmente**

## 🏆 Oportunidade de Bonus

Este dataset está disponível em **competição no Kaggle**:
- Submissão válida: +0.5 pontos
- Top 50% do leaderboard: +0.5 pontos adicionais

Link: https://www.kaggle.com/c/bike-sharing-demand

## 📝 Notas Importantes

1. **Execução Sequencial**: Execute as células na ordem (top-down)
2. **Tempo de Treinamento**: ~2-5 minutos (depende do hardware)
3. **Reprodutibilidade**: Seed=42 garante mesmos resultados
4. **Dados Normalizados**: Predições são desnormalizadas para interpretação
5. **Visualizações**: Requerem matplotlib e seaborn

## 🔍 Próximos Passos (Melhorias Futuras)

1. **Otimizadores avançados**: Adam, RMSprop
2. **Arquiteturas**: LSTM/GRU para capturar dependência temporal
3. **Regularização adicional**: Dropout, Batch Normalization
4. **Features**: Lags temporais, interações
5. **Ensemble**: Combinar MLP com XGBoost/Random Forest
6. **Hyperparameter tuning**: Grid search ou Bayesian optimization
7. **Deployment**: API com FastAPI/Flask

## 📧 Contato

Para dúvidas sobre o projeto, consulte:
- Material do curso
- Office hours do professor
- Documentação oficial das bibliotecas

---

**Data de criação:** Outubro 2025
**Versão:** 1.0
**Status:** ✅ Completo e pronto para execução
