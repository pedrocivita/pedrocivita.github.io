# 🚀 Guia Rápido de Início

**Projeto de Regressão - Bike Sharing Demand**

---

## ⚡ Início Rápido (3 passos)

### 1️⃣ Instalar Dependências

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

### 2️⃣ Baixar Dataset

**Opção A - Automático (recomendado):**
```bash
python download_dataset.py
```

**Opção B - Manual:**
1. Acesse: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
2. Baixe `Bike-Sharing-Dataset.zip`
3. Extraia e copie `hour.csv` para esta pasta

### 3️⃣ Executar Notebook

```bash
jupyter notebook regression.ipynb
```

---

## 📊 O que esperar

### Estrutura do Notebook (49 células)

```
📌 Seções Principais:
├── 1-4:   Introdução e Dataset
├── 5-9:   Exploração de Dados
├── 10-14: Pré-processamento
├── 15-20: Implementação MLP
├── 21-30: Treinamento
├── 31-40: Avaliação
└── 41-49: Análise e Conclusão
```

### Tempo Estimado

- ⏱️ **Execução completa:** ~5-10 minutos
- ⏱️ **Treinamento do modelo:** ~2-3 minutos
- ⏱️ **Leitura e análise:** ~30-45 minutos

---

## 🎯 Checklist de Execução

### Antes de Executar

- [ ] Python 3.8+ instalado
- [ ] Bibliotecas instaladas (numpy, pandas, matplotlib, seaborn)
- [ ] Arquivo `hour.csv` na pasta do notebook
- [ ] Jupyter Notebook funcionando

### Durante a Execução

- [ ] Executar células sequencialmente (não pular células)
- [ ] Verificar se dataset foi carregado (célula 4)
- [ ] Aguardar treinamento completo (~100-150 épocas)
- [ ] Verificar convergência nas curvas de loss
- [ ] Analisar métricas finais (R², RMSE, MAE)

### Após Execução

- [ ] R² > 0.3 no test set (mínimo aceitável)
- [ ] Gap train-val < 20% (sem overfitting severo)
- [ ] Resíduos centrados em zero
- [ ] Visualizações geradas corretamente

---

## 🔧 Hiperparâmetros Principais

| Parâmetro | Valor Padrão | Onde Modificar |
|-----------|--------------|----------------|
| Learning Rate | 0.001 | Célula 30 (`LEARNING_RATE`) |
| Batch Size | 64 | Célula 30 (`BATCH_SIZE`) |
| Épocas Máx | 200 | Célula 30 (`EPOCHS`) |
| Arquitetura | [64,32,16] | Célula 30 (`HIDDEN_SIZES`) |
| L2 Lambda | 0.001 | Célula 30 (`REG_LAMBDA`) |

**💡 Dica:** Para experimentar, modifique os valores e re-execute a célula 30 em diante.

---

## 📈 Resultados Esperados

### Métricas Típicas (após treinamento)

| Métrica | Valor Esperado | Interpretação |
|---------|----------------|---------------|
| **R²** | 0.40 - 0.70 | Quanto da variância é explicada |
| **RMSE** | 60 - 100 | Erro médio em nº de bikes |
| **MAE** | 40 - 70 | Erro absoluto médio |
| **MAPE** | 20% - 40% | Erro percentual |

### Curva de Convergência

```
✓ Esperado:
  - Loss decrescente nas primeiras 50 épocas
  - Plateau ou oscilação mínima depois
  - Train e Val loss próximos (gap < 20%)

❌ Problemas:
  - Loss crescente: learning rate muito alto
  - Não converge: learning rate muito baixo
  - Val >> Train: overfitting
```

---

## 🐛 Troubleshooting

### Erro: "File 'hour.csv' not found"
```bash
# Solução:
python download_dataset.py
```

### Erro: "ModuleNotFoundError: No module named 'numpy'"
```bash
# Solução:
pip install numpy pandas matplotlib seaborn
```

### Erro: "Kernel died"
```bash
# Possíveis causas:
# 1. Memória insuficiente
# 2. Conflito de versões

# Solução:
# Reinicie o kernel: Kernel → Restart
# Se persistir, reduza BATCH_SIZE para 32
```

### Warning: "RuntimeWarning: overflow encountered"
```bash
# Causa: Learning rate muito alto ou pesos explodindo
# Solução:
# 1. Reduza LEARNING_RATE para 0.0005
# 2. Aumente REG_LAMBDA para 0.01
# 3. Re-execute o treinamento
```

### Loss não converge (fica oscilando)
```bash
# Soluções:
# 1. Reduza LEARNING_RATE (ex: 0.0005)
# 2. Aumente BATCH_SIZE (ex: 128)
# 3. Aumente L2 regularization (REG_LAMBDA = 0.01)
```

### Overfitting (Val >> Train loss)
```bash
# Soluções:
# 1. Aumente REG_LAMBDA (ex: 0.01)
# 2. Reduza arquitetura (ex: [32, 16, 8])
# 3. Reduza EARLY_STOPPING_PATIENCE (ex: 10)
```

---

## 📚 Seções Importantes

### **Células Essenciais (não pule!)**

- **Célula 3:** Carregamento do dataset
- **Célula 7:** Análise de outliers
- **Célula 14:** Normalização (salva y_mean e y_std)
- **Célula 16:** Split train/val/test
- **Células 18-20:** Implementação MLP (classe completa)
- **Célula 21:** Treinamento (pode demorar 2-3 min)
- **Célula 28:** Desnormalização (crítico para métricas corretas)

### **Células Opcionais (podem pular se tiver pressa)**

- Células de visualização exploratória (5, 6, 8, 9)
- Análise detalhada de resíduos (37, 38)
- Exemplos de predições (39)

---

## 💡 Dicas de Otimização

### Para Resultados Melhores

1. **Feature Engineering:**
   - Já implementado: transformação cíclica (sin/cos)
   - Experimente: adicionar lags temporais, interações

2. **Arquitetura:**
   - Teste diferentes profundidades: [128,64,32], [32,16]
   - Cuidado com underfitting (muito simples) ou overfitting (muito complexo)

3. **Hiperparâmetros:**
   - Learning rate: teste [0.0001, 0.0005, 0.001, 0.005]
   - Batch size: teste [32, 64, 128, 256]
   - L2 lambda: teste [0.0001, 0.001, 0.01]

4. **Treinamento:**
   - Aumente EPOCHS para 300-500 se convergência lenta
   - EARLY_STOPPING_PATIENCE pode ir até 20-30

### Para Execução Mais Rápida

1. Reduza EPOCHS para 50-100
2. Aumente BATCH_SIZE para 128
3. Reduza arquitetura: [32, 16]
4. Pule células de visualização

---

## 🎓 Conceitos Aprendidos

Ao final deste projeto, você terá implementado:

- ✅ Forward propagation
- ✅ Backpropagation
- ✅ Gradient descent (mini-batch)
- ✅ He initialization
- ✅ ReLU activation
- ✅ L2 regularization
- ✅ Early stopping
- ✅ Feature engineering (sin/cos)
- ✅ Métricas de regressão
- ✅ Análise de resíduos

---

## 📞 Suporte

### Documentação Completa

Leia [README.md](README.md) para documentação detalhada.

### Material de Apoio

- https://caioboa.github.io/DeepLearningPages/
- https://insper.github.io/ann-dl/versions/2025.2/projects/regression/

### Dúvidas Comuns

Consulte a seção **Discussão e Análise** (célula 44) do notebook.

---

## 🎉 Próximos Passos

Após executar com sucesso:

1. **Análise:** Interprete os resultados nas células 40-45
2. **Experimentação:** Modifique hiperparâmetros e re-treine
3. **Comparação:** Compare com baseline (célula 32)
4. **Melhorias:** Implemente sugestões da seção "Melhorias Futuras"
5. **Bonus:** Submeta para competição Kaggle (+0.5 pts)

---

**Boa sorte! 🚀**

Se encontrar problemas, revise o README ou consulte o material do curso.
