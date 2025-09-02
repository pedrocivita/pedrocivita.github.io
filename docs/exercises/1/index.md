# 1. Data

!!! abstract "Activity: Data Preparation and Analysis for Neural Networks"
    Esta atividade testa sua habilidade de **gerar dados sintéticos**, lidar com **dados reais** e **preparar dados** para redes neurais.

=== ":material-calendar: Deadline and Submission"
    - **📅 05.sep (friday)**  
    - **🕐 Commits until 23:59**  
    - **👤 Individual**  
    - **🔗 Submit:** apenas o **link do GitHub Pages** em *insper.blackboard.com*.

=== ":material-information: Observações Importantes"
    - **Sem prorrogação de prazo.**
    - **Colaboração com IA** é permitida, **mas você deve entender** e conseguir explicar **todo** o trabalho.
    - Entregas **individuais** via Blackboard.

---

## Exercise 1
### Exploring Class Separability in 2D

Entender a distribuição dos dados é o primeiro passo antes de projetar uma arquitetura. Gere e visualize um conjunto **2D** para discutir a complexidade da fronteira de decisão que uma rede precisaria aprender.

#### Instructions

1. **Generate the Data** (400 amostras = 4×100). **Gaussianas** com parâmetros:
   - **Class 0**: Mean = **[2, 3]**, Std = **[0.8, 2.5]**  
   - **Class 1**: Mean = **[5, 6]**, Std = **[1.2, 1.9]**  
   - **Class 2**: Mean = **[8, 1]**, Std = **[0.9, 0.9]**  
   - **Class 3**: Mean = **[15, 4]**, Std = **[0.5, 2.0]**
2. **Plot the Data**: Scatter 2D com **cores por classe**.
3. **Analyze and Draw Boundaries**:
   - Descreva a **distribuição/overlap**.
   - Uma **fronteira linear** separa tudo?
   - Esboce **fronteiras** que uma rede poderia aprender.

---

## Exercise 2
### Non-Linearity in Higher Dimensions

Crie duas classes **5D** (A e B) com **Normal multivariada**.

#### Instructions
- **Class A**: vetor de médias e **Σ** (5×5) conforme enunciado.
- **Class B**: vetor de médias e **Σ** (5×5) conforme enunciado.
- **PCA → 2D** para visualização; scatter colorido por classe.
- Discuta a **separabilidade linear** e por que **modelos com não-linearidade** (MLP) são adequados.

---

## Exercise 3
### Preparing Real-World Data for a Neural Network (Spaceship Titanic)

Prepare o dataset do Kaggle para uma rede com **tanh** nas camadas ocultas.

#### Instructions
- **Objetivo**: `Transported` (True/False).
- **Descrever colunas**: numéricas (`Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`) e categóricas (`HomePlanet`, `CryoSleep`, `Cabin`, `Destination`, `VIP`).
- **Faltantes**: quantifique e trate (mediana para numéricas; modo/`Unknown` para categóricas).
- **Codificação**: one-hot nas categóricas.
- **Escala**: padronize numéricas (**média 0, desvio 1**).
- **Visualize**: histogramas **antes/depois** (ex.: `Age`, `FoodCourt`).

---

## Evaluation Criteria

**Exercise 1 (3 pts)**
- 1 pt: dados gerados corretamente e scatter claro (rótulos/cores).
- 2 pts: análise de separabilidade e fronteiras plausíveis.

**Exercise 2 (3 pts)**
- 1 pt: dados 5D corretos.
- 1 pt: PCA→2D e gráfico claros.
- 1 pt: análise de não-linearidade e justificativa de MLP.

**Exercise 3 (4 pts)**
- 1 pt: descrição correta dos dados.
- 2 pts: *missing*, one-hot e **escala** apropriada p/ `tanh` (com justificativa).
- 1 pt: visualizações mostrando o efeito do pré-processamento.

---

## Notebook (executado)
> Versão HTML incorporada (gere com `nbconvert`).

<iframe src="ex1.html" width="100%" height="900" style="border:1px solid #ddd;"></iframe>

---

## Reprodutibilidade

```bash
# gerar HTML do notebook
jupyter nbconvert --to html --output ex1.html docs/exercises/1/ex1_pedrotpc.ipynb

# servir local
mkdocs serve

# publicar (branch gh-pages)
mkdocs gh-deploy --clean
