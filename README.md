# Pedro Toledo Piza Civita - Portfólio

[![MkDocs](https://img.shields.io/badge/MkDocs-Material-526CFE?logo=materialformkdocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-222222?logo=githubpages&logoColor=white)](https://pedrocivita.github.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Portfólio de projetos profissionais e acadêmicos desenvolvidos durante minha formação em Engenharia de Computação no Insper - Instituto de Ensino e Pesquisa.

**Site ao vivo:** [https://pedrocivita.github.io](https://pedrocivita.github.io)

## Sobre

Este repositório serve como meu portfólio principal no GitHub Pages, apresentando uma coleção dos meus trabalhos acadêmicos e profissionais. O site é construído com MkDocs Material, proporcionando uma documentação profissional e navegável dos meus projetos em diferentes áreas da computação.

## Projetos em Destaque

### Redes Neurais Artificiais e Deep Learning

Coleção completa de trabalhos teóricos e práticos explorando conceitos fundamentais e aplicações avançadas em redes neurais artificiais e aprendizado profundo, desenvolvidos durante o curso no Insper.

**Tópicos cobertos:**
- Pré-processamento de dados
- Perceptrons e Multi-Layer Perceptrons (MLPs)
- Autoencoders Variacionais (VAEs)
- Modelos de Classificação, Regressão e Generativos

## Tecnologias

Este portfólio utiliza um stack moderno baseado em Python:

- **MkDocs Material** - Geração de documentação profissional
- **Python 3.8+** - Linguagem de programação principal
- **NumPy, Pandas, Matplotlib** - Ciência de dados e visualização
- **Jupyter Notebook** - Ambiente de desenvolvimento interativo

### Plugins de Documentação

- **mkdocs-material** - Tema moderno Material Design
- **mkdocs-jupyter** - Integração com Jupyter notebooks
- **mkdocs-git-authors-plugin** - Atribuição de autoria
- **mkdocs-git-revision-date-localized-plugin** - Datas de revisão localizadas
- **mkdocs-glightbox** - Funcionalidade de lightbox para imagens
- **mkdocs-badges** - Geração dinâmica de badges

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip package manager
- Git

### Instruções de Configuração

1. Clone o repositório:

```bash
git clone https://github.com/pedrocivita/pedrocivita.github.io.git
cd pedrocivita.github.io
```

2. Crie um ambiente virtual Python:

```bash
python3 -m venv env
```

3. Ative o ambiente virtual:

```bash
# No Linux/macOS
source ./env/bin/activate

# No Windows
.\env\Scripts\activate
```

4. Instale as dependências:

```bash
python3 -m pip install -r requirements.txt --upgrade
```

## Uso

### Desenvolvimento Local

Para visualizar a documentação localmente com live reload:

```bash
mkdocs serve -o
```

A documentação estará disponível em `http://127.0.0.1:8000/`

### Deploy

Para fazer deploy da documentação no GitHub Pages:

```bash
mkdocs gh-deploy
```

Este comando compila o site e o envia para o branch `gh-pages`.

## Estrutura do Repositório

```
.
├── docs/                       # Arquivos fonte da documentação
│   ├── index.md               # Página inicial do portfólio
│   └── portfolio/             # Projetos do portfólio
│       └── neural-networks/   # Projeto de Redes Neurais
│           ├── exercises/     # Exercícios do curso (1-4)
│           └── projects/      # Projetos do curso (1-3)
├── mkdocs.yml                 # Configuração do MkDocs
├── requirements.txt           # Dependências Python
└── README.md                  # Este arquivo
```

## Autor

**Pedro Toledo Piza Civita**

Estudante de Engenharia de Computação no Insper - Instituto de Ensino e Pesquisa

- Email: pedrotpc@al.insper.edu.br
- LinkedIn: [Pedro Toledo Piza Civita](https://linkedin.com/in/pedro-toledo-piza-civita)
- GitHub: [@pedrocivita](https://github.com/pedrocivita)

## Contexto Acadêmico

**Instituição:** Insper - Instituto de Ensino e Pesquisa  
**Programa:** Engenharia de Computação  
**Período:** 2025.2

## Licença

Este projeto faz parte de trabalhos acadêmicos e é fornecido para fins educacionais. Por favor, consulte as políticas de integridade acadêmica da sua instituição antes de reutilizar qualquer conteúdo.

## Agradecimentos

- Corpo docente e funcionários do Insper pela orientação e suporte
- Comunidade open-source pelas excelentes ferramentas e bibliotecas
- Colegas de curso pela colaboração e compartilhamento de conhecimento

---

Para mais informações, visite a [documentação ao vivo](https://pedrocivita.github.io) ou entre em contato através dos canais listados acima.
