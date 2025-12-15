# Portfólio - Pedro Toledo Piza Civita

[![MkDocs](https://img.shields.io/badge/MkDocs-Material-526CFE?logo=materialformkdocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-222222?logo=githubpages&logoColor=white)](https://pedrocivita.github.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Portfólio de projetos profissionais e acadêmicos desenvolvidos durante a graduação em Engenharia da Computação no Insper - Instituto de Ensino e Pesquisa.

**Live Documentation:** [https://pedrocivita.github.io](https://pedrocivita.github.io)

## Sobre

Este repositório serve como meu portfólio principal no GitHub Pages, reunindo projetos e trabalhos desenvolvidos ao longo da graduação em Engenharia da Computação no Insper. O conteúdo está organizado como um site de documentação profissional construído com MkDocs Material.

## Projetos em Destaque

### Redes Neurais Artificiais e Deep Learning

Portfólio completo de projetos e exercícios do curso de Redes Neurais Artificiais e Deep Learning, abrangendo:

- Preparação e análise de dados
- Implementação de Perceptron
- Redes Neurais Multi-Camada (MLP)
- Autoencoders Variacionais (VAE)
- Projetos de Classificação, Regressão e Modelos Generativos

### Futuros Projetos

Este portfólio será expandido com novos projetos das seguintes áreas:

- Desenvolvimento de Software
- Sistemas Embarcados
- Arquitetura de Computadores
- Banco de Dados
- E muito mais...

## Tecnologias Utilizadas

### Documentação

- **MkDocs Material** - Theme moderno com Material Design
- **mkdocs-jupyter** - Integração com Jupyter notebooks
- **mkdocs-glightbox** - Lightbox para imagens
- **Diversos plugins** - Para funcionalidades avançadas

### Projetos

As tecnologias variam conforme o projeto. Consulte a documentação de cada projeto para detalhes específicos.

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip package manager
- Git

### Instruções de Setup

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

Este comando constrói o site e o publica no branch `gh-pages`.

## Estrutura do Repositório

```
.
├── docs/                           # Arquivos fonte da documentação
│   ├── index.md                   # Homepage do portfólio
│   └── neural-networks/           # Projeto de Redes Neurais
│       ├── index.md              # Página inicial do projeto
│       ├── exercises/            # Exercícios do curso (1-4)
│       └── projects/             # Projetos do curso (1-3)
├── mkdocs.yml                     # Configuração do MkDocs
├── requirements.txt               # Dependências Python
└── README.md                      # Este arquivo
```

## Autor

**Pedro Toledo Piza Civita**

Estudante de Engenharia da Computação no Insper - Instituto de Ensino e Pesquisa

- Email: pedrotpc@al.insper.edu.br
- LinkedIn: [Pedro Toledo Piza Civita](https://linkedin.com/in/pedro-toledo-piza-civita)
- GitHub: [@pedrocivita](https://github.com/pedrocivita)

## Contexto Acadêmico

**Instituição:** Insper - Instituto de Ensino e Pesquisa  
**Programa:** Engenharia da Computação  
**Período:** 2022 - 2026

## Licença

Este projeto faz parte de trabalhos acadêmicos e é fornecido para fins educacionais. Por favor, consulte as políticas de integridade acadêmica de sua instituição antes de reutilizar qualquer conteúdo.

## Agradecimentos

- Professores e equipe do Insper pelo apoio e orientação
- Comunidade open-source pelas excelentes ferramentas e bibliotecas
- Colegas de curso pela colaboração e compartilhamento de conhecimento

---

Para mais informações, visite a [documentação live](https://pedrocivita.github.io) ou entre em contato através dos canais listados acima.
