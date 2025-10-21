# Como Executar o Notebook de VAE

## 🚀 Opção 1: Google Colab (RECOMENDADA)

**Por que Colab?**
- ✅ Já vem com tudo instalado
- ✅ Sem problemas de DLL no Windows
- ✅ GPU grátis disponível
- ✅ Zero configuração

**Como usar:**
1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Clique em "Arquivo" → "Fazer upload de notebook"
3. Faça upload do `ex04_pedrotpc.ipynb`
4. Execute célula por célula!

## 💻 Opção 2: Local (se Colab não for possível)

### Windows

**Problema comum:** Erro de DLL (`c10.dll falhou`)

**Solução:**

1. Instale o Visual C++ Redistributable:
   - [Download aqui](https://aka.ms/vs/17/release/vc_redist.x64.exe)
   - Instale e reinicie o PC

2. Use Conda (mais confiável que pip no Windows):
```bash
# Criar ambiente
conda create -n vae python=3.10
conda activate vae

# Instalar PyTorch via conda
conda install pytorch torchvision cpuonly -c pytorch

# Instalar outras dependências
pip install matplotlib scikit-learn

# Executar Jupyter
jupyter notebook
```

**Ou com pip (se conda não funcionar):**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib scikit-learn jupyter
jupyter notebook
```

### Mac/Linux

Bem mais simples:
```bash
pip install torch torchvision matplotlib scikit-learn jupyter
jupyter notebook
```

## 📊 Tempo de Execução Estimado

- **2D VAE** (principal): ~3-5 minutos
- **Experimento com 3 dimensões** (2D, 10D, 20D): ~10-15 minutos
- **Total**: ~15-20 minutos

## 🎯 Checklist Antes de Submeter

- [ ] Todas as células executaram sem erro
- [ ] Gráficos foram gerados corretamente
- [ ] Reconstruções estão visíveis
- [ ] Espaço latente 2D foi plotado
- [ ] Geração de amostras funcionou
- [ ] Interpolação entre dígitos apareceu
- [ ] Comparação entre dimensões latentes completa
- [ ] Notebook exportado para GitHub Pages
- [ ] Link do GitHub Pages submetido no Blackboard

## 🆘 Problemas Comuns

### "No module named 'torch'"
→ PyTorch não instalado. Veja instruções acima ou use Colab.

### "DLL load failed" (Windows)
→ Instale Visual C++ Redistributable ou use Colab.

### "CUDA out of memory"
→ Não se preocupe, o código usa CPU por padrão.

### Notebook muito lento
→ Use Colab ou reduza `num_epochs` de 20 para 10-15.

## 📝 Próximos Passos

1. Execute o notebook completo
2. Verifique se todas as visualizações aparecem
3. Leia as análises e conclusões
4. Prepare-se para explicar:
   - Como funciona o reparameterization trick
   - Por que usamos KL divergence
   - Diferença entre VAE e autoencoder normal
5. Submeta no Blackboard antes de 26/out 23:59

Boa sorte! 🚀
