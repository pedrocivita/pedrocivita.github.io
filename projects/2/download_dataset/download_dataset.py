"""
Script para baixar o dataset Bike Sharing automaticamente.

Uso:
    python download_dataset.py

O script tentará baixar o dataset do UCI Repository.
"""

import urllib.request
import zipfile
import os
import sys

def download_dataset():
    """Baixa e extrai o dataset Bike Sharing do UCI Repository."""

    print("="*80)
    print("DOWNLOAD DO DATASET BIKE SHARING")
    print("="*80)

    # URL do dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
    zip_filename = "Bike-Sharing-Dataset.zip"
    target_file = "hour.csv"

    # Verificar se já existe
    if os.path.exists(target_file):
        print(f"\n[INFO] Arquivo '{target_file}' já existe!")
        response = input("Deseja baixar novamente? (s/n): ")
        if response.lower() != 's':
            print("[INFO] Download cancelado.")
            return True

    try:
        # Baixar arquivo
        print(f"\n[1/3] Baixando dataset de: {url}")
        print("[INFO] Aguarde, isso pode levar alguns segundos...")

        urllib.request.urlretrieve(url, zip_filename)
        print(f"[OK] Arquivo baixado: {zip_filename}")

        # Extrair arquivo
        print(f"\n[2/3] Extraindo arquivo ZIP...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(".")
        print(f"[OK] Arquivos extraídos")

        # Verificar se hour.csv existe
        print(f"\n[3/3] Verificando arquivo '{target_file}'...")
        if os.path.exists(target_file):
            file_size = os.path.getsize(target_file) / 1024  # KB
            print(f"[OK] Arquivo '{target_file}' encontrado ({file_size:.2f} KB)")

            # Remover ZIP (opcional)
            if os.path.exists(zip_filename):
                os.remove(zip_filename)
                print(f"[INFO] Arquivo ZIP removido")

            print("\n" + "="*80)
            print("DOWNLOAD CONCLUÍDO COM SUCESSO!")
            print("="*80)
            print(f"\nPróximo passo:")
            print(f"  Execute o notebook: jupyter notebook regression.ipynb")
            print("="*80)
            return True
        else:
            print(f"[ERRO] Arquivo '{target_file}' não encontrado após extração!")
            print(f"[INFO] Verifique o conteúdo extraído:")
            for item in os.listdir("."):
                if item.endswith(".csv"):
                    print(f"  - {item}")
            return False

    except urllib.error.URLError as e:
        print(f"\n[ERRO] Falha ao baixar o arquivo!")
        print(f"[INFO] Erro: {e}")
        print(f"\n[SOLUÇÃO] Baixe manualmente:")
        print(f"  1. Acesse: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset")
        print(f"  2. Baixe 'Bike-Sharing-Dataset.zip'")
        print(f"  3. Extraia e copie 'hour.csv' para esta pasta")
        return False

    except zipfile.BadZipFile:
        print(f"\n[ERRO] Arquivo ZIP corrompido!")
        print(f"[INFO] Tente baixar novamente ou use download manual")
        return False

    except Exception as e:
        print(f"\n[ERRO] Erro inesperado: {e}")
        print(f"\n[SOLUÇÃO] Tente download manual:")
        print(f"  https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset")
        return False

def show_dataset_info():
    """Mostra informações sobre o dataset."""

    target_file = "hour.csv"

    if not os.path.exists(target_file):
        print(f"\n[ERRO] Arquivo '{target_file}' não encontrado!")
        print(f"[INFO] Execute este script primeiro para baixar o dataset")
        return

    try:
        import pandas as pd

        print("\n" + "="*80)
        print("INFORMAÇÕES DO DATASET")
        print("="*80)

        df = pd.read_csv(target_file)

        print(f"\nDimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
        print(f"\nColunas:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")

        print(f"\nPrimeiras 5 linhas:")
        print(df.head())

        print(f"\nEstatísticas da variável target 'cnt':")
        print(f"  Média: {df['cnt'].mean():.2f}")
        print(f"  Desvio padrão: {df['cnt'].std():.2f}")
        print(f"  Mínimo: {df['cnt'].min():.0f}")
        print(f"  Máximo: {df['cnt'].max():.0f}")

        print("\n" + "="*80)

    except ImportError:
        print("\n[INFO] Instale pandas para ver informações detalhadas:")
        print("  pip install pandas")
    except Exception as e:
        print(f"\n[ERRO] Erro ao ler dataset: {e}")

if __name__ == "__main__":
    print("\nBike Sharing Dataset - Download Automático")
    print("UCI Machine Learning Repository\n")

    # Baixar dataset
    success = download_dataset()

    # Mostrar informações se sucesso
    if success:
        show_dataset_info()

    print("\n")
