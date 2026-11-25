# Previsão de Evasão Estudantil no Ensino Superior com Redes Neurais Multilayer Perceptron (MLP)

Projeto desenvolvido para a disciplina de **Sistemas Inteligentes (3ª Unidade)**.

## Objetivo

Treinar, validar e comparar redes neurais artificiais para **prever o risco de evasão estudantil no ensino superior**, utilizando como modelo principal uma **Multilayer Perceptron (MLP)** implementada em **PyTorch**, contrastada com uma rede neural de camada única (**Perceptron**) e uma **Regressão Logística** regularizada.

O projeto também avalia a **capacidade de detecção precoce** em três momentos da jornada do aluno (**na matrícula → após o 1º semestre → após o 2º semestre**), permitindo identificar em qual etapa a instituição de ensino já consegue agir de forma preventiva.

## Fontes oficiais do dataset

- **Dataset:** *Predict Students' Dropout and Academic Success* (`4.424` estudantes reais de 17 cursos de graduação, `36` atributos demográficos, socioeconômicos, de ingresso e de desempenho acadêmico, **`0` valores ausentes**).
- **Kaggle:** https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention
- **UCI Machine Learning Repository (ID 697 — download oficial automatizado):** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- **Artigo científico original:** Realinho, V.; Machado, J.; Baptista, L.; Martins, M. V. (2022). *Predicting Student Dropout and Academic Success*. **Data**, 7(11), 146 (Instituto Politécnico de Portalegre, Portugal).

## Estrutura do repositório

```text
.
├── data/
│   ├── raw/
│   │   └── student_dropout_raw.csv        # CSV oficial baixado do UCI/Kaggle
│   └── processed/
│       ├── student_dropout_binary.csv     # Desfecho final conhecido: Evasão vs. Formado (3.630 alunos)
│       ├── student_dropout_all.csv        # Base completa: Evasão vs. Não-Evasão (4.424 alunos)
│       └── dataset_metadata.json          # Auditoria completa de qualidade e estatísticas exploratórias
├── reports/
│   ├── experiment_results.json            # Métricas completas de todos os modelos e ablações
│   ├── RELATORIO_TECNICO.md               # Relatório fundamentado pronto para a apresentação
│   └── figures/                           # 5 gráficos em alta resolução prontos para os slides
├── src/
│   ├── config.py                          # URLs oficiais, tipagem das variáveis e estágios precoces
│   ├── collect_and_prepare.py             # Download oficial, limpeza, taxas acadêmicas e auditoria
│   └── train_and_evaluate.py              # Pré-processamento sem vazamento, PyTorch MLP, ablações e figuras
├── requirements.txt
└── README.md
```

## Como reproduzir

```bash
# 1. Criar ambiente virtual e instalar dependências
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Baixar, auditar e preparar o dataset oficial
python -m src.collect_and_prepare

# 3. Treinar modelos (Perceptron, LogReg e PyTorch MLP), rodar ablações e gerar gráficos
python -m src.train_and_evaluate
```

Consulte todos os fundamentos, tabelas e gráficos em [reports/RELATORIO_TECNICO.md](reports/RELATORIO_TECNICO.md).
