# Fraud Detection Project

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange?style=for-the-badge&logo=scikit-learn) ![Pandas](https://img.shields.io/badge/Pandas-Latest-green?style=for-the-badge&logo=pandas) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)

Sistema de detecção de fraudes utilizando técnicas de Machine Learning para identificar transações fraudulentas. Desenvolvido com foco em boas práticas de ciência de dados e análise preditiva.

**[Funcionalidades](#funcionalidades)** • **[Tecnologias](#tecnologias)** • **[Instalação](#instalação)** • **[Uso](#uso)** • **[Resultados](#resultados)** • **[Contribuir](#contribuindo)**

---

## Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Tecnologias](#tecnologias)
- [Pipeline de Machine Learning](#pipeline-de-machine-learning)
- [Instalação](#instalação)
- [Uso](#uso)
- [Resultados](#resultados)
- [Métricas de Avaliação](#métricas-de-avaliação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Contribuindo](#contribuindo)
- [Contato](#contato)

---

## Sobre o Projeto

**Fraud Detection Project** é uma solução de Machine Learning desenvolvida para identificar transações fraudulentas em datasets transacionais. O projeto aborda o desafio comum de **desbalanceamento de classes** e implementa técnicas de reamostragem para melhorar a detecção de fraudes.

### Principais Características

- **Pré-processamento Avançado** - Padronização de features com StandardScaler
- **Balanceamento de Classes** - Implementação de SMOTE (Synthetic Minority Over-sampling Technique)
- **Modelo de Classificação** - Regressão Logística com regularização L2
- **Validação Robusta** - Validação cruzada estratificada (5-fold)
- **Métricas Completas** - ROC AUC, Precision, Recall, F1-Score
- **Visualizações Interativas** - Curva ROC, Matriz de Confusão e gráficos de métricas
- **Notebook Interativo** - Desenvolvido em Jupyter/Google Colab

---

## Funcionalidades

### Pré-processamento de Dados

- **Análise Exploratória**

  - Verificação de dados nulos
  - Análise de dimensionalidade
  - Identificação de desbalanceamento de classes

- **Tratamento de Dados**
  - Separação de features e target
  - Split estratificado (80% treino, 20% teste)
  - Padronização com StandardScaler
  - Balanceamento com SMOTE

### Modelagem

- **Regressão Logística**

  - Regularização L2 para evitar overfitting
  - Validação cruzada estratificada (5-fold)
  - Otimização baseada em ROC AUC

- **Avaliação**
  - Matriz de confusão
  - Curva ROC
  - Classification Report (Precision, Recall, F1-Score)
  - AUC Score

### Visualizações

- **Análise de Desbalanceamento** - Distribuição de classes
- **Matriz de Confusão** - Visualização de predições
- **Curva ROC** - Avaliação do trade-off FPR/TPR
- **Métricas por Classe** - Gráficos comparativos de performance

---

## Tecnologias

### Core

| Tecnologia       | Versão | Descrição                |
| ---------------- | ------ | ------------------------ |
| Python           | 3.8+   | Linguagem de programação |
| Pandas           | Latest | Manipulação de dados     |
| NumPy            | Latest | Computação numérica      |
| Scikit-Learn     | Latest | Machine Learning         |
| Imbalanced-Learn | Latest | Técnicas de reamostragem |

### Visualização

| Tecnologia | Versão | Descrição                  |
| ---------- | ------ | -------------------------- |
| Matplotlib | Latest | Visualizações estáticas    |
| Seaborn    | Latest | Visualizações estatísticas |

### Ambiente

- **Google Colab** - Ambiente de desenvolvimento cloud
- **Jupyter Notebook** - Notebooks interativos

---

## Pipeline de Machine Learning

```
┌─────────────────┐
│  Raw Dataset    │
│  (fraud.csv)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Analysis   │
│ - Null check    │
│ - Shape         │
│ - Class balance │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│ - Train/Test    │
│ - StandardScale │
│ - SMOTE         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│ - Logistic Reg  │
│ - Cross Val     │
│ - L2 Penalty    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Evaluation     │
│ - Predictions   │
│ - Metrics       │
│ - Visualizations│
└─────────────────┘
```

---

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Jupyter Notebook ou Google Colab

### Instalação Local

#### 1. Clone o repositório

```bash
git clone https://github.com/nevesmarcos42/Fraud-Detection-Project.git
cd Fraud-Detection-Project
```

#### 2. Crie um ambiente virtual (recomendado)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Instale as dependências

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

#### 4. Inicie o Jupyter Notebook

```bash
jupyter notebook
```

### Usando Google Colab

1. Acesse o notebook diretamente: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nevesmarcos42/Fraud-Detection-Project/blob/main/FraudDetectionProject.ipynb)
2. Faça upload do dataset `fraud_dataset.csv`
3. Execute as células sequencialmente

---

## Uso

### Estrutura do Notebook

O notebook está organizado nas seguintes seções:

1. **Importação de Bibliotecas**

   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   ```

2. **Carregamento dos Dados**

   ```python
   df = pd.read_csv('fraud_dataset.csv')
   df.head()
   ```

3. **Análise Exploratória**

   - Verificação de valores nulos
   - Análise de dimensões
   - Distribuição de classes

4. **Pré-processamento**

   - Split de dados (80/20)
   - Padronização com StandardScaler
   - Balanceamento com SMOTE

5. **Treinamento do Modelo**

   - Regressão Logística
   - Validação cruzada 5-fold
   - Cálculo de ROC AUC

6. **Avaliação**
   - Predições
   - Métricas de classificação
   - Visualizações

### Exemplo de Uso

```python
# Carregar e preprocessar dados
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

X = df.drop('fraud', axis=1)
y = df['fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Padronizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balancear
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Treinar modelo
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2')
model.fit(X_train_resampled, y_train_resampled)

# Avaliar
y_pred = model.predict(X_test_scaled)
```

---

## Resultados

### Performance do Modelo

O modelo de Regressão Logística apresentou excelente performance na detecção de fraudes:

| Métrica       | Classe 0 (Normal) | Classe 1 (Fraude) |
| ------------- | ----------------- | ----------------- |
| **Precision** | 0.99              | 0.58              |
| **Recall**    | 0.93              | 0.95              |
| **F1-Score**  | 0.96              | 0.72              |

### ROC AUC Score

- **AUC Score**: ~0.94
- Indica excelente capacidade de discriminação entre classes

### Interpretação

✅ **Pontos Fortes:**

- Alto recall na classe de fraude (95%) - captura a maioria das fraudes
- Excelente performance geral (AUC ~0.94)
- Baixa taxa de falsos negativos

⚠️ **Pontos de Atenção:**

- Precisão moderada na classe de fraude (58%) - alguns falsos positivos
- Trade-off aceitável para cenários onde capturar fraudes é prioridade

### Conclusão

O modelo apresenta um **bom desempenho geral**, com destaque para o **alto recall na classe de fraude**. Isso é especialmente relevante em cenários onde capturar fraudes é prioridade, mesmo ao custo de alguns falsos positivos. Há espaço para melhorias na precisão da classe de fraude, possivelmente com:

- Otimização de hiperparâmetros
- Métodos adicionais de reamostragem
- Engenharia de features
- Ensemble de modelos

---

## Métricas de Avaliação

### Matriz de Confusão

A matriz de confusão visualiza as predições corretas e incorretas do modelo:

- **Verdadeiros Negativos (TN)**: Transações normais corretamente identificadas
- **Falsos Positivos (FP)**: Transações normais classificadas como fraude
- **Falsos Negativos (FN)**: Fraudes não detectadas
- **Verdadeiros Positivos (TP)**: Fraudes corretamente identificadas

### Curva ROC

A curva ROC demonstra o trade-off entre Taxa de Verdadeiros Positivos (TPR) e Taxa de Falsos Positivos (FPR). Uma AUC próxima de 1.0 indica excelente performance.

### Classification Report

- **Precision**: Proporção de predições positivas corretas
- **Recall**: Proporção de casos positivos corretamente identificados
- **F1-Score**: Média harmônica entre precision e recall

---

## Estrutura do Projeto

```
Fraud-Detection-Project/
├── FraudDetectionProject.ipynb   # Notebook principal
├── README.md                       # Documentação
└── fraud_dataset.csv              # Dataset (não incluído no repositório)
```

### Dataset

O dataset deve conter:

- Features numéricas representando características das transações
- Coluna `fraud` (target): 0 = Normal, 1 = Fraude

**Nota**: O dataset não está incluído no repositório. Você pode usar seus próprios dados ou datasets públicos de detecção de fraudes.

---

## Contribuindo

Contribuições são bem-vindas! Siga os passos:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

### Padrões de Código

- Seguir convenções PEP 8 para Python
- Documentar funções e classes
- Adicionar comentários explicativos
- Manter notebooks organizados e limpos
- Incluir visualizações para facilitar interpretação

### Sugestões de Melhorias

- [ ] Testar outros modelos (Random Forest, XGBoost, Neural Networks)
- [ ] Implementar GridSearch para otimização de hiperparâmetros
- [ ] Adicionar análise de feature importance
- [ ] Criar pipeline automatizado com sklearn.pipeline
- [ ] Implementar validação temporal (time-series split)
- [ ] Adicionar testes unitários
- [ ] Criar API para servir o modelo

---

## Contato

**Marcos Neves**

📧 Email: nevesmarcos42@gmail.com

💼 LinkedIn: [linkedin.com/in/nevesmarcos](https://www.linkedin.com/in/nevesmarcos/)

🐱 GitHub: [github.com/nevesmarcos42](https://github.com/nevesmarcos42)

---

## Sobre

Sistema de detecção de fraudes utilizando Machine Learning com técnicas avançadas de balanceamento de classes e validação cruzada. Implementa Regressão Logística com regularização L2 e SMOTE para otimizar a detecção de transações fraudulentas.

**Versão**: 1.0.0

**Última Atualização**: Novembro 2025

---

**Desenvolvido como projeto de estudo em Machine Learning e Ciência de Dados** 🚀
