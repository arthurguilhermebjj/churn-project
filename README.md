# Modelo de Previsão de Churn em Python

## Sobre o Projeto

Este projeto implementa um pipeline completo de Machine Learning para previsão de churn (cancelamento de clientes), utilizando Python e Scikit-Learn.

O objetivo é identificar clientes com maior probabilidade de cancelamento, permitindo que empresas adotem estratégias preventivas de retenção.

Todo o projeto foi desenvolvido seguindo boas práticas de Ciência de Dados, incluindo preparação dos dados, treinamento, validação, avaliação, serialização do modelo e inferência em novos clientes.

---

## Objetivos

- Realizar limpeza e preparação dos dados.
- Construir um pipeline completo de Machine Learning.
- Prever a probabilidade de churn de novos clientes.
- Segmentar clientes por nível de risco.
- Disponibilizar o modelo para utilização futura.

---

## Tecnologias Utilizadas

- Python 3.12
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Pickle

---

## Estrutura do Projeto

```
churn_project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── outputs/
│
├── images/
│
├── models/
│
├── src/
│   └── predict.py
│
├── churn_pipeline.py
├── requirements.txt
└── README.md
```

---

## Pipeline Desenvolvido

O projeto contempla todas as etapas de um fluxo real de Machine Learning:

1. Carregamento dos dados
2. Limpeza dos dados
3. Engenharia de atributos
4. Divisão em treino, validação e teste
5. Treinamento da Regressão Logística
6. Avaliação utilizando métricas de classificação
7. Curva ROC e AUC
8. Segmentação dos clientes
9. Salvamento do modelo com Pickle
10. Predição em novos clientes

---

## Resultados Obtidos

### Métricas no conjunto de teste

| Métrica | Valor |
|----------|---------|
| Accuracy | 0.3267 |
| Precision | 0.2295 |
| Recall | 0.9596 |
| F1-Score | 0.3704 |
| AUC | 0.6461 |

O modelo foi ajustado para maximizar o Recall, reduzindo a quantidade de clientes que realmente cancelariam e não seriam identificados.

---

## Segmentação dos Clientes

Foram definidos três grupos de risco:

| Segmento | Descrição |
|-----------|-----------|
| Críticos | Alta probabilidade de churn |
| Risco Médio | Necessitam acompanhamento |
| Estáveis | Baixa probabilidade de churn |

---

## Arquivos Gerados

O pipeline produz automaticamente:

- Modelo treinado (.pkl)
- Configuração do modelo (.json)
- Curva ROC
- Matriz de Confusão
- Importância das variáveis
- CSV com previsões
- CSV de segmentação
- Relatórios estatísticos

---

## Como Executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o treinamento:

```bash
python churn_pipeline.py
```

Realize previsões:

```bash
python src/predict.py --input arquivo.csv --output previsoes.csv
```

---

## Autor

**Arthur Guilherme**

Especialização em Análise de Dados e Inteligência Artificial

LinkedIn:

https://www.linkedin.com/in/arthurguilhermebjj