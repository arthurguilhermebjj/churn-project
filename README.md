PROJETO: MODELO DE PREVISÃO DE CHURN EM PYTHON

Desenvolvido por: Arthur Guilherme
Linguagem: Python 3.12

Dependências:
As bibliotecas necessárias para execução estão listadas no arquivo requirements.txt.

INSTRUÇÕES PARA RECRIAR O AMBIENTE:

1.Criar ambiente virtual:
python -m venv venv

2.Ativar o ambiente:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

3.Instalar as dependências:
pip install -r requirements.txt

EXECUÇÃO DO TREINAMENTO:

python churn_pipeline.py

EXECUÇÃO DE PREVISÃO EM NOVOS DADOS:

python src/predict.py --input caminho_do_csv --output caminho_saida_csv