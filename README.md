# MIKAEL_IWAMOTO_DDF_SOLUTION_052024
# Case Técnico - Engenheiro de Soluções Junior/Trainee

## Item 1 - Sobre Solutions Engineering

Devido ao tempo curto e pouco conhecimento prévio de Arquitetura de dados não realizei o vídeo

## Item  2 - Sobre a Dadosfera

Tive um grande trabalho para entender a fonte de dados, nunca tive contato com um arquivo .jsonl .
Percebi que cada linha era um json especificando um produto.
Na plataforma da Dadosfera so aceita importar manualmente dados .csv , parquet e json e que possuem no max 250mb e os dados fornecidos é em outro formato e com 2,90GB
Dessa maneira fiquei estudando durante 2 dias a documentação da pipeline para importar arquivos.
Tentei, com codigo "comuncationdadosfera.py" e "testeconexao.py" (para pegar o token), fazer um upload utulizando a API da dadosfera mas obti o erro 413, request entity too large.
Depois de 2 dias nesse problema descobri que na propria fonte de dados ele fornece os mesmos dados em formato .parquet dividido em 4 partes.
Dessa maneira fiz o upload manualmente dos 4 arquivos e cataloguei-os.

![Imagem do dataset importado](img/dataimport.png)

## Item 3 - Sobre GenAI e LLMs

Tive dificuldade em transformar o dadaset fornecido (não estruturado, texto) para extruturado.
Fiquei confuso se era para realizar tal preprocessamento dos dados na propria plataforma da dadosfera ou codando.
Não consegui ver nenhuma ferramenta da dadosfera inbutida para conseguir fazer essa estração de features.
Procurando por fora da plataforma tentei, usando chatGPT, tentar criar alguma função que, a partir de um texto, retirar features.
Consegui fazer um funcao que preprocessa os textos e os transformar em dicionarios, (main.ipynb)

## Item  4 - Sobre SQL e Python

Pensando no todo, para não perder a data limite e demonstrar meus conhecimentos de SQL e python fiz um upload de um dataset "HOUSE_PRICE.csv".
A partir dele fiz um dashboard com algumas imagens relevantes para a analise do Dataset.

![Dashboard](img/dashboard_dadosfera.png)
![Query](img/query_dadosfera.png)

## Item  5 - Sobre Data Apps

Realizei o app com stremlit com 2 paginas.
- exploratory data analysis
- Prediction(Machine learning) Problema de regressão utilizando random forest, com r2 score, feature importance e grafico de dispersao

codigo: app.py

![Exploratory data analisys](img/app1.png)
![Histogram](img/app1.png)
![R2 score](img/app1.png)
![Grafico de Disperção](img/app1.png)