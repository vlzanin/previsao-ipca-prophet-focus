# previsao-ipca-prophet-focus
Código em Pyrhon utilizando Séries Temporais para realizar a projeção do índice IPCA.

Roda diretamente no Google Colab.

Previsão do IPCA com Prophet e expectativas do Focus

Projeto em Python para explorar previsões mensais do IPCA usando o Prophet, séries históricas do Banco Central do Brasil e expectativas de mercado do Relatório Focus para câmbio e Selic.

O código realiza a coleta de dados, prepara uma grade mensal, trata outliers do IPCA, ajusta hiperparâmetros por busca em grade, executa backtesting com janelas móveis, compara o Prophet com baselines simples e gera tabelas e gráficos.


Aviso sobre as expectativas Focus: a consulta automática usa as medianas anuais mais recentes disponíveis para Câmbio e Selic. Como essas expectativas são anuais, o código repete o valor de cada ano nos meses correspondentes. Isso é uma aproximação para fornecer covariáveis mensais ao modelo; não é uma trajetória mensal publicada pelo Banco Central nem uma previsão oficial.

Configurações principais

As opções ficam no início do arquivo:

•
data_inicial_str: início do período histórico.

•
meses_para_prever: número de meses do horizonte futuro.

•
meses_para_teste: duração do período usado no backtesting.

•
passo_backtest: intervalo entre origens do backtesting.

•
EXECUTAR_GRID_SEARCH: ativa ou desativa a busca de hiperparâmetros.

•
COMPARAR_MODOS: compara as configurações univariada, future_naive e future_real.

•
MODO_FINAL: escolhe o modo final ou usa auto para seleção automática.

•
TRATAR_OUTLIERS: ativa ou desativa o tratamento de outliers do IPCA.

•
N_JOBS: número de processos paralelos do joblib. -1 usa todos os núcleos disponíveis.

•
BUSCAR_FOCUS_AUTOMATICAMENTE: ativa a consulta automática das expectativas Focus.

Para usar projeções próprias, atribua a PROJECOES_COVARIAVEIS um DataFrame com as colunas ds, dolar e selic. A coluna ds deve conter as datas mensais. Quando projeções manuais forem fornecidas, elas substituem a consulta automática.

Modos de uso das covariáveis

•
univariada: usa apenas o histórico do IPCA.

•
future_naive: repete no futuro o último valor observado de cada covariável.

•
future_real: usa os valores futuros disponíveis na série e as projeções fornecidas para preencher os demais meses. No backtesting, esse modo pode usar informações futuras observadas; por isso é um cenário-oráculo, não uma avaliação operacional realista.

Interpretação dos resultados

O backtesting mede como o modelo teria se comportado em origens históricas. O script compara métricas como MAE, RMSE, WAPE e MASE e mostra intervalos de previsão e gráficos. Resultados históricos não garantem desempenho futuro e dependem da disponibilidade e qualidade das séries, das escolhas de modelagem e das hipóteses feitas para as covariáveis.

Este projeto tem finalidade educacional e exploratória. As saídas não são previsões oficiais do Banco Central nem recomendações econômicas ou financeiras.






