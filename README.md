# Credit Card Fraud Detection
![Design sem nome](https://github.com/Gustavo-michel/Credit-Card-Fraud-Detection/assets/127684360/ced5f9b2-4664-47d7-8b3c-b297f90d0f90)

## Sobre
Este projeto visa desenvolver um sistema de detecção de fraudes em cartões de crédito, utilizando técnicas de análise exploratoria de dados e machine learning o dataset utilizado neste projeto contém transações de cartões de crédito, onde cada observação é rotulada como fraude ou não fraude. As variáveis presentes no dataset:
- V1... V28 - São dados mascarados por segurança, pelo autor da base de dados.
- Time - Intervalo de tempo entre cada transição e ordena os dados como indice.
- Amount - Quantia da transação.
- Class - valor de previsão 0 siginfica transação legitima e 1 siginifica fraude.

## Análise dos Dados
Para a análise dos dados, foram realizadas as seguintes etapas:

1. Limpeza e Preparação dos Dados: Remoção de valores ausentes e duplicados.
2. Exploração de Dados: Análise estatística descritiva para entender a distribuição das variáveis e identificar possíveis padrões ou discrepâncias.
3. Balanceamento de Classes: Consideração do desbalanceamento entre as classes de fraude e não fraude para evitar viés no modelo, utilizado estrategia de Undersampling.
4. modelagem dos dados: Pré-processamento e criação do modelo preditivo.

## Insights
Alguns insights importantes obtidos durante a análise dos dados incluem:

* Media de transação de 88$.
* 284807 linhas de dados.
* Previsores já pré-padronizados.
* Dataframe com valores de fraudes desequilibrados.
* Sem valores nulos.
* Correlação de mais peso da nossa classe são [V11 V4 V2 V21 V19 V20 V8 V27 V28 ] seguido do amount.
* Valores dupicados achados, terão que ser tratados.
* A maioria das transferencias fraudulentas tem o valor baixo de $10.
* A maioria dos valores apresentado nos dados são abaixo de $500.
* Distribuição irregular das variaveis de X.
* Dados com maior coorelação diante da Classe tendem a ter menos outliers.

## Insights Gráficos
A visualização dos dados proporcionou insights valiosos, incluindo:
![Verificar a dispersão dos dados de acordo com a classe](https://github.com/Gustavo-michel/Credit-Card-Fraud-Detection/assets/127684360/11d726ba-1e64-4587-ae57-bd1faf20ae31  "Matriz de dispersão")

![Distribuição da coluna Amount](https://github.com/Gustavo-michel/Credit-Card-Fraud-Detection/assets/127684360/08623c80-2cd4-474f-a924-32ae8df39d12 "Distribuição da quantia de transferencia")

![Detecção dos valores dispersos no DataFrame](https://github.com/Gustavo-michel/Credit-Card-Fraud-Detection/assets/127684360/8f36710c-66fd-4413-91ab-b96a859b51b8 "Detectecção de outliers boxplot")

![Verificando correlação pelo Matriz de confusão](https://github.com/Gustavo-michel/Credit-Card-Fraud-Detection/assets/127684360/d7eb5ded-6360-40ee-b592-d43d55109424 "Correlação heatmap")

![PCA-view](https://github.com/Gustavo-michel/Credit-Card-Fraud-Detection/assets/127684360/ce54bf04-1f7b-402b-aeeb-e306c106200c)

## Construção do modelo

![Variancethreshold](https://github.com/Gustavo-michel/Credit-Card-Fraud-Detection/assets/127684360/29a19cd9-e9db-423d-b7f9-943e42bf422c)

![](https://github.com/Gustavo-michel/Credit-Card-Fraud-Detection/assets/127684360/bb542318-8647-4953-9fba-cb7dbbd2594b "Comparando modelos ROC-CURVE")

## Acesso ao Código-Fonte
O código-fonte está disponível no repositório do GitHub. Para instalar as dependências necessárias, execute o seguinte comando:

```python
pip install -r requirements.txt
```
No app.py execute o codigo no servidor.

## Acesso ao Modelo
O modelo treinado pode ser acessado através do site oficial do projeto. Para carregar o modelo usando a biblioteca pickle em Python, utilize o seguinte código:

```python
import pickle

# Carregar o modelo
with open('modelo.pkl', 'rb') as file:
    model = pickle.load(file)
# Utilizar o modelo para fazer previsões
resultado = model.predict(dados)
```

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request ao repositório.

DashBoard com power BI estará presente no projeto em breve.
