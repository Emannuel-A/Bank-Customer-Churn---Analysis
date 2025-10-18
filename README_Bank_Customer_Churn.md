```python

```

# PROJETO DE ANÁLISE EM UM DATASET DE UM BANCO

### Sumário da Análise

1.  **Introdução**
    * 1.1. Problema de Negócio: O Custo do Churn
    * 1.2. Objetivos da Análise
2.  **Configuração do Ambiente e Dados**
    * 2.1. Importação das Bibliotecas
    * 2.2. Carregamento e Verificação Inicial do Dataset
3.  **Análise Exploratória de Dados (EDA)**
    * 3.1. Análise Univariada: Quem são os nossos clientes?
    * 3.2. Análise Multivariada: Que fatores influenciam o Churn?
4.  **Conclusões e Recomendações Estratégicas**
    * 4.1. Perfil do Cliente com Risco de Churn
    * 4.2. Recomendações Acionáveis
    * 4.3. Próximos Passos




# Análise de Fatores de Churn em Clientes Bancários

## 1. Introdução

### 1.1. Problema de Negócio: O Custo do Churn
A perda de clientes (*churn*) representa um desafio significativo para a sustentabilidade da nossa instituição. Com uma taxa de cancelamento de **20.4%** na base de dados analisada, é crucial identificar as causas para mitigar a perda de receita e os custos associados à aquisição de novos clientes.

### 1.2. Objetivos da Análise
Esta Análise Exploratória de Dados (EDA) visa:
1.  Identificar o perfil demográfico e comportamental dos clientes com maior propensão ao churn.
2.  Quantificar o impacto de variáveis-chave no risco de cancelamento.
3.  Fornecer *insights* para a criação de estratégias de retenção eficazes e direcionadas.


## 2. Configuração do Ambiente e Carregamento de Dados

Nesta etapa, preparamos o ambiente de trabalho importando as bibliotecas essenciais e carregando o dataset para análise.



```python
# Comentário para a sua célula de código:

# 2.1. Importação de Bibliotecas
# pandas: para manipulação e análise de dados.
# numpy: para operações numéricas.
# matplotlib.pyplot e seaborn: para visualização de dados.
import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

# Sistema
import os

# Configurações de estilo para os gráficos e dataset
plt.style.use('ggplot')
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 40)

# 2.2. Carregamento do Dataset
# Define o Kaggle dataset identificador
dataset_id = 'gauravtopre/bank-customer-churn-dataset'
# Download do dataset
dataset_path = kagglehub.dataset_download(dataset_id)
downloaded_files = os.listdir(dataset_path)
print("Files downloaded:")
print(downloaded_files)

csv_file_name = downloaded_files[0] if downloaded_files else 'Bank Customer Churn Prediction.csv'
csv_file_path = os.path.join(dataset_path, csv_file_name)

# Leitura do dataset com o Pandas
df = pd.read_csv(csv_file_path)

# Verificação inicial dos dados carregados
print("Dataset carregado com sucesso!")
display(df.head())
```

    Using Colab cache for faster access to the 'bank-customer-churn-dataset' dataset.
    Files downloaded:
    ['Bank Customer Churn Prediction.csv']
    Dataset carregado com sucesso!




  <div id="df-6c1c5135-3ea9-434f-8bff-476ce7e4e577" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>credit_score</th>
      <th>country</th>
      <th>gender</th>
      <th>age</th>
      <th>tenure</th>
      <th>balance</th>
      <th>products_number</th>
      <th>credit_card</th>
      <th>active_member</th>
      <th>estimated_salary</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15634602</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15647311</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15619304</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15701354</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15737888</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6c1c5135-3ea9-434f-8bff-476ce7e4e577')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6c1c5135-3ea9-434f-8bff-476ce7e4e577 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6c1c5135-3ea9-434f-8bff-476ce7e4e577');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-d6e69eac-3f9d-4533-90b6-1cd4dd9a4875">
      <button class="colab-df-quickchart" onclick="quickchart('df-d6e69eac-3f9d-4533-90b6-1cd4dd9a4875')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-d6e69eac-3f9d-4533-90b6-1cd4dd9a4875 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



## 3. Análise Exploratória de Dados (EDA)

### 3.1. Análise Univariada: Quem são os nossos clientes?

Antes de analisar o churn, é fundamental compreender a composição da nossa base de clientes.


```python
# Sempre importante enteder como o dataset está distribuido
# quais são os valores unicos por colunas

df.nunique().sort_values(ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>customer_id</th>
      <td>10000</td>
    </tr>
    <tr>
      <th>estimated_salary</th>
      <td>9999</td>
    </tr>
    <tr>
      <th>balance</th>
      <td>6382</td>
    </tr>
    <tr>
      <th>credit_score</th>
      <td>460</td>
    </tr>
    <tr>
      <th>age</th>
      <td>70</td>
    </tr>
    <tr>
      <th>tenure</th>
      <td>11</td>
    </tr>
    <tr>
      <th>products_number</th>
      <td>4</td>
    </tr>
    <tr>
      <th>country</th>
      <td>3</td>
    </tr>
    <tr>
      <th>gender</th>
      <td>2</td>
    </tr>
    <tr>
      <th>credit_card</th>
      <td>2</td>
    </tr>
    <tr>
      <th>active_member</th>
      <td>2</td>
    </tr>
    <tr>
      <th>churn</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



### Análise da Distribuição de Idade


```python
# sobre as idades, sempre importante verificar a extensão do banco
print("Min age:", df['age'].min())
print("Max age:", df['age'].max())
```

    Min age: 18
    Max age: 92



```python
# Vou fazer uma visão geral sobre as idades dos Clientes
df['age'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.921800</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.487806</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>44.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>92.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>




```python
# Análise da distribuição da variável numérica 'age' (Idade).
plt.hist(df['age'], bins=100)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Calculate mean and median para incluir na vizualização
mean_age = df['age'].mean()
median_age = df['age'].median()

# Agora, para melhorar a vizualização, adicionei duas linhas para identificar a média e a mediana
plt.axvline(mean_age, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_age:.2f}')
plt.axvline(median_age, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_age:.2f}')

plt.legend()
plt.show()
```


    
![png](output_10_0.png)
    


O histograma da distribuição de idade dos clientes mostra uma distribuição aproximadamente normal, com a maioria dos clientes concentrada em torno da média e da mediana. A média das idades está em aproximadamente **39** anos e a mediana em **37** anos, indicando que o centro da distribuição está na faixa dos 30 a 40 anos.

A forma da distribuição parece ser ligeiramente assimétrica à direita (positivamente assimétrica), com uma cauda se estendendo para idades mais altas, embora a maior concentração de clientes esteja na faixa-etária mais jovem. A presença de alguns clientes com idades mais elevadas (visíveis na cauda direita do histograma) indica uma variabilidade na faixa etária dos clientes.

Ou seja, os clientes do banco estão entre os mais jovem a de meia-idade, o que pode ser um fator importante a ser considerado em análises futuras, especialmente ao investigar o perfil de clientes que cancelaram suas contas(churn).

### Distribuição dos Clientes por País


```python
country_counts = df['country'].value_counts().reset_index()
country_counts.columns = ['country', 'count']
country_counts['relativ'] = country_counts['count'] / country_counts['count'].sum()
print("Count of customers per country:")
display(country_counts)
```

    Count of customers per country:




  <div id="df-35e75292-0046-4704-9fea-b6425e6d3ce5" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>count</th>
      <th>relativ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>France</td>
      <td>5014</td>
      <td>0.5014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Germany</td>
      <td>2509</td>
      <td>0.2509</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spain</td>
      <td>2477</td>
      <td>0.2477</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-35e75292-0046-4704-9fea-b6425e6d3ce5')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-35e75292-0046-4704-9fea-b6425e6d3ce5 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-35e75292-0046-4704-9fea-b6425e6d3ce5');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-79d8e085-08ab-4073-a74d-7c14bab9a637">
      <button class="colab-df-quickchart" onclick="quickchart('df-79d8e085-08ab-4073-a74d-7c14bab9a637')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-79d8e085-08ab-4073-a74d-7c14bab9a637 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_ce49898c-296f-4dff-a378-45e29865f15f">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('country_counts')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_ce49898c-296f-4dff-a378-45e29865f15f button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('country_counts');
      }
      })();
    </script>
  </div>

    </div>
  </div>




```python
plt.bar(country_counts['country'], country_counts['count'])
plt.title('Count of Customers per Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_14_0.png)
    


Como pode-se notar, pouco mais de 50% dos clientes da base do banco estão na França, enquanto Alemanha e Espanha tem a quantidade relativamente igual de clientes.

### Distribuição por Genênro


```python
#Para enteder melhor a distribuição por genêro vou sumarizar os dados
gender_counts = df['gender'].value_counts().reset_index()
gender_counts.columns = ['gender', 'count']
gender_counts['relativ'] = gender_counts['count'] / gender_counts['count'].sum()
print("Count of customers per gender:")
display(gender_counts)
```

    Count of customers per gender:




  <div id="df-1b21a12d-4b96-4cce-861d-d62fc3855cfc" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>count</th>
      <th>relativ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>5457</td>
      <td>0.5457</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>4543</td>
      <td>0.4543</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1b21a12d-4b96-4cce-861d-d62fc3855cfc')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1b21a12d-4b96-4cce-861d-d62fc3855cfc button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1b21a12d-4b96-4cce-861d-d62fc3855cfc');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-4a3ad239-dcc3-406e-8e3f-187783043a57">
      <button class="colab-df-quickchart" onclick="quickchart('df-4a3ad239-dcc3-406e-8e3f-187783043a57')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-4a3ad239-dcc3-406e-8e3f-187783043a57 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_c91c072c-1939-4a92-82a3-1297f4218ea8">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('gender_counts')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_c91c072c-1939-4a92-82a3-1297f4218ea8 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('gender_counts');
      }
      })();
    </script>
  </div>

    </div>
  </div>




```python
# Um gráfico para melhor entender a distribuição por genero.
plt.bar(gender_counts['gender'], gender_counts['count'])
plt.title('Count of Customers per Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_18_0.png)
    


54% dos clientes são do genêro masculino, um dado interessante, pois segundo o site https://countrymeters.info/pt/Europe a população europeia se divide em População masculina atual (48.2%) População feminina atual (51.8%)(estimativa 2025), o que mostra uma ligeira discrepancia com a estimativa populacional real.

### Clientes ativo/inativos


```python
# Quantidade de Membros ativos
df['active_member'].value_counts()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>active_member</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5151</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4849</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



### Perfil Salarial


```python
# Pra entender melhor o perfil de clientes, vamos agrupar por estimated_salary
display(df['estimated_salary'].describe())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estimated_salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>100090.239881</td>
    </tr>
    <tr>
      <th>std</th>
      <td>57510.492818</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.580000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>51002.110000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>100193.915000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>149388.247500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>199992.480000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



```python
# Criei ranges de salários usando como base o numero de clientes nessa amostras, que é igual a 10k, ou seja
# 4 grupos de 2500 pessoas
df['salary_range'] = pd.qcut(df['estimated_salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# agora mostro como ficou o agrupamento de maneira mais fácil
print("\nCounts of customers in each salary range:")
display(df['salary_range'].value_counts())
```

    
    Counts of customers in each salary range:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>salary_range</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Q1</th>
      <td>2500</td>
    </tr>
    <tr>
      <th>Q2</th>
      <td>2500</td>
    </tr>
    <tr>
      <th>Q3</th>
      <td>2500</td>
    </tr>
    <tr>
      <th>Q4</th>
      <td>2500</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



```python
# Mostrando na prática como ficou a distribuição, para entedermos melhor.
salary_range_stats = df.groupby('salary_range', observed=True)['estimated_salary'].agg(['min', 'max', 'mean'])
salary_range_stats.columns = ['Min Salary', 'Max Salary', 'Mean Salary']
print("Estimated Salary Statistics per Salary Range:")
display(salary_range_stats)
```

    Estimated Salary Statistics per Salary Range:




  <div id="df-67ea374a-02c6-48f5-b4f5-5ae1bd7de909" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Min Salary</th>
      <th>Max Salary</th>
      <th>Mean Salary</th>
    </tr>
    <tr>
      <th>salary_range</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Q1</th>
      <td>11.58</td>
      <td>50974.57</td>
      <td>25407.102916</td>
    </tr>
    <tr>
      <th>Q2</th>
      <td>51011.29</td>
      <td>100187.43</td>
      <td>75420.523720</td>
    </tr>
    <tr>
      <th>Q3</th>
      <td>100200.40</td>
      <td>149384.43</td>
      <td>124553.795324</td>
    </tr>
    <tr>
      <th>Q4</th>
      <td>149399.70</td>
      <td>199992.48</td>
      <td>174979.537564</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-67ea374a-02c6-48f5-b4f5-5ae1bd7de909')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-67ea374a-02c6-48f5-b4f5-5ae1bd7de909 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-67ea374a-02c6-48f5-b4f5-5ae1bd7de909');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-95db53cf-4dfa-4227-87cd-2108591fab38">
      <button class="colab-df-quickchart" onclick="quickchart('df-95db53cf-4dfa-4227-87cd-2108591fab38')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-95db53cf-4dfa-4227-87cd-2108591fab38 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_ef2de013-4c43-47af-be16-e6c351b9b98b">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('salary_range_stats')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_ef2de013-4c43-47af-be16-e6c351b9b98b button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('salary_range_stats');
      }
      })();
    </script>
  </div>

    </div>
  </div>




```python
# agora para facilitar o entendimento, vamos vizualizar com um gráfico
plt.bar(salary_range_stats.index, salary_range_stats['Mean Salary'])
plt.title('Salary Range')
plt.xlabel('Salary Range')
plt.ylabel('Mean Estimated Salary')

# Para ter uma vizualização rápida adicionei o min, max e média no topo das barras.
for i, row in salary_range_stats.iterrows():
    min_salary = row['Min Salary']
    max_salary = row['Max Salary']
    mean_salary = row['Mean Salary']
    plt.text(i, mean_salary + 1000,
             f'Min: {min_salary:.2f}\nMax: {max_salary:.2f}\nMean: {mean_salary:.2f}',
             ha='center', va='bottom', fontsize=8) # Added a small offset and reduced font size
plt.ylim(0, salary_range_stats['Mean Salary'].max() * 1.14)

plt.show()
```


    
![png](output_26_0.png)
    


### Distribuição de salário
A maior grupo de clientes do Banco tem um salário maior que 149 mil euros por ano, ou seja, são clientes que tem um bom poder aquisitivo(De acordo com o Eurostat, em 2023, o salário médio anual ajustado a tempo inteiro por trabalhador variou entre 13 503 euros na Bulgária e 81 064 euros no Luxemburgo, com a média da UE a situar-se em 37 863 euros. link: https://pt.euronews.com/business/2024/12/24/classificacao-dos-salarios-medios-na-europa-quais-sao-os-paises-que-pagam-mais), enquanto o menor grupo tem uma média de 25 mil euros/ano.
Podemos fazer análise interessantes cruzando com a questão salárial, como: Qual gênero mais predominate por faixa, qual a media de idade de cada faixa, e posterior a isso, quando formos fazer a analise de churn, saber se existe algum desses grupos que se sobressai.


```python
# Vou criar os gráficos para enteder melhor a dritribuição dos salários
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Salary Range by Gender
sns.countplot(data=df, x='salary_range', hue='gender', ax=axes[0], width=0.4)
axes[0].set_title('Distribution of Gender by Salary Range')
axes[0].set_xlabel('Salary Range')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='Gender')

gender_counts = df.groupby(['salary_range', 'gender'], observed=True).size().unstack(fill_value=0)
gender_proportions = gender_counts.apply(lambda x: x / x.sum(), axis=1)

for i, salary_range in enumerate(gender_counts.index):
    for j, gender in enumerate(gender_counts.columns):
        count = gender_counts.loc[salary_range, gender]
        proportion = gender_proportions.loc[salary_range, gender]
        x_offset = (j - len(gender_counts.columns) / 2 + 0.5) * axes[0].patches[0].get_width() / axes[0].patches[0].get_width() * 0.2 # Adjust offset based on the patch width and number of groups
        x_position = i + x_offset
        y_position = count

        if count > 0:
             axes[0].text(x_position, y_position,
                          f'{count}\n({proportion:.1%})',
                          ha='center', va='bottom', fontsize=8)


# Plot 2: Salary Range by Age (using boxplot to show distribution)
sns.boxplot(data=df, x='salary_range', y='age', ax=axes[1])
axes[1].set_title('Distribution of Age by Salary Range')
axes[1].set_xlabel('Salary Range')
axes[1].set_ylabel('Age')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()
```


    
![png](output_28_0.png)
    


Os gráficos revelam que a distribuição de clientes por Gênero é ligeiramente tendenciosa para o Masculino na maioria das faixas salariais (Q1, Q2, Q4). Crucialmente, a Idade dos clientes é uniformemente distribuída e não se correlaciona com a Faixa Salarial, pois as medianas e dispersões de idade são praticamente idênticas em todos os quartis de salário.

### Saldo Bancário


```python
# vamos verificar os clientes que não tem saldo em conta
print('Quantidade de clientes com saldo 0 é:',(df['balance'] == 0).sum())
print('Quantidade de clientes com saldo positivo > 0 é:',(df['balance'] > 0).sum())
print('Quantidade de clientes com saldo negativo é:',(df['balance'] < 0).sum())
# Agora vamos a média de saldo para os clientes que tem saldo em conta
positive_balance_df = df[df['balance'] > 0]

# Calcula a média do dataframe
mean_positive_balance = positive_balance_df['balance'].mean()
print(f"A média do saldo dos clientes(contando apenas os que tem saldo maior que zero) é: {mean_positive_balance:.2f}")
```

    Quantidade de clientes com saldo 0 é: 3617
    Quantidade de clientes com saldo positivo > 0 é: 6383
    Quantidade de clientes com saldo negativo é: 0
    A média do saldo dos clientes(contando apenas os que tem saldo maior que zero) é: 119827.49



```python
positive_balance_df.hist(column='balance', bins=100)
plt.title('Distribution of Positive Balance') # Changed title for clarity
plt.xlabel('Balance')
plt.ylabel('Frequency')

# Calculo da média e mediana para os cliente com saldo maior que zero
mean_positive_balance = positive_balance_df['balance'].mean()
median_positive_balance = positive_balance_df['balance'].median()

# Linhas verticais de media e mediana
plt.axvline(mean_positive_balance, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_positive_balance:.2f}')
plt.axvline(median_positive_balance, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_positive_balance:.2f}')

plt.legend()
plt.show()
```


    
![png](output_32_0.png)
    


### Distribuição do Saldo Bancário


Podemos notar que as linhas de média e mediana estão praticamento no mesmo lugar, isso indica que o dado está bastante simétrico, em linhas gerais, a distribuição de saldo está equilibrada, não há grandes saldos gigantescos que destoam e elevam a média para cima, lembrando que eu retirei os clientes sem saldo dessa vizualização, para entendermos melhor os clientes que tem dinheiro guardado, mesmo que provisóriamente.

### 3.2. Análise Multivariada: Que fatores influenciam o Churn?

Agora, que ja exploramos um pouco nosso dataset, entendemos de forma geral o perfil dos clientes, vamos as combinações para uma análise mais profunda dos clientes e entender se há alguma correlação com o churn(lembrando que correlação não implica necessáriamente em causalidade)

Nesta secção, cruzarei as variáveis demográficas e comportamentais com a variável alvo (`churn`) para identificar os perfis com maior risco de cancelamento.


```python
# aqui irei plotar um gráfico que tentar achar uma correlação entre salario e saldo bancário segregando por país.
sns.relplot(x = 'estimated_salary', y = 'balance', data = df, col='country', color='b')
```




    <seaborn.axisgrid.FacetGrid at 0x7d8fe1b75f40>




    
![png](output_35_1.png)
    


É possivel notar que não há correlação entre o país das pessoas com sua renda e saldo bancário.


```python
# para verificar confirmar o que verifiquei acima, plotarei um gráfio com todos os clientes agora.
sns.jointplot(x = 'estimated_salary', y = 'balance', data = df, color='b')
```




    <seaborn.axisgrid.JointGrid at 0x7d8fe48cfcb0>




    
![png](output_37_1.png)
    


Se confirmou o que foi constatado acima, o que se mostra intrigante, pois nesse dataset não existe uma relação clara entre salário e saldo bancário dos clientes.

# Análise de Churn


```python
# Aqui iremos ter uma visão geral do churn, e tentar achar uma correlação ou fator que possa levar a uma investigação mais profunda.
print('Aq quantidade de clientes que deram churn = ', df['churn'].sum())
print('Aq quantidade de clientes que não deram churn = ', (df['churn'] == 0).sum())
print('Porcentagem de clientes que deram churn = ', (df['churn'].sum() / len(df)) * 100,"%")
```

    Aq quantidade de clientes que deram churn =  2037
    Aq quantidade de clientes que não deram churn =  7963
    Porcentagem de clientes que deram churn =  20.369999999999997 %



```python
# ANÁLISE DE CHURN VS. FATORES NUMÉRICOS
# Boxplots são ideais para comparar a distribuição de uma variável numérica (como Idade ou Saldo) entre os dois grupos de churn.

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.boxplot(x='churn', y='age', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Age vs. Churn')
axes[0, 0].set_xlabel('Churn')
axes[0, 0].set_ylabel('Age')

sns.boxplot(x='churn', y='balance', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Balance vs. Churn')
axes[0, 1].set_xlabel('Churn')
axes[0, 1].set_ylabel('Balance')

sns.boxplot(x='churn', y='credit_score', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Credit Score vs. Churn')
axes[1, 0].set_xlabel('Churn')
axes[1, 0].set_ylabel('Credit Score')

sns.boxplot(x='churn', y='estimated_salary', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Estimated Salary vs. Churn')
axes[1, 1].set_xlabel('Churn')
axes[1, 1].set_ylabel('Estimated Salary')

plt.tight_layout()
plt.show()
```


    
![png](output_41_0.png)
    


**Age vs. Churn (Idade vs. Churn)**:

A mediana (linha central) da idade é visivelmente mais alta para clientes que cancelaram (Churn=1) do que para os que não cancelaram (Churn=0).

O corpo da caixa para Churn=1 também está deslocado para idades mais altas.

**Conclusão:** Clientes mais velhos tendem a ter uma taxa de churn maior.

**Balance vs. Churn (Saldo vs. Churn)**:

A mediana e a distribuição geral do saldo (Balance) são ligeiramente mais altas para clientes que cancelaram (Churn=1), embora as caixas se sobreponham bastante.

**Conclusão:** Clientes com saldo tendem a ter uma taxa de churn um pouco maior, mas a diferença não é tão drástica quanto a idade.

**Credit Score vs. Churn (Pontuação de Crédito vs. Churn):**

As medianas e a distribuição (caixa) das pontuações de crédito são praticamente idênticas para os dois grupos (Churn=0 e Churn=1).

Conclusão: A pontuação de crédito não demonstra ser um fator discriminante significativo de churn.

**Estimated Salary vs. Churn (Salário Estimado vs. Churn):**

As medianas e a distribuição (caixa) dos salários estimados são muito semelhantes para ambos os grupos (Churn=0 e Churn=1).

**Conclusão:** O salário estimado não tem correlação aparente com a taxa de churn.

**Em Resumo**
A análise dos boxplots sugere que a Idade tem a correlação mais forte com churn, pois a mediana é maior para clientes que cancelaram (1). O Saldo também é ligeiramente maior para o grupo que cancelou. Pontuação de Crédito e Salário Estimado não mostram diferença significativa entre clientes que cancelaram e que não cancelaram.

### Agora vamos continuar a analise por:
Distribution of Country by Churn (Distribuição de País por Churn);
Distribution of Gender by Churn (Distribuição de Gênero por Churn);
Distribution of Active Member by Churn (Distribuição de Membro Ativo por Churn);
Distribution of Products Number by Churn (Distribuição de Número de Produtos por Churn).


```python
# ANÁLISE DE CHURN VS. FATORES CATEGÓRICOS
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.countplot(x='country', hue='churn', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Country by Churn')
axes[0, 0].set_xlabel('Country')
axes[0, 0].set_ylabel('Count')

sns.countplot(x='gender', hue='churn', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Gender by Churn')
axes[0, 1].set_xlabel('Gender')
axes[0, 1].set_ylabel('Count')

sns.countplot(x='active_member', hue='churn', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Active Member by Churn')
axes[1, 0].set_xlabel('Active Member (0: No, 1: Yes)')
axes[1, 0].set_ylabel('Count')

sns.countplot(x='products_number', hue='churn', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Products Number by Churn')
axes[1, 1].set_xlabel('Number of Products')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()
```


    
![png](output_44_0.png)
    


Analisando os quatro gráficos de barras agrupadas (countplots), que mostram a distribuição de categorias em relação ao Churn (0 = Não Cancelou, 1 = Cancelou):

**Distribution of Country by Churn (Distribuição de País por Churn):**

França tem o maior número total de clientes, mas uma taxa de churn (a proporção da barra azul para a vermelha) relativamente baixa.

Alemanha tem o menor número total de clientes, mas a proporção da barra azul (churn=1) em relação à barra vermelha (churn=0) é notavelmente a mais alta, indicando que a Alemanha tem a maior taxa de churn.

Espanha está no meio, com uma taxa de churn intermediária.

**Distribution of Gender by Churn (Distribuição de Gênero por Churn):**

Embora haja mais clientes do sexo Masculino no total, a proporção de clientes que cancelaram (barra azul) é visivelmente maior para o sexo Feminino do que para o Masculino.

**Conclusão:** Clientes do sexo feminino têm maior probabilidade de churn.

**Distribution of Active Member by Churn (Distribuição de Membro Ativo por Churn):**

Clientes que não são membros ativos (0) têm uma proporção de churn (barra azul) visivelmente maior do que aqueles que são membros ativos (1).

**Conclusão:** Clientes inativos são mais propensos a cancelar.

**Distribution of Products Number by Churn (Distribuição de Número de Produtos por Churn):**

Clientes com 1 ou 2 produtos representam a maioria, e a taxa de churn é baixa a moderada.

Clientes com 3 ou 4 produtos têm uma taxa de churn extremamente alta (a barra azul é quase igual ou maior que a barra vermelha).

**Conclusão:** Ter 3 ou mais produtos está fortemente associado ao churn.

## 4. Conclusões e Recomendações Estratégicas

### 4.1. Perfil do Cliente com Risco de Churn

A análise revelou que o churn não é um evento aleatório. O perfil de maior risco é um cliente:
- **Comportamentalmente:** Que possui **3 ou 4 produtos** e é **inativo**.
- **Demograficamente:** **Mais velho**, do sexo **feminino** e residente na **Alemanha**.
- **Financeiramente:** A pontuação de crédito e o salário não são fatores decisivos.

### 4.2. Recomendações Acionáveis

1.  **Ação Imediata (Grupo Crítico):** Investigar a experiência do cliente com **3+ produtos**. A taxa de churn neste segmento é alarmante e exige uma análise de causa-raiz.
2.  **Campanhas de Retenção Segmentadas:** Lançar campanhas proativas focadas nos perfis de alto risco (ex: clientes na Alemanha, mulheres, clientes seniores).
3.  **Programa de Reengajamento:** Criar uma estratégia para reativar membros inativos antes que o cancelamento ocorra.

### 4.3. Próximos Passos
- **Desenvolver um Modelo Preditivo:** Usar estes *insights* para treinar um modelo de Machine Learning que preveja o risco de churn a nível individual.
- **Monitorizar KPIs:** Acompanhar o sucesso das ações de retenção implementadas.


##Próximos Passos e Insights Estratégicos
Com base nestas descobertas, as iniciativas de retenção devem ser altamente segmentadas:

Foco em Segmentos de Risco (Retenção Ativa):

Desenvolver estratégias de retenção específicas e proativas para clientes mais velhos e aqueles com saldo em conta, talvez oferecendo benefícios ou suporte especializado.

Investigação Aprofundada (Intervenções Específicas):

É essencial investigar as razões subjacentes ao alto churn nos segmentos: clientes na Alemanha, mulheres, membros inativos e aqueles com múltiplos produtos (3 ou 4). Por exemplo, podem ser necessárias melhorias na experiência do usuário para multi-produtos ou campanhas de engajamento direcionadas para membros inativos.

A implementação dessas ações focadas permitirá à empresa mitigar o churn de forma mais eficiente, concentrando recursos onde o risco é maior.

