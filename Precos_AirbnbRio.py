#!/usr/bin/env python
# coding: utf-8

# # Projeto Airbnb Rio - Ferramenta de Previsão de Preço de Imóvel para pessoas comuns 

# ### Importar Bibliotecas e Bases de Dados

# In[4]:


import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# In[5]:


meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

caminho_bases = pathlib.Path('dataset')

base_airbnb = pd.DataFrame()

for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]
    
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv' , ''))
    
    df = pd.read_csv(r'dataset\{}'.format(arquivo.name))
    
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = base_airbnb.append(df)
#display(base_airbnb)

#abril2018_df = pd.read_csv(r'dataset\abril2018.csv')
#display(abril2018_df)


# - Colunas a serem excluidas:
# - 1. IDs, Links, informações não releveantes
# - 2. Colunas repetidas ou semelhantes
# - 3. Colunas com texto livre
# - 4. Colunas com todos os valores iguais
# 

# In[6]:


print(list(base_airbnb.columns))

base_airbnb.head(1000).to_csv('primeiros_1000.csv', sep = ';')


# ### Se tivermos muitas colunas, já vamos identificar quais colunas podemos excluir
# 
# 
# Com isso trabalheremos com essas 34 colunas

# In[7]:


colunas = ['host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']

base_airbnb = base_airbnb.loc[:, colunas]

#display(base_airbnb)


# ### Tratar Valores Faltando
# 
# 
# Colunas com o somatório de valores NaN supeior à 30000, serão descartadas essas colunas da anaálise

# In[8]:


for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 30000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)
            

print(base_airbnb.isnull().sum())


# In[9]:


base_airbnb = base_airbnb.dropna()

print(base_airbnb.shape)


# ### Verificar Tipos de Dados em cada coluna

# In[10]:


print(base_airbnb.dtypes)
print('-'*60)
print(base_airbnb.iloc[0])


# In[11]:


#price

base_airbnb['minimum_nights'] = base_airbnb['minimum_nights'].astype(np.float32,  copy = False)

base_airbnb['price'] = base_airbnb['price'].str.replace('$','')
base_airbnb['price'] = base_airbnb['price'].str.replace(',','')
base_airbnb['price'] = base_airbnb['price'].str.replace(' ','')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32,  copy = False)
#extrapeople

base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$','')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',','')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(' ','')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32,  copy = False)



print(base_airbnb.dtypes)
print('-'*60)
print(base_airbnb.iloc[0])


# ### Análise Exploratória e Tratar Outliers

# In[12]:


plt.figure(figsize = (15, 10))
sns.heatmap(base_airbnb.corr(), annot=True, cmap='Blues')


# ### Definição de funções para outliers
# 
# Vamos definir fuinções para ajudar na análise de outliers

# In[13]:


def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude

def excluir_outliers(df, nome_coluna):
    linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), :]
    linhas_removidas = linhas - df.shape[0]
    return df, linhas_removidas


# In[14]:


def diagrama_caixa(coluna):
    fig, (ax1, ax2) =  plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    
def histograma(coluna):
    plt.figure(figsize =(15,5))
    sns.distplot(coluna, hist= True)
    
def grafico_barra(coluna):
    plt.figure(figsize =(15,5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))


# ### price

# In[15]:


diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])


# In[16]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print(linhas_removidas)


# In[17]:


histograma(base_airbnb['price'])


# ### extra_people

# In[18]:


diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])


# In[19]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
print(linhas_removidas)


# ### host_listings_count

# In[20]:


diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barra(base_airbnb['host_listings_count'])


# Excluindo outliers, pois para o objetivo do projeto host com mais de 6 imóveis foge do público alvo.

# In[21]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print(linhas_removidas)


# ### accommodates

# In[22]:


diagrama_caixa(base_airbnb['accommodates'])
grafico_barra(base_airbnb['accommodates'])


# In[23]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print(linhas_removidas)


# ### bathrooms

# In[24]:


diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize =(15,5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())


# In[25]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print(linhas_removidas)


# ### bedrooms

# In[26]:


diagrama_caixa(base_airbnb['bedrooms'])
grafico_barra(base_airbnb['bedrooms'])


# In[27]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
print(linhas_removidas)


# ### beds

# In[28]:


diagrama_caixa(base_airbnb['beds'])
grafico_barra(base_airbnb['beds'])


# In[29]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print(linhas_removidas)


# ###  guests_included

# In[30]:


diagrama_caixa(base_airbnb['guests_included'])
plt.figure(figsize =(15,5))
sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())


# Removendo a feature da análise, suopõe-se que os usuários usam comumente o valor 1 como guest_included. Podendo levar nosso modelo a considerar uma feature que não seja essencial para definição do preço. Parecendo melhor a coluna da análise.
# 

# In[31]:


base_airbnb = base_airbnb.drop('guests_included', axis=1)
base_airbnb.shape


# ### minimum_nights

# In[32]:


diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barra(base_airbnb['minimum_nights'])


# In[33]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print(linhas_removidas)


# ### maximum_nights

# In[34]:


diagrama_caixa(base_airbnb['maximum_nights'])
grafico_barra(base_airbnb['maximum_nights'])


# In[35]:


base_airbnb = base_airbnb.drop('maximum_nights', axis=1)
base_airbnb.shape


# #### number_of_reviews             

# In[36]:


diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barra(base_airbnb['number_of_reviews'])


# In[37]:


base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)
base_airbnb.shape


# ### property_type

# In[38]:


print(base_airbnb['property_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# In[39]:


tipos_casa = base_airbnb['property_type'].value_counts()
colunas_agrupar = []


for tipo in tipos_casa.index:
    if tipos_casa[tipo] < 2000:
        colunas_agrupar.append(tipo)
        
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type'] == tipo , 'property_type'] = 'Outros'

print(base_airbnb['property_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# ### room_type

# In[40]:


print(base_airbnb['room_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('room_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# ### bed_type

# In[41]:


print(base_airbnb['bed_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


tabela_bed = base_airbnb['bed_type'].value_counts()
colunas_agrupar = []


for tipo in tabela_bed.index:
    if tabela_bed[tipo] < 10000:
        colunas_agrupar.append(tipo)
        
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['bed_type'] == tipo , 'bed_type'] = 'Outros'

print(base_airbnb['bed_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# ### cancellation_policy

# In[42]:


print(base_airbnb['cancellation_policy'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

#agrupando categorias de cancelation policy

tab_cancelation = base_airbnb['cancellation_policy'].value_counts()
colunas_agrupar = []

for tipo in tab_cancelation.index:
    if tab_cancelation[tipo] < 10000:
        colunas_agrupar.append(tipo)
        
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['cancellation_policy'] == tipo , 'cancellation_policy'] = 'Strict'
    
print(base_airbnb['cancellation_policy'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# ### amenities           
# 
# 
# Por existir uma diversidade grande de amenities, e com a mesma amenitie podendo ser escrita de formas diferentes, vamos avaliar a quantidade de amenities como parametro para o nosso modelo.

# In[43]:


print(base_airbnb['amenities'].iloc[1].split(','))
print(len(base_airbnb['amenities'].iloc[1].split(',')))


base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)



# In[44]:


base_airbnb = base_airbnb.drop('amenities', axis=1)
base_airbnb.shape
#print(len(base_airbnb['n_amenities']))


# In[45]:


diagrama_caixa(base_airbnb['n_amenities'])
grafico_barra(base_airbnb['n_amenities'])


# In[57]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
print(linhas_removidas)

print(base_airbnb.shape)


# ### Visualização de Mapa

# In[47]:


amostra = base_airbnb.sample(n=50000)
centro_mapa= {'lat': amostra.latitude.mean(), 'lon': amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon ='longitude', z='price', radius=2.5, 
                         center=centro_mapa, zoom=10, mapbox_style = 'stamen-terrain')

mapa.show()


# ### Encoding
# 
# Ajustando features para facilitar o trabalho do modelo
# 
# •Features de True ou False, substituiremos True por 1 e False por 0.
# 
# •Features de Categoria(método encoding de variáveis dummies)

# In[48]:


colunas_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 't', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 'f', coluna] = 0


# In[49]:


colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy' ]

base_airbnb_cod = pd.get_dummies(base_airbnb_cod, columns=colunas_categorias)

display(base_airbnb_cod.head())


# ### Modelo de Previsão

# Métricas de Avaliação

# In[50]:


def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\nR²: {r2:.2%}\nRSME: {RSME:.2f}'


# Escolha dos modelos para teste:
# 1. Random Forest
# 2. Linear Regression
# 3. Extra Tree

# In[58]:


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegression': modelo_lr,
          'ExtraTrees': modelo_et
          }

y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis=1)

print(x.shape, y.shape)


# ### Análise do Melhor Modelo

# Separar os dados de treino e teste + Treino do Modelo

# In[63]:


x_train, x_test, y_train, y_test = train_test_split(x , y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(x_train, y_train)
    #testar
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# -Melhor Modelo: ExtraTreesRegressor
# 
#     O modelo obteve o maior R² e também o menor RSME. Como não houve uma grande diferença de velocidade de treino e de previsão desse modelo comparado ao RandomForest, vamos escolher o modelo ExtraTrees.
#     
#     O modelo de regressão linear não obteve resultado satisfatório, com os parâmetros inferiores aos demais modelos

# 

# ### Ajustes e Melhorias no Melhor Modelo

# In[64]:


#print(modelo_et.feature_importances_)

#print(x_train.columns)

importancia_features = pd.DataFrame(modelo_et.feature_importances_, x_train.columns)
importancia_features = importancia_features.sort_values(by= 0, ascending = False)
display(importancia_features)

plt.figure(figsize =(15,5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
ax.tick_params(axis='x', rotation = 90)


#     Após a análise de todas as features que envolvem o caso estudado, percebe-se a importância do número de quartos, superando até a localização para a estimativa de preço, banheiros e número de amenities também mostram bastante relevância no estudo, sendo inferiores à quantidade extra de pessoas. Também incremetam o valor do imóvel o número de camas, mínimo de estadias, política de cancelamento e acomodidades.
#     Por outro lado vemos Imóveis para Viagens à Trabalho sendo ignorado no levantamento do preço assim como outros que possuem pouco impacto como Tipo de Cama, Tipo de Propriedade

# In[65]:


base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis=1)

y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis=1)


x_train, x_test, y_train, y_test = train_test_split(x , y, random_state=10)


 #treinar
modelo_et.fit(x_train, y_train)
    #testar
previsao = modelo_et.predict(x_test)
print(avaliar_modelo('ExtraTreesRegressor', y_test, previsao))


# In[66]:


x['price'] = y
x.to_csv('dados.csv')


# In[67]:


import joblib

joblib.dump(modelo_et, 'modelo.joblib')


# In[2]:


print(base_airbnb_cod)


# In[ ]:




