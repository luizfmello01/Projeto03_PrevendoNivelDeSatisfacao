#!/usr/bin/env python
# coding: utf-8

# # Prevendo o Nível de Satisfação dos Clientes do Santander

# ## 1.0 - Problema de negócio

# Descrição do problema:
# A satisfação do cliente é uma medida fundamental de sucesso.
# Clientes insatisfeitos cancelam seus serviços e raramente expressam sua insatisfação antes de sair. Clientes satisfeitos, por outro lado, se tornam defensores da marca!
# O Banco Santander está pedindo para ajudá-los a identificar clientes insatisfeitos no início do relacionamento.
# 
# Objetivo da análise:
# Identificar os clientes que estão satisfeitos ou insatisfeitos com sua experiência bancária.
# Após a entrega do produto o banco podera identificar os clientes insatisfeitos no início do relacionamento, permitindo que o Santander adote uma medida proativa para melhorar a felicidade do cliente e não perde-lo para concorrência.
# 
# Temos dados históricos para realizar a tarefa? Sim
# 
# O que será feito nesse notebook? Análise do conjunto de dados e treinamento do modelo preditivo que deve ter no mínimo 70% de acerto.
# 
# <span style="color:red">OBS: Esse é um projeto da formação Cientista de dados da Data Science Academy</span><br/>
# <span style="color:blue">Fonte de dados: </span> <a style="text-decoration:none" href="https://www.kaggle.com/c/santander-customer-satisfaction/data" target="_blank">Kaggle</a>

# ## 2.0 - Carregar as bibliotecas e dataset

# In[1]:


# Carregando os pacotes que vão ser utilizados nesse projeto...
# !pip install imblearn
# !pip install xgboost
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import math
from scipy.stats import normaltest
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


# In[2]:


# Função para retornar lista de modelos para teste
def obter_modelos():
    modelos = []
    modelos.append(('LR', LogisticRegression()))
    modelos.append(('LDA', LinearDiscriminantAnalysis()))
    modelos.append(('NB', GaussianNB()))
    modelos.append(('KNN', KNeighborsClassifier()))
    modelos.append(('CART', DecisionTreeClassifier()))
    modelos.append(('SVM', SVC()))
    return modelos

# Função para avaliar o modelo
def metricas_modelo(nome, y, previsao):
    print("Modelo:", nome, "\n")
    print("Confusion Matrix:")
    print( pd.DataFrame( confusion_matrix(y, previsao), 
                  columns=["Satisfeito", "Não Satisfeito"], 
                  index=["Satisfeito", "Não Satisfeito"] ) )
    print("\nRelatório de classificação:")
    print( classification_report(y, previsao, target_names=["Satisfeito", "Não Satisfeito"]) )
    
# Função para balancear as classes da variável target
def balancear_classe(x, y, seed = 42):
    smote = SMOTE(random_state = seed)
    return smote.fit_resample(x, y)


# In[3]:


# Carregando o datasets que vai ser utilizado nesse projeto...
dados_original = pd.read_csv("../data/train.csv", sep=",", encoding="UTF-8")
dados_teste = pd.read_csv("../data/test.csv", sep=",", encoding="UTF-8")
resultado_teste = pd.read_csv("../data/sample_submission.csv", sep=",", encoding="UTF-8").TARGET


# ## 3.0 - Análise Exploratória

# In[4]:


# Primeiras linhas do dataset
dados_original.head()


# In[5]:


# Dimensões do dataset
print( "Dimensão do dataset:", dados_original.shape )


# Resultado: Conjunto de dados com muitas variáveis, portanto teremos que aplicar alguma técnica de correlação de variáveis para realizar a análise exploratória de dados.<br/>
# Variável target (Descrição do Kaggle):
# <ul>
#     <li>0 -> Clientes satisfeitos</li>
#     <li>1 -> Clientes insatisfeitos</li>
# </ul>

# In[6]:


# Visualizar nome dos atribuitos e tipo de dados
dados_original.dtypes


# #### Resultado: O nome dos atributos são totalmente anônimos, dificultando a análise.

# In[7]:


# Visualizando valores NA
print( "Contagem de valores NA:", dados_original.isna().sum().sum() )


# #### Resultado: Não existe valores NA no dataset

# In[8]:


# Removendo variável "ID", Motivo: Variável não tem relevância para a análise exploratória
clientes = dados_original.drop("ID", axis=1)


# In[9]:


clientes.head()


# #### Resultado: Variável ID retirado do dataset

# In[10]:


# Função para definir variável numérica
def variavelNumerica(df, variavel):
    if df[variavel].nunique() > 10:
        return True
    else:
        return False


# In[11]:


# Separar variáveis numéricas das variáveis categóricas
variaveis_numericas = np.array([i for i in clientes if variavelNumerica(clientes, i)])
variaveis_categoricas = clientes.columns.drop(variaveis_numericas).values
variaveis_numericas = np.append(variaveis_numericas, ["TARGET"]) # Adicionar variável target para fazer análise de correlação


# In[12]:


variaveis_numericas


# ### 3.1 - Análise das variáveis numéricas

# #### Correlação (Pearson)

# In[13]:


# Matriz de correlação das variáveis numéricas
clientes_cor = clientes[variaveis_numericas].corr()


# In[14]:


# Correlação das variáveis preditoras com a variável target
target_cor = clientes_cor["TARGET"].abs()

# Exibir correlação das variáveis com a variável target
print( target_cor )


# #### Resultado: Variáveis preditoras tem uma correlação muito fraca com a variável "TARGET"

# In[15]:


# Filtrar as variáveis com a correlação absoluta maior ou igual a 0.005 com a variável "TARGET"
variaveis_numericas_cor = target_cor[ target_cor > 0.005].keys().values

# Exibir variáveis com mais correlação com a variável "TARGET"
print( variaveis_numericas_cor )


# #### Resultado: Essas são as variáveis que contem valor da correlação absoluta acima de 0.005 com a variável TARGET

# In[16]:


# Filtrar dataset com as variáveis com maior correlação
clientes_aed_num = clientes[variaveis_numericas_cor]


# In[17]:


# Visualizar dataset com as variáveis com maior correlação
clientes_aed_num.head()


# In[18]:


# Visualizar sumário
clientes_aed_num.describe()


# #### Resultado: Variáveis não estão padronizadas e com possíveis outliers.

# In[19]:


# Moda das variáveis
clientes_aed_num.mode()


# #### Resultado: Muitas variáveis com o valor da moda como 0.0

# #### Simetria (Skewness)

# In[20]:


# Simetria das variáveis
clientes_aed_num.skew()


# #### Resultado: Variáveis assimétricas com valores maior que 0, logo a distribuição tem uma cauda direita (valores acima da média) mais pesada

# #### Histograma

# In[21]:


# Histograma das variáveis numéricas
clientes_aed_num.hist(figsize=(20,70), layout=(25,3))
plt.show()


# #### Resultado: De acordo com o histograma as variáveis não estão em uma distribuição normal e a variável TARGET está totalmente desbalanceada

# #### Teste de hipótese (Distribuição normal)

# In[22]:


# Função para testar distribuição normal das variáveis
def teste_distribuicao_normal(variable):
    alpha = 0.05
    k2, p = normaltest(variable)
    
    print("Variavel:", variable.name)
    if p < alpha:
        print("Rejeitar H0, variável não tem uma distribuição normal. P-Value:", p, "\n")
        return False
    else:
        print("Falha ao rejeitar a H0, variável tem uma distribuição normal. P-Value:", p, "\n")
        return True


# <p>Teste de hipótese:</p>
# <p>H0: Variável tem uma distribuição normal</p>
# <p>Ha: Variável não tem uma distribuição normal</p>

# In[23]:


# Percorrer todas as variáveis numéricas para verificar se estão em uma distribuição normal
var_dnormal = clientes_aed_num.drop("TARGET", axis = 1).apply(teste_distribuicao_normal, axis = 0)


# #### Resultado: As variáveis numéricas não estão com uma distribuição normal

# #### Boxplot

# In[24]:


# Boxplot das variáveis numéricas
clientes_aed_num.plot(kind='box', layout=(25,3), subplots=True, figsize=(20,70))
plt.show()


# #### Resultado: As variáveis estão com muitos outliers (Dados fora da curva)

# #### Valores únicos de cada variável

# In[25]:


# Valores únicos de cada variável
clientes_aed_num.apply(lambda i: len(np.unique(i)), axis = 0)


# #### Resultado: A variável "saldo_var30", "saldo_var42" e "saldo_medio_var5_hace2" tem muitos valores únicos, podemos aplicar técnicas de quantization na variável.

# ### 3.2 - Análise das variáveis categóricas

# #### Correlação (Spearman)

# In[26]:


# Matriz de correlação das variáveis categóricas
clientes_cor = clientes[variaveis_categoricas].corr(method="spearman")


# In[27]:


# Correlação das variáveis preditoras com a variável target
target_cor = clientes_cor["TARGET"].abs()

# Exibir correlação das variáveis com a variável target
print( target_cor )


# #### Resultado: Variáveis preditoras tem uma correlação muito fraca com a variável "TARGET"

# In[28]:


# Filtrar as variáveis com a correlação absoluta maior ou igual a 0.025 com a variável "TARGET"
variaveis_categoricas_cor = target_cor[ target_cor > 0.025].keys().values

# Exibir variáveis com mais correlação com a variável "TARGET"
print( variaveis_categoricas_cor )


# #### Resultado: Essas são as variáveis que contem valor da correlação absoluta acima de 0.025 com a variável TARGET

# In[29]:


# Filtrar dataset com as variáveis com maior correlação
clientes_aed_cat = clientes[variaveis_categoricas_cor]


# In[30]:


# Visualizar dataset com as variáveis com maior correlação
clientes_aed_cat.head()


# In[31]:


# Visualizar sumário do dataset com as variáveis com maior correlação
clientes_aed_cat.describe()


# In[32]:


# Visualizar moda
clientes_aed_cat.mode()


# #### Barplot

# Barplot de cada variável com stack da variável "TARGET"

# In[33]:


for i in clientes_aed_cat.drop("TARGET", axis = 1).columns:
    pd.crosstab(clientes_aed_cat[i], clientes_aed_cat["TARGET"]).plot(kind = "bar",
                                                                     stacked = True,
                                                                     figsize = (10,6),
                                                                     title = i)


# #### Proporção de cada categoria das variáveis preditoras com a variavel "TARGET"

# In[34]:


for i in clientes_aed_cat.drop("TARGET", axis = 1).columns:
    print("Variável:", i)
    print( pd.crosstab(clientes_aed_cat[i], clientes_aed_cat["TARGET"], normalize=True) )
    print("\n")


# #### Resultado: O dataset contem muitos dados com a variável TARGET igual á 0

# #### Countplot da variável TARGET

# In[35]:


plt.figure(figsize=(10,6))
sns.countplot(x = "TARGET", data = clientes_aed_cat, )
plt.title("Countplot da variável TARGET")
plt.show()


# #### Resultado: Confirmando a informação que já recolhemos na análise de variáveis numéricas, a variável "TARGET" está desbalanceada

# ## 4.0 - Principal Component Analysis e Machine Learning

# De acordo com a análise exploratória realizada nos passos anteriores, foi verificado que as variáveis preditoras tem uma fraca correlação com a variável target, vamos utilizar o PCA para agrupar as variáveis em 10 componentes e realizar o treinamento do modelo de machine learning.

# In[36]:


# Separar variáveis preditoras e variável target
x = clientes.drop("TARGET", axis = 1)
y = clientes["TARGET"]


# In[37]:


# Normalizar as variáveis numéricas no dataset de treino
# OBS: Slicing utilizado para remover variável target da normalização.
x[variaveis_numericas[0:-1]] = StandardScaler().fit_transform(x[variaveis_numericas[0:-1]])


# In[38]:


# Separar dados de treino e teste
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.33, random_state=42)


# In[39]:


# Balancear classe da variável target nos dados de treino
treino_x_balanceado, treino_y_balanceado = balancear_classe(treino_x, treino_y)


# In[40]:


# Countplot da variável TARGET
plt.figure(figsize=(10,6))
sns.countplot(x = treino_y_balanceado)
plt.title("Countplot da variável TARGET")
plt.show()


# #### Resultado: Variável TARGET foi balanceada com o método SMOTE

# In[41]:


# Realizando PCA nos dados X
pca = PCA(n_components=10)
treino_x_balanceado = pca.fit_transform(treino_x_balanceado)
teste_x = pca.fit_transform(teste_x)


# In[42]:


# Utilizar a função para obter a lista com modelos de machine learning que vão ser treinados
modelos = obter_modelos()


# In[43]:


# Lista com as métricas de cada modelo
metricas = []

# Treinar o modelo de machine learning
for nome, modelo in modelos:
    kfold = KFold(n_splits=5)
    modelo_resultado = cross_val_score( modelo, treino_x_balanceado, treino_y_balanceado, cv = kfold, scoring = "accuracy" )
    metricas.append( (nome, modelo_resultado) )
    msg = "%s: %f (%f)" % (nome, modelo_resultado.mean(), modelo_resultado.std())
    print( msg )


# In[44]:


# Treinar o modelo KNN
modelo_knn_pca = KNeighborsClassifier()
modelo_knn_pca.fit(treino_x_balanceado, treino_y_balanceado)

# Avaliar o modelo KNN com dados de teste
previsao_knn_pca = modelo_knn_pca.predict(teste_x)
metricas_modelo("KNN", teste_y, previsao_knn_pca)


# In[45]:


# Treinar o modelo SVM
modelo_svm_pca = SVC()
modelo_svm_pca.fit(treino_x_balanceado, treino_y_balanceado)

# Avaliar o modelo SVM com dados de teste
previsao_svm_pca = modelo_svm_pca.predict(teste_x)
metricas_modelo("SVM", teste_y, previsao_svm_pca)


# #### Resultado: O resultado dos modelos não foi satisfatório, no modelo KNN, temos uma acurácia de 96%, porém o modelo não aprendeu corretamente a relação da classe 1 (Não satisfeito) e no modelo SVM foi o inverso do KNN, o modelo não aprendeu corretamente a relação da classe 0 (Satisfeito).
# #### Vamos manipular os dados com objetivo de melhorar a performance dos modelos

# ## 5.0 - Manipulação dos dados

# #### Filtrar dataset principal com as variáveis com mais correlação com a variável "TARGET", avaliadas na análise exploratória

# In[46]:


variaveis = np.unique( np.concatenate((variaveis_categoricas_cor, variaveis_numericas_cor)) )
clientes_mesclado = clientes[variaveis]


# In[47]:


clientes_mesclado.describe()


# #### Sumário do dataset com as variáveis com maior correlação com a variável TARGET

# #### Tratamento de outliers

# In[48]:


# Função para remover outlier
def remover_outlier(valor):
    Q1 = np.percentile(valor, 25)
    Q3 = np.percentile(valor, 75)
    IQR = 1.5 * (Q3 - Q1)
    upper_tail = IQR + Q3
    lower_tail = Q1 - IQR
    valor = np.where(valor > upper_tail, np.nan, valor)
    valor = np.where(valor < lower_tail, np.nan, valor)
    return valor


# In[49]:


# Copia do dataset mesclado com as melhores variáveis
clientes_mung = clientes_mesclado.copy()


# In[50]:


# Retirando a variável target do array
variaveis_numericas_sem_target = np.delete(variaveis_numericas_cor, len(variaveis_numericas_cor)-1)


# In[51]:


# Removendo os valores outliers das variáveis
clientes_mung[variaveis_numericas_sem_target] = clientes_mesclado[variaveis_numericas_sem_target].copy().apply(remover_outlier)
clientes_mung.dropna(inplace = True)


# In[52]:


clientes_mung[variaveis_numericas_sem_target].plot(kind = "box", subplots = True, layout = (25,3), figsize = (20,70))
plt.show()


# #### Resultado: Excluído outliers das variáveis numéricas, porém algumas variáveis ainda estão com outliers

# In[53]:


clientes_mung.describe()


# #### Resultado: Podemos verificar no sumário do dataset que muitas variáveis ficaram somente com valores zero, portanto vamos remover as variáveis que contem valores unicos.

# In[54]:


# Função para retornar variaveis que tenham apenas um valor
def var_um_valor(df):
    lista_de_variaveis_com_valores_unicos = []
    for i in df.columns:
        if len( df[i].value_counts().index.values ) < 2:
            lista_de_variaveis_com_valores_unicos = np.append(lista_de_variaveis_com_valores_unicos, [i])
    return lista_de_variaveis_com_valores_unicos


# In[55]:


variaveis_com_valores_unicos = var_um_valor(clientes_mung)


# In[56]:


# Remover variáveis com apenas um valor do dataset
clientes_mung = clientes_mung.drop(variaveis_com_valores_unicos, axis = 1).reset_index().drop("index", axis = 1)


# In[57]:


clientes_mung.head()


# #### Resultado: Primeiras linhas do dataset com as variáveis com maior correlação e com mais de 1 valor nas variáveis.

# In[58]:


# Atualizar valor das variaveis numericas, foram retiradas algumas variáveis durante a manipulação de dados.
variaveis_numericas_mung = [x for x in variaveis_numericas_sem_target if x not in variaveis_com_valores_unicos]


# In[59]:


# Boxplot das variáveis numéricas
clientes_mung[variaveis_numericas_mung].plot(kind='box', layout=(9,3), subplots=True, figsize=(16,28))
plt.show()


# #### Resultado: Ainda temos outliers em nossas variáveis numéricas, porém não temos como saber se são outliers naturais porque são dados anonimos do banco Santander

# In[60]:


# Countplot da variável TARGET
sns.countplot(x = "TARGET", data = clientes_mung)
plt.show()


# #### Resultado: Variável TARGET está desbalanceada

# #### Separar variáveis preditoras da variável target

# In[61]:


# Separar variáveis preditoras da variável target
x = clientes_mung.drop("TARGET", axis = 1)
y = clientes_mung["TARGET"]


# #### Normalizar as variáveis numéricas

# In[62]:


x[variaveis_numericas_mung] = StandardScaler().fit_transform(x[variaveis_numericas_mung])


# #### Separar dados de treino e de teste

# In[63]:


treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.33, random_state=42)


# In[64]:


treino_x.head()


# #### Dataset de treino com as variáveis numéricas normalizadas.

# #### Balancear classe TARGET dos dados de treino

# In[65]:


treino_x_balanceado, treino_y_balanceado = balancear_classe(treino_x, treino_y)


# In[66]:


# Countplot da variável TARGET
sns.countplot(x = treino_y_balanceado)
plt.show()


# Resultado: Variável TARGET balanceada com o método SMOTE

# ## 6.0 - Feature Selection

# #### Seleção de recursos com SelectKBest

# In[67]:


kbest_f = SelectKBest(score_func=f_classif, k=6).fit(treino_x_balanceado, treino_y_balanceado)


# In[68]:


x_pontos_variaveis = []
y_pontos_variaveis = []

for i in np.arange(0, len(kbest_f.scores_)):
    x_pontos_variaveis.append(treino_x.columns.values[i])
    y_pontos_variaveis.append(np.round(kbest_f.scores_[i], 2))
    
kbest_f_df = pd.DataFrame({"Variavel": x_pontos_variaveis, "Score": y_pontos_variaveis}).sort_values("Score", ascending=False).reset_index().drop("index", axis = 1)


# In[69]:


# Plot do score das variaveis com f_classif
figure = plt.figure(figsize=(23,10))
sns.barplot(x = "Variavel", y = "Score", data = kbest_f_df[0:10])
plt.title("Score das variáveis com f_classif", fontdict = {'fontsize': 16})
plt.show()


# #### Seleção de recursos com RandomForest

# In[70]:


kbest_cat = ExtraTreesClassifier(n_estimators=100).fit(treino_x_balanceado, treino_y_balanceado)


# In[71]:


x_pontos_variaveis = []
y_pontos_variaveis = []

for i in np.arange(0, len(kbest_cat.feature_importances_)):
    x_pontos_variaveis.append(treino_x.columns.values[i])
    y_pontos_variaveis.append(np.round(kbest_cat.feature_importances_[i], 2))
    
kbest_cat_df = pd.DataFrame({"Variavel": x_pontos_variaveis, "Score": y_pontos_variaveis}).sort_values("Score", ascending=False).reset_index().drop("index", axis = 1)


# In[72]:


# Plot do score das variaveis com ExtraTreesClassifier
figure = plt.figure(figsize=(23,10))
sns.barplot(x = "Variavel", y = "Score", data = kbest_cat_df[0:10])
plt.title("Score das variáveis com ExtraTreesClassifier", fontdict = {'fontsize': 16})
plt.show()


# #### Selecionando as variáveis com score maior que 1000 do SelectKBest com f_classif

# In[73]:


selecao_variaveis = kbest_f_df[ kbest_f_df["Score"] > 1000.00 ].Variavel


# In[74]:


# Filtrar dados de treino e teste X com as melhores variáveis (Utilizado ExtraTreesClassifier)
treino_x_balanceado = treino_x_balanceado[selecao_variaveis]
teste_x = teste_x[selecao_variaveis]


# In[75]:


treino_x_balanceado.head()


# #### Dataset de treino com 17 variáveis preditoras

# ## 7.0 - Treinamento e Avaliação do modelo de Machine Learning

# #### Função para estruturar dados de teste para entrada do modelo de machine learning

# In[76]:


def estruturar_dados(df):
    # Verificar se contem todas as colunas no dataset
    if ( False in np.isin(selecao_variaveis, df.columns) ):
        print("Não foi encontrada alguma das colunas necessárias no dataset, lista das colunas abaixo")
        print("Colunas:", selecao_variaveis)
        return None
    
    # Pegar somente as variáveis que contem no modelo
    df = df[selecao_variaveis]
    # Normalizar variáveis numéricas
    variaveis_para_normalizar = [i for i in df.columns if np.isin(i, variaveis_numericas_mung)]
    df[variaveis_para_normalizar] = StandardScaler().fit_transform(df[variaveis_para_normalizar])
    return df


# #### Cross-Validation com modelos da lista + XGBoost

# In[77]:


# Obtendo a lista de modelos
modelos_ml = obter_modelos()


# In[78]:


# Treinando todos os modelos da lista (Cross-Validation)
for nome, modelo in modelos_ml:
    kfold = KFold(n_splits=5)
    modelo_crossval = cross_val_score(modelo, treino_x_balanceado, treino_y_balanceado, cv = kfold, scoring = "accuracy")
    msg = "%s: %f (%f)" % (nome, modelo_crossval.mean(), modelo_crossval.std())
    print( msg )


# In[79]:


# Treinamento com XGBoost (Cross-Validation)
modelo_xgb_cv = xgb.XGBClassifier()
treino_xgb = xgb.DMatrix(treino_x_balanceado, label = treino_y_balanceado)
param_xgb = {'objective':'reg:logistic'}

modelo_crossval = xgb.cv(param_xgb, treino_xgb, nfold=10,
       metrics={'error'}, seed=42)

print( "XGB:", np.round(1 - np.mean( modelo_crossval.to_dict()['test-error-mean'][1] ), 2) )


# #### Resultado: Os 3 melhores modelos foram KNN, CART e XGBoost

# #### Treinamento e Avaliação do modelo Regressão Logística

# In[80]:


# Treinar o modelo de Regressão Logística
modelo_lr = LogisticRegression()
modelo_lr.fit(treino_x_balanceado, treino_y_balanceado)

# Avaliar o modelo Regressão Logística com dados de teste
previsao_lr = modelo_lr.predict(teste_x)
metricas_modelo("Logistic Regression", teste_y, previsao_lr)


# #### Treinamento e Avaliação do modelo KNN

# In[81]:


# Treinar o modelo KNN
modelo_knn = KNeighborsClassifier()
modelo_knn.fit(treino_x_balanceado, treino_y_balanceado)

# Avaliar o modelo KNN com dados de teste
previsao_knn = modelo_knn.predict(teste_x)
metricas_modelo("KNN", teste_y, previsao_knn)


# #### Treinamento e Avaliação do modelo CART

# In[82]:


# Treinar o modelo CART
modelo_cart = DecisionTreeClassifier()
modelo_cart.fit(treino_x_balanceado, treino_y_balanceado)

# Avaliar o modelo CART com dados de teste
previsao_cart = modelo_cart.predict(teste_x)
metricas_modelo("CART", teste_y, previsao_cart)


# #### Treinamento e Avaliação do modelo SVM

# In[83]:


# Treinar o modelo SVM
modelo_svm = SVC()
modelo_svm.fit(treino_x_balanceado, treino_y_balanceado)

# Avaliar o modelo SVM com dados de teste
previsao_svm = modelo_svm.predict(teste_x)
metricas_modelo("SVM", teste_y, previsao_svm)


# #### Treinamento e Avaliação do modelo XGBoost

# In[84]:


# Treinar o modelo XGBoost
teste_xgb = xgb.DMatrix(teste_x, teste_y)
modelo_xgb = xgb.train(param_xgb, treino_xgb)

# Avaliar o modelo XGBoost com dados de teste
previsao_xgb = modelo_xgb.predict(teste_xgb)
metricas_modelo("XGB", teste_y, np.around(previsao_xgb))


# #### Resultado: Entre os modelos treinados, os modelos escolhidos para otimização são Regressão Linear e XGBoost porque apresentaram uma boa acurácia e recall

# ## 8.0 - Otimização do modelo de Machine Learning

# #### Otimizar modelo de Regressão Logística

# In[85]:


grid = {
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "penalty": ["none", "l1", "l2", "elasticnet"],
    "C": [100, 10, 1.0, 0.1, 0.01]
}

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(LogisticRegression(), param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(treino_x_balanceado, treino_y_balanceado)

print("Melhor: %f use %s" % (grid_result.best_score_, grid_result.best_params_))


# In[86]:


# Treinar o modelo de Regressão Logística Otimizado
modelo_lr_otimizado = LogisticRegression(C = 0.1, penalty = 'l2', solver = 'lbfgs')
modelo_lr_otimizado.fit(treino_x_balanceado, treino_y_balanceado)


# Avaliar o modelo Regressão Logística Otimizado com dados de teste
previsao_lr_otimizado = grid_result.predict(teste_x)
metricas_modelo("Logistic Regression Otimizado", teste_y, previsao_lr_otimizado)


# #### Resultado: Tivemos uma melhora de 0.01 em recall otimizando o modelo de regressão linear

# #### Otimizar modelo XGBoost

# In[87]:


params = {'max_depth': [3,6,10], 
           'learning_rate': [0.01, 0.05, 0.1], 
           'n_estimators': [100, 500, 1000], 
           'colsample_bytree': [0,3, 0,7]}

xgbc = xgb.XGBClassifier(seed = 42, eval_metric = 'logloss')

grid_search = GridSearchCV(estimator = xgbc,
                          param_grid = params,
                          scoring = 'accuracy',
                          verbose = 1,
                          n_jobs = -1)

grid_result = grid_search.fit(treino_x_balanceado, treino_y_balanceado)
print("Melhor: %f use %s" % (grid_result.best_score_, grid_result.best_params_))


# In[88]:


# Treinar o modelo XGBoost otimizado
modelo_xgb_otimizado = xgb.XGBClassifier(objective = 'reg:logistic',
                 eval_metric = 'logloss',
                 seed = 42,
                 colsample_bytree = 0,
                 learning_rate = 0.1,
                 max_depth = 10,
                 n_estimators = 1000)
                 
modelo_xgb_otimizado.fit(treino_x_balanceado, treino_y_balanceado)


# Avaliar o modelo XGBoost Otimizado com dados de teste
previsao_xgb_otimizado = grid_result.predict(teste_x)
metricas_modelo("XGBoost Otimizado", teste_y, np.around(previsao_xgb_otimizado))


# #### Resultado: Tivemos uma melhora de 0.03 de acurácia, porém tivemos uma perda de 0.01 em recall otimizando o XGBoost

# ### Testando perfomance dos modelos otimizados com dataset de teste disponibilizado no Kaggle (test.csv)

# In[89]:


# Visualizar dataset de teste
dados_teste.head()


# #### Resultado: Dataset de teste disponibilizado pelo Kaggle, dados não estão estruturado para a entrada do modelo, vamos estruturar...

# In[90]:


# Transformar dataset de teste para entrada dos modelos
dados_teste_mod = estruturar_dados(dados_teste)


# In[91]:


# Visualizar dataset de teste tratado para entrada de dados dos modelos
dados_teste_mod.head()


# #### Resultado: Primeiras linha do dataset de teste com as variáveis preditores e variáveis numéricas normalizadas

# #### Dataset de teste com Regressão Logística Otimizado

# In[92]:


prever_dataset_teste_rl = modelo_lr_otimizado.predict(dados_teste_mod)
metricas_modelo("Dataset de teste: Regressão Logística", resultado_teste, prever_dataset_teste_rl)


# #### Dataset de teste com XGBoost Otimizado

# In[93]:


prever_dataset_teste_xgb = modelo_xgb_otimizado.predict(dados_teste_mod)
metricas_modelo("Dataset de teste: XGBoost", resultado_teste, prever_dataset_teste_xgb)


# #### Resultado: Podemos observar os dois modelos otimizados com GridSearchCV, o modelo de Regressão Logística apresentou melhor performance comparado ao XGBoost, portanto o modelo de Regressão Logística será o modelo final para entrega do projeto.

# ## 9.0 - Salvar modelo de Machine Learning para entrega final do projeto

# In[94]:


# Salvando modelo de regressão logística otimizado
caminho_modelo = "../modelo/ml_regressaoLogistica.sav"
pickle.dump(modelo_lr_otimizado, open(caminho_modelo, 'wb'))


# ### Resultado: Fim do projeto, foi proposto uma métrica de 70% de acurácia para o modelo de machine learning, o modelo entregue tem 85% de acurácia e uma boa taxa de recall, modelo foi testado com os dados de teste disponibilizado pelo Kaggle.
