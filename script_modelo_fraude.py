# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:36:02 2026

@author: silva
"""

# In[0.1]: Instalação dos pacotes

# pip install pandas
# pip install numpy
# pip install -U seaborn
# pip install matplotlib
# pip install plotly
# pip install scipy
# pip install statsmodels
# pip install scikit-learn
# pip install --upgrade statstests

# In[0.2]: Importação dos pacotes

import pandas as pd  # Manipulação de dados em formato de dataframe
import numpy as np   # Operações matemáticas
import seaborn as sns  # Visualização gráfica
import matplotlib.pyplot as plt  # Visualização gráfica
import statsmodels.api as sm  # Estimação de modelos
from statstests.process import stepwise
import plotly.graph_objects as go  # Gráficos interativos (Plotly)
import plotly.io as pio
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# Configuração para abrir o gráfico Plotly direto no navegador padrão
pio.renderers.default = 'browser'

# In[0.3]: Baixando a base de dados

df_credito = pd.read_csv('base_credito.csv', delimiter=',', index_col=0)
print("Colunas da base:", df_credito.columns.tolist())

# Características das variáveis do dataset
df_credito.info()

# Estatísticas univariadas
print(df_credito.describe())

# Tabela de frequência absolutas da variável 'is_fraude'
print("\nDistribuição da variável Alvo (is_fraude):")
print(df_credito['is_fraude'].value_counts().sort_index())

# In[0.4]: Estimação do modelo logístico binário (Sem Score Externo)

# Definindo apenas as variáveis de comportamento e cadastro para evitar quase-separação
variaveis_comportamentais = ['idade', 'renda_mensal', 'tempo_emprego_meses', 
                             'valor_emprestimo', 'numero_parcelas', 'utilizacao_limite']

X = df_credito[variaveis_comportamentais]
y = df_credito['is_fraude']

# Adicionar a constante
X = sm.add_constant(X)

modelo_fraude = sm.Logit(y, X).fit()
print(modelo_fraude.summary())

# In[0.5]: Aplicando o procedimento Stepwise

step_modelo_fraude = stepwise(modelo_fraude, pvalue_limit=0.05)
print(step_modelo_fraude.summary())

    
    

# In[0.6]: Criação da matriz de confusão

# 1. Gerar as probabilidades passando o DataFrame original (df_credito)
df_credito['probabilidade'] = step_modelo_fraude.predict(df_credito)

# 2. Definir a nova linha de corte otimizada (Cutoff em 5%)
df_credito['predicao'] = (df_credito['probabilidade'] > 0.05).astype(int)

# 3. Criar a Matriz de Confusão real vs predito
matriz = confusion_matrix(df_credito['is_fraude'], df_credito['predicao'])

# 4. Plotar a matriz de forma visual
plt.figure(figsize=(6, 4))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Previsto Saudável', 'Previsto Fraude'],
            yticklabels=['Real Saudável', 'Real Fraude'])
plt.title('Matriz de Confusão - Modelo de Fraude (Cutoff 5%)')
plt.ylabel('Realidade')
plt.xlabel('Previsão do Modelo')
plt.tight_layout()
plt.show()

# 5. Exibir o relatório com Acurácia, Recall (Sensibilidade) e Precisão
print("\n=== RELATÓRIO DE PERFORMANCE ===")
print(classification_report(df_credito['is_fraude'], df_credito['predicao']))

# In[0.7]: Criação da curva ROC e cálculo do Gini

# 1. Mapeando as variáveis para os cálculos de métricas
y_real = df_credito['is_fraude']
y_probs = df_credito['probabilidade']

# 2. Calcular a taxa de falsos positivos (FPR) e verdadeiros positivos (TPR)
fpr, tpr, thresholds = roc_curve(y_real, y_probs)

# 3. Calcular o AUC (Área sob a Curva) e o coeficiente de Gini baseado na curva estável
auc = roc_auc_score(y_real, y_probs)
gini = (2 * auc) - 1

# 4. Plotar a Curva ROC
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkblue', lw=2, label=f'Modelo de Fraude (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Chute Aleatório (AUC = 0.5000)')

# Customização do gráfico
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade / Recall)')
plt.title('Curva ROC - Modelo de Detecção de Fraude')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)

# Exibir os valores de AUC e Gini no gráfico como texto explicativo
plt.text(0.5, 0.3, f'AUC: {auc:.4f}\nGini: {gini:.4f}', 
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.show()

