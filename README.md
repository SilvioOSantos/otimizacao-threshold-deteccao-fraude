# Otimização de Threshold para Detecção de Fraude em Crédito

Este projeto apresenta uma esteira completa de modelagem estatística para detecção de fraude na concessão de crédito utilizando **Regressão Logística Binária** em Python. O objetivo principal é simular o impacto financeiro e operacional com foco na calibração e otimização do ponto de corte (*cutoff/threshold*).

---

## 🎯 Contexto de Negócio

Em modelos de concessão de crédito e prevenção à fraude, o dilema clássico reside no *trade-off* entre:
* **Perdas Financeiras:** Aprovar um cliente fraudulento (Falso Negativo).
* **Atrito Operacional / Perda de Receita:** Barrar ou enviar para análise manual um cliente saudável (Falso Positivo), gerando fricção na experiência do usuário.

Considerando que o custo de uma fraude é severamente superior ao custo do atrito operacional, este modelo foi calibrado com um **Cutoff agressivo de 5%**. Ou seja, se o cliente apresentar uma probabilidade de fraude maior que 5%, a transação é preventivamente classificada como risco.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

* **Pandas & NumPy:** Manipulação de dados e operações matemáticas.
* **Statsmodels & Statstests:** Estimação do modelo logístico e automação do procedimento *Stepwise*.
* **Scikit-Learn:** Cálculo de matriz de confusão, curva ROC, AUC e métricas de classificação (`classification_report`).
* **Seaborn & Matplotlib:** Visualização de dados e plotagem das matrizes de diagnóstico.

---

## 📈 Metodologia e Estrutura do Código

1. **Análise Descritiva:** Diagnóstico inicial do desbalanceamento da variável alvo (`is_fraude`).
2. **Seleção de Variáveis (Evitando Quase-Separação):** Utilização exclusiva de variáveis cadastrais e comportamentais puras (`idade`, `renda_mensal`, `tempo_emprego_meses`, `valor_emprestimo`, `numero_parcelas`, `utilizacao_limite`), mitigando o risco de sobreajuste por *scores* externos.
3. **Seleção Estatística (Stepwise):** Aplicação do algoritmo *Stepwise* com critério de parada baseado em p-valor ($p < 0.05$) para retenção apenas dos seletores estatisticamente significativos.
4. **Calibração do Threshold:** Aplicação do ponto de corte em 5% para priorizar a sensibilidade (*Recall*) do modelo sobre o risco.
5. **Avaliação Estatística:** Mensuração de performance através da **Matriz de Confusão**, **Curva ROC** e cálculo do **Coeficiente de Gini**.

---

## 📊 Principais Indicadores de Performance

O modelo avalia a aderência e a separação de safras por meio de duas das principais métricas utilizadas por bureaus de crédito e instituições financeiras:

* **AUC (Area Under the Curve):** Mede a capacidade global do modelo em distinguir entre clientes saudáveis e fraudadores.
* **Coeficiente de Gini:** Utilizado para validar o poder de discriminação do modelo de risco. É calculado através da fórmula:

$$Gini = 2 \times AUC - 1$$

---

## 🚀 Como Executar o Projeto

1. Certifique-se de ter o arquivo contendo a base histórica (`base_credito.csv`) no mesmo diretório do script.
2. Instale as dependências necessárias:
   pip install pandas numpy seaborn matplotlib statsmodels scikit-learn statstest
