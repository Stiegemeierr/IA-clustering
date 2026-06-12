Trabalho Prático: Algoritmos de Clustering 

Cada dupla deverá escolher um dataset real da área da saúde e realizar o pré-processamento necessário antes da aplicação dos algoritmos de clustering, utilizando algum método de análise descritiva. 

> **OBS:** Não usar o rótulo na clusterização (remover a coluna de classe/outcome antes de rodar os algoritmos). Porém, após a clusterização, os rótulos originais podem ser readicionados para avaliação e interpretação dos clusters encontrados. 

---

Requisitos do Dataset 

O dataset deve obedecer aos seguintes critérios:

* Ser real (não sintético). 
* Possuir mais de 1000 instâncias. 
* Ser relacionado a Diabetes, Doença cardiovascular ou Hipertensão. 


*OBS: No máximo duas duplas com o mesmo tipo de dataset.* 


* Ter sido utilizado em artigo científico (journal paper). 
* O artigo utilizado deve ser citado no relatório. 


**Sugestões de sites para encontrar o dataset** **:**

* UCI Machine Learning Repository 
* Kaggle Datasets 
* PhysioNet 
* Bioinformatics Data 

---

Definição do Problema 

Antes de rodar os algoritmos, os alunos devem definir claramente um problema que querem resolver com o clustering. 

* **Exemplos de problemas bem definidos** **:** 
* **Diabetes:** Ver se pacientes jovens formam um grupo e idosos outro. 
* **Hipertensão:** Descobrir se existe grupo de pressão muito alta versus pressão moderada. 
* **Doença Cardiovascular:** Ver se grupos correspondem a risco baixo, médio e alto. 



---

Análise Descritiva 

* **Univariada:** Entender cada variável sozinha e detectar outliers. 
*Métodos:* Média, Mediana, Desvio padrão, Mínimo, Máximo, Quartis, Histograma, Boxplot. 
* **Bivariada:** Medir relação entre pares de variáveis e identificar redundâncias. 
*Métodos:* Correlação de Pearson, Correlação de Spearman, Gráfico de dispersão. 
* **Multivariada:** Visualizar relações gerais e reduzir dimensionalidade. 
*Métodos:* Matriz de correlação (heatmap), PCA (Análise de Componentes Principais). 

---

Pré-processamento 
 
* **Tratar valores faltantes:** Algoritmos não aceitam NaN. 
* **Tratar outliers:** Evitam distorção dos clusters. 
* **Padronizar/normalizar:** Evita que variáveis com valores grandes dominem as pequenas. 
* **Remover colunas desnecessárias:** ID, nome e data só adicionam ruído. 
* **Remover o rótulo:** Clustering é não supervisionado (não pode usar a resposta). 
**Reduzir com PCA (opcional):** Diminui variáveis, custo e facilita visualização. 
* **Verificar NaN e dados numéricos:** Garante que tudo está pronto para os algoritmos. 

---

Algoritmos 

Implementar utilizando *Scikit-learn*, garantindo que cada algoritmo seja testado com diferentes hiperparâmetros. 

* K-Means 
* Agglomerative Clustering (com diferentes métodos de linkage) 
* DBSCAN 



---

Métricas de Avaliação 

Comparar internamente quais hiperparâmetros tiveram melhores resultados e depois comparar os algoritmos entre si, utilizando as seguintes métricas:

* Silhouette Score 
* Davies-Bouldin Index 
* Calinski-Harabasz Score 

---

O Relatório De

* Descrição do dataset (análise descritiva). 
* Pré-processamento realizado. 
* Tabelas e gráficos dos resultados. 
* Comparação e discussão das similaridades e diferenças entre os resultados obtidos. 



---

Entrega 

* **Data Limite e E-mail:** Enviar até **03/06** para: ecradaelli@inf.ufsm.br. 
* **Formato do Envio:** A pasta `.zip` deve conter:
* Código Python (`.py`) 
* Dataset (`.csv`) 
* Relatório (`.pdf`) 
* **Nomeação da pasta:** `Nome(s)_trabalho_clustering.zip` 
**OBS:** Caso seja em dupla, ambos devem enviar a pasta.