# =============================================================================
# Trabalho Prático: Algoritmos de Clustering
# Dataset: Maternal Health Risk Data Set
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

# ========================= CONFIGURAÇÕES GERAIS ==============================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graficos')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilo visual dos gráficos
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.facecolor': 'white',
})
sns.set_style('whitegrid')

RANDOM_STATE = 42

# ========================= CARREGAMENTO DOS DADOS ============================
print("=" * 70)
print("1. CARREGAMENTO DOS DADOS")
print("=" * 70)

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'Maternal Health Risk Data Set.csv')
df = pd.read_csv(csv_path)

print(f"Dimensões do dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")
print(f"\nColunas: {list(df.columns)}")
print(f"\nTipos de dados:\n{df.dtypes}")
print(f"\nPrimeiras 5 linhas:")
print(df.head())
print(f"\nDistribuição de RiskLevel:")
print(df['RiskLevel'].value_counts())

# ========================= SEÇÃO 1: ANÁLISE DESCRITIVA (EDA) =================
print("\n" + "=" * 70)
print("2. ANÁLISE DESCRITIVA (EDA)")
print("=" * 70)

# --- 2.1 Análise Univariada ---
print("\n--- 2.1 Análise Univariada ---")
features_numericas = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

# Estatísticas descritivas completas
stats = df[features_numericas].describe().T
stats['mediana'] = df[features_numericas].median()
stats['variancia'] = df[features_numericas].var()
stats['IQR'] = stats['75%'] - stats['25%']
print("\nEstatísticas descritivas:")
print(stats.to_string())

# Verificação de valores faltantes
print(f"\nValores faltantes por coluna:\n{df.isnull().sum()}")
print(f"Total de NaN: {df.isnull().sum().sum()}")

# Detecção de outliers suspeitos
print(f"\nValores mínimos de HeartRate: {sorted(df['HeartRate'].unique())[:5]}")
hr_outliers = df[df['HeartRate'] < 30]
print(f"Registros com HeartRate < 30 (prováveis erros de digitação):")
print(hr_outliers.to_string())

# Gráfico 01: Histogramas univariados
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle('Distribuição das Variáveis — Histogramas', fontsize=14, fontweight='bold')
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
for i, (col, color) in enumerate(zip(features_numericas, colors)):
    ax = axes[i // 3, i % 3]
    ax.hist(df[col], bins=30, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequência')
    # Linha vertical na média
    ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.2, label=f'Média: {df[col].mean():.1f}')
    ax.axvline(df[col].median(), color='navy', linestyle=':', linewidth=1.2, label=f'Mediana: {df[col].median():.1f}')
    ax.legend(fontsize=7)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '01_histogramas.png'), bbox_inches='tight')
plt.close()
print("\n[OK] Gráfico salvo: 01_histogramas.png")

# Gráfico 02: Boxplots univariados
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle('Detecção de Outliers — Boxplots', fontsize=14, fontweight='bold')
for i, (col, color) in enumerate(zip(features_numericas, colors)):
    ax = axes[i // 3, i % 3]
    bp = ax.boxplot(df[col], patch_artist=True, vert=True,
                    boxprops=dict(facecolor=color, alpha=0.6),
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    ax.set_title(col, fontweight='bold')
    ax.set_ylabel(col)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '02_boxplots.png'), bbox_inches='tight')
plt.close()
print("[OK] Gráfico salvo: 02_boxplots.png")

# --- 2.2 Análise Bivariada ---
print("\n--- 2.2 Análise Bivariada ---")

# Correlação de Pearson e Spearman
corr_pearson = df[features_numericas].corr(method='pearson')
corr_spearman = df[features_numericas].corr(method='spearman')

print("\nCorrelação de Pearson:")
print(corr_pearson.round(3).to_string())
print("\nCorrelação de Spearman:")
print(corr_spearman.round(3).to_string())

# Gráfico 03: Scatter plots bivariados (pares relevantes)
pares = [('SystolicBP', 'DiastolicBP'), ('Age', 'BS'), ('BS', 'SystolicBP'),
         ('Age', 'HeartRate'), ('SystolicBP', 'HeartRate'), ('BS', 'BodyTemp')]
risk_colors = {'low risk': '#4CAF50', 'mid risk': '#FF9800', 'high risk': '#F44336'}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Análise Bivariada — Scatter Plots por Nível de Risco', fontsize=14, fontweight='bold')
for idx, (x_col, y_col) in enumerate(pares):
    ax = axes[idx // 3, idx % 3]
    for risk, color in risk_colors.items():
        mask = df['RiskLevel'] == risk
        ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                   c=color, label=risk, alpha=0.5, s=20, edgecolor='none')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{x_col} × {y_col}', fontweight='bold')
    if idx == 0:
        ax.legend(fontsize=8, loc='upper left')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '03_scatter_bivariada.png'), bbox_inches='tight')
plt.close()
print("[OK] Gráfico salvo: 03_scatter_bivariada.png")

# --- 2.3 Análise Multivariada ---
print("\n--- 2.3 Análise Multivariada ---")

# Gráfico 04: Heatmap de correlação
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Matrizes de Correlação', fontsize=14, fontweight='bold')

sns.heatmap(corr_pearson, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
            square=True, linewidths=0.5, ax=axes[0],
            vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
axes[0].set_title('Correlação de Pearson', fontweight='bold')

sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
            square=True, linewidths=0.5, ax=axes[1],
            vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
axes[1].set_title('Correlação de Spearman', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUTPUT_DIR, '04_heatmap_correlacao.png'), bbox_inches='tight')
plt.close()
print("[OK] Gráfico salvo: 04_heatmap_correlacao.png")

# Gráfico 05: PCA colorido por RiskLevel (antes do pré-processamento)
scaler_eda = StandardScaler()
X_eda_scaled = scaler_eda.fit_transform(df[features_numericas])
pca_eda = PCA(n_components=2)
X_pca_eda = pca_eda.fit_transform(X_eda_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
for risk, color in risk_colors.items():
    mask = df['RiskLevel'] == risk
    ax.scatter(X_pca_eda[mask, 0], X_pca_eda[mask, 1],
               c=color, label=risk, alpha=0.5, s=25, edgecolor='none')
ax.set_xlabel(f'PC1 ({pca_eda.explained_variance_ratio_[0]:.1%} variância explicada)')
ax.set_ylabel(f'PC2 ({pca_eda.explained_variance_ratio_[1]:.1%} variância explicada)')
ax.set_title('PCA — Visualização 2D dos Dados (colorido por RiskLevel original)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, markerscale=2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_pca_risklevel.png'), bbox_inches='tight')
plt.close()

print(f"\nVariância explicada pelo PCA:")
print(f"  PC1: {pca_eda.explained_variance_ratio_[0]:.4f} ({pca_eda.explained_variance_ratio_[0]:.1%})")
print(f"  PC2: {pca_eda.explained_variance_ratio_[1]:.4f} ({pca_eda.explained_variance_ratio_[1]:.1%})")
print(f"  Total (2 componentes): {sum(pca_eda.explained_variance_ratio_[:2]):.1%}")
print("[OK] Gráfico salvo: 05_pca_risklevel.png")

# Gráfico 05b: Scree Plot — Variância Explicada Cumulativa
pca_full = PCA(n_components=len(features_numericas))
pca_full.fit(X_eda_scaled)

fig, ax = plt.subplots(figsize=(8, 5))
n_comp = range(1, len(features_numericas) + 1)
var_individual = pca_full.explained_variance_ratio_
var_cumulativa = np.cumsum(var_individual)

ax.bar(n_comp, var_individual, alpha=0.7, color='#2196F3', edgecolor='white',
       linewidth=1.5, label='Variância individual')
ax.plot(n_comp, var_cumulativa, 'ro-', markersize=8, linewidth=2, label='Variância cumulativa')
ax.set_xlabel('Componente Principal')
ax.set_ylabel('Proporção da Variância Explicada')
ax.set_title('Scree Plot — Variância Explicada por Componente (PCA)',
             fontsize=13, fontweight='bold')
ax.set_xticks(list(n_comp))
ax.set_xticklabels([f'PC{i}' for i in n_comp])
ax.legend(fontsize=10)
for i, (ind, cum) in enumerate(zip(var_individual, var_cumulativa)):
    ax.annotate(f'{cum:.1%}', (i + 1, cum), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05b_scree_plot.png'), bbox_inches='tight')
plt.close()
print("[OK] Gráfico salvo: 05b_scree_plot.png")
print(f"\nVariância explicada por componente: {[f'{v:.4f}' for v in var_individual]}")
print(f"Variância cumulativa: {[f'{v:.4f}' for v in var_cumulativa]}")

# ========================= SEÇÃO 2: PRÉ-PROCESSAMENTO =======================
print("\n" + "=" * 70)
print("3. PRÉ-PROCESSAMENTO")
print("=" * 70)

# --- 3.1 Salvar rótulos e remover coluna RiskLevel ---
risk_labels = df['RiskLevel'].copy()
df_clean = df.drop(columns=['RiskLevel']).copy()
print(f"\n[OK] Coluna 'RiskLevel' removida. Salva em variável separada para interpretação posterior.")
print(f"  Colunas restantes: {list(df_clean.columns)}")

# --- 3.2 Verificar e tratar NaN ---
nan_count = df_clean.isnull().sum().sum()
print(f"\nValores NaN encontrados: {nan_count}")
if nan_count > 0:
    # Imputar com mediana
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            mediana = df_clean[col].median()
            n_nan = df_clean[col].isnull().sum()
            df_clean[col].fillna(mediana, inplace=True)
            print(f"  → {col}: {n_nan} NaN imputados com mediana ({mediana})")
else:
    print("  → Nenhum NaN encontrado. Nenhum tratamento necessário.")

# --- 3.3 Remover outliers extremos (HeartRate = 7) ---
n_before = len(df_clean)
mask_hr_outlier = df_clean['HeartRate'] < 30
n_hr_outliers = mask_hr_outlier.sum()
print(f"\nOutliers extremos (HeartRate < 30): {n_hr_outliers} registros")
if n_hr_outliers > 0:
    print(f"  Registros removidos:")
    print(df_clean[mask_hr_outlier].to_string())
    df_clean = df_clean[~mask_hr_outlier].reset_index(drop=True)
    risk_labels = risk_labels[~mask_hr_outlier].reset_index(drop=True)
    print(f"  → {n_hr_outliers} registros removidos. Restam {len(df_clean)} registros.")

# --- 3.4 Tratamento de outliers via IQR ---
print(f"\nTratamento de outliers via IQR (clipping):")
outlier_report = {}
for col in df_clean.columns:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_low = (df_clean[col] < lower).sum()
    n_high = (df_clean[col] > upper).sum()
    n_total = n_low + n_high
    outlier_report[col] = {'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
                           'lower': lower, 'upper': upper,
                           'n_outliers': n_total}
    if n_total > 0:
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        print(f"  → {col}: {n_total} outliers clipados (limites: [{lower:.2f}, {upper:.2f}])")
    else:
        print(f"  → {col}: 0 outliers (limites: [{lower:.2f}, {upper:.2f}])")

print(f"\nDataset após pré-processamento: {df_clean.shape[0]} linhas × {df_clean.shape[1]} colunas")

# --- 3.5 Padronização com StandardScaler ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)
X_scaled_df = pd.DataFrame(X_scaled, columns=df_clean.columns)

print(f"\n[OK] Padronização com StandardScaler aplicada.")
print(f"  Médias (devem ser ~0): {X_scaled_df.mean().round(6).to_dict()}")
print(f"  Desvios (devem ser ~1): {X_scaled_df.std().round(4).to_dict()}")

# --- 3.6 PCA para visualização de clusters (2D) ---
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA para visualização 2D:")
print(f"  Variância explicada: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")
print(f"  Total: {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_):.1%})")

# ========================= SEÇÃO 3: MODELAGEM ================================
print("\n" + "=" * 70)
print("4. MODELAGEM — CLUSTERING")
print("=" * 70)

# Função auxiliar para calcular métricas
def calcular_metricas(X, labels, nome_config):
    """Calcula Silhouette, Davies-Bouldin e Calinski-Harabasz."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1) if -1 in labels else 0

    if n_clusters < 2:
        return {
            'config': nome_config,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan
        }

    # Para métricas, filtrar pontos de ruído do DBSCAN
    if n_noise > 0:
        mask_valid = np.array(labels) != -1
        X_valid = X[mask_valid]
        labels_valid = np.array(labels)[mask_valid]
    else:
        X_valid = X
        labels_valid = labels

    if len(set(labels_valid)) < 2:
        return {
            'config': nome_config,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan
        }

    sil = silhouette_score(X_valid, labels_valid)
    dbi = davies_bouldin_score(X_valid, labels_valid)
    chi = calinski_harabasz_score(X_valid, labels_valid)

    return {
        'config': nome_config,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': sil,
        'davies_bouldin': dbi,
        'calinski_harabasz': chi
    }

# ============================================================================
# 4.1 K-MEANS
# ============================================================================
print("\n--- 4.1 K-Means ---")
kmeans_results = []
kmeans_models = {}
inertias = []
k_range = [2, 3, 4, 5]
init_methods = ['k-means++', 'random']

for k in k_range:
    for init_method in init_methods:
        km = KMeans(n_clusters=k, init=init_method, n_init=10,
                    random_state=RANDOM_STATE, max_iter=300)
        labels_km = km.fit_predict(X_scaled)
        config_name = f"K={k}, init={init_method}"
        metricas = calcular_metricas(X_scaled, labels_km, config_name)
        kmeans_results.append(metricas)
        kmeans_models[config_name] = {'model': km, 'labels': labels_km}

        if init_method == 'k-means++':
            inertias.append((k, km.inertia_))

        print(f"  {config_name}: Sil={metricas['silhouette']:.4f}, "
              f"DB={metricas['davies_bouldin']:.4f}, CH={metricas['calinski_harabasz']:.1f}")

kmeans_df = pd.DataFrame(kmeans_results)
print("\nResultados K-Means:")
print(kmeans_df.to_string(index=False))

# Gráfico 06: Método do Cotovelo
fig, ax = plt.subplots(figsize=(8, 5))
ks = [x[0] for x in inertias]
inerts = [x[1] for x in inertias]
ax.plot(ks, inerts, 'bo-', markersize=8, linewidth=2)
ax.set_xlabel('Número de Clusters (K)')
ax.set_ylabel('Inércia (WCSS)')
ax.set_title('Método do Cotovelo — K-Means', fontsize=13, fontweight='bold')
ax.set_xticks(ks)
for k_val, inert in zip(ks, inerts):
    ax.annotate(f'{inert:.0f}', (k_val, inert), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_elbow_method.png'), bbox_inches='tight')
plt.close()
print("[OK] Gráfico salvo: 06_elbow_method.png")

# Gráfico 07: Silhouette Score por K (K-Means)
fig, ax = plt.subplots(figsize=(8, 5))
sil_by_k = kmeans_df[kmeans_df['config'].str.contains('k-means')][['config', 'silhouette']].copy()
sil_by_k['K'] = sil_by_k['config'].str.extract(r'K=(\d+)').astype(int)
ax.bar(sil_by_k['K'], sil_by_k['silhouette'], color=['#2196F3', '#4CAF50', '#FF9800', '#E91E63'],
       alpha=0.8, edgecolor='white', linewidth=1.5)
ax.set_xlabel('Número de Clusters (K)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score por K — K-Means (init=k-means++)', fontsize=13, fontweight='bold')
ax.set_xticks(sil_by_k['K'])
for _, row in sil_by_k.iterrows():
    ax.annotate(f'{row["silhouette"]:.4f}', (row["K"], row["silhouette"]),
                textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_silhouette_kmeans.png'), bbox_inches='tight')
plt.close()
print("[OK] Gráfico salvo: 07_silhouette_kmeans.png")

# ============================================================================
# 4.2 AGGLOMERATIVE CLUSTERING
# ============================================================================
print("\n--- 4.2 Agglomerative Clustering ---")
agglo_results = []
agglo_models = {}
n_clusters_range = [2, 3, 4, 5]
linkage_methods = ['ward', 'complete', 'average', 'single']

for n_c in n_clusters_range:
    for link in linkage_methods:
        agglo = AgglomerativeClustering(n_clusters=n_c, linkage=link)
        labels_agglo = agglo.fit_predict(X_scaled)
        config_name = f"n_clusters={n_c}, linkage={link}"
        metricas = calcular_metricas(X_scaled, labels_agglo, config_name)
        agglo_results.append(metricas)
        agglo_models[config_name] = {'model': agglo, 'labels': labels_agglo}

        print(f"  {config_name}: Sil={metricas['silhouette']:.4f}, "
              f"DB={metricas['davies_bouldin']:.4f}, CH={metricas['calinski_harabasz']:.1f}")

agglo_df = pd.DataFrame(agglo_results)
print("\nResultados Agglomerative Clustering:")
print(agglo_df.to_string(index=False))

# Gráfico 08: Dendrograma (usando linkage ward nos dados padronizados)
# Amostragem para visualização legível
np.random.seed(RANDOM_STATE)
sample_size = min(200, len(X_scaled))
sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_idx]

Z = linkage(X_sample, method='ward')
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, ax=ax, truncate_mode='level', p=5,
           leaf_rotation=90, leaf_font_size=8,
           color_threshold=0.7 * max(Z[:, 2]),
           above_threshold_color='gray')
ax.set_title('Dendrograma — Agglomerative Clustering (Ward, amostra de 200)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Amostras')
ax.set_ylabel('Distância')
ax.axhline(y=Z[-2, 2], color='red', linestyle='--', linewidth=1.2,
           label=f'Corte sugerido (3 clusters): {Z[-2, 2]:.2f}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_dendrograma.png'), bbox_inches='tight')
plt.close()
print("[OK] Gráfico salvo: 08_dendrograma.png")

# ============================================================================
# 4.3 DBSCAN
# ============================================================================
print("\n--- 4.3 DBSCAN ---")
dbscan_results = []
dbscan_models = {}
eps_range = [0.3, 0.5, 0.7, 1.0, 1.5]
min_samples_range = [3, 5, 10, 15]

for eps_val in eps_range:
    for ms in min_samples_range:
        db = DBSCAN(eps=eps_val, min_samples=ms)
        labels_db = db.fit_predict(X_scaled)
        config_name = f"eps={eps_val}, min_samples={ms}"
        metricas = calcular_metricas(X_scaled, labels_db, config_name)
        dbscan_results.append(metricas)
        dbscan_models[config_name] = {'model': db, 'labels': labels_db}

        n_clusters_found = metricas['n_clusters']
        n_noise_found = metricas['n_noise']
        sil_str = f"{metricas['silhouette']:.4f}" if not np.isnan(metricas['silhouette']) else "N/A"
        print(f"  {config_name}: clusters={n_clusters_found}, ruído={n_noise_found}, Sil={sil_str}")

dbscan_df = pd.DataFrame(dbscan_results)
print("\nResultados DBSCAN:")
print(dbscan_df.to_string(index=False))

# Gráfico 09: K-Distance para escolha de eps (k=5)
k_neighbors = 5
nn = NearestNeighbors(n_neighbors=k_neighbors)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_dist = np.sort(distances[:, k_neighbors - 1])[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(k_dist)), k_dist, color='#2196F3', linewidth=1.5)
ax.set_xlabel('Pontos (ordenados por distância)')
ax.set_ylabel(f'Distância ao {k_neighbors}º vizinho mais próximo')
ax.set_title(f'K-Distance Plot (k={k_neighbors}) — Para escolha de eps',
             fontsize=13, fontweight='bold')
ax.axhline(y=0.7, color='red', linestyle='--', linewidth=1.2, label='eps=0.7')
ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=1.2, label='eps=1.0')
ax.axhline(y=1.5, color='green', linestyle='--', linewidth=1.2, label='eps=1.5')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_kdistance.png'), bbox_inches='tight')
plt.close()
print("[OK] Gráfico salvo: 09_kdistance.png")

# ========================= SEÇÃO 4: AVALIAÇÃO COMPARATIVA ====================
print("\n" + "=" * 70)
print("5. AVALIAÇÃO COMPARATIVA")
print("=" * 70)

# Selecionar o melhor modelo de cada algoritmo (pelo Silhouette Score)
def selecionar_melhor(results_df, nome_algo):
    """Seleciona a configuração com melhor Silhouette Score."""
    validos = results_df.dropna(subset=['silhouette'])
    if validos.empty:
        return None
    melhor = validos.loc[validos['silhouette'].idxmax()]
    print(f"\nMelhor {nome_algo}: {melhor['config']}")
    print(f"  Silhouette: {melhor['silhouette']:.4f}")
    print(f"  Davies-Bouldin: {melhor['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz: {melhor['calinski_harabasz']:.1f}")
    return melhor

def selecionar_melhor_dbscan(results_df, total_pontos, max_ruido_pct=0.50):
    """Seleciona melhor DBSCAN com restrição de cobertura mínima.

    O DBSCAN com parâmetros muito restritivos (eps pequeno) pode gerar
    Silhouette artificialmente alto ao classificar a maioria dos pontos
    como ruído. Esta função impõe um limite máximo de ruído para
    garantir que o modelo selecionado tenha utilidade prática.
    """
    validos = results_df.dropna(subset=['silhouette']).copy()
    if validos.empty:
        print(f"\n[!] DBSCAN: Nenhuma configuração válida encontrada.")
        return None

    validos['ruido_pct'] = validos['n_noise'] / total_pontos
    cobertos = validos[validos['ruido_pct'] <= max_ruido_pct]

    if cobertos.empty:
        print(f"\n[!] DBSCAN: Nenhuma configuração com ≤{max_ruido_pct:.0%} de ruído.")
        cobertos = validos.nsmallest(3, 'ruido_pct')

    melhor = cobertos.loc[cobertos['silhouette'].idxmax()]
    print(f"\nMelhor DBSCAN (restrição: ≤{max_ruido_pct:.0%} ruído): {melhor['config']}")
    print(f"  Silhouette: {melhor['silhouette']:.4f}")
    print(f"  Davies-Bouldin: {melhor['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz: {melhor['calinski_harabasz']:.1f}")
    print(f"  Ruído: {int(melhor['n_noise'])} pontos ({melhor['ruido_pct']:.1%})")
    return melhor

best_km = selecionar_melhor(kmeans_df, 'K-Means')
best_agglo = selecionar_melhor(agglo_df, 'Agglomerative')
best_dbscan = selecionar_melhor_dbscan(dbscan_df, len(X_scaled), max_ruido_pct=0.50)

# Tabela comparativa
print("\n\n=== TABELA COMPARATIVA DOS MELHORES MODELOS ===")
comparacao = pd.DataFrame({
    'Algoritmo': ['K-Means', 'Agglomerative', 'DBSCAN'],
    'Configuração': [best_km['config'] if best_km is not None else 'N/A',
                     best_agglo['config'] if best_agglo is not None else 'N/A',
                     best_dbscan['config'] if best_dbscan is not None else 'N/A'],
    'Silhouette': [best_km['silhouette'] if best_km is not None else np.nan,
                   best_agglo['silhouette'] if best_agglo is not None else np.nan,
                   best_dbscan['silhouette'] if best_dbscan is not None else np.nan],
    'Davies-Bouldin': [best_km['davies_bouldin'] if best_km is not None else np.nan,
                       best_agglo['davies_bouldin'] if best_agglo is not None else np.nan,
                       best_dbscan['davies_bouldin'] if best_dbscan is not None else np.nan],
    'Calinski-Harabasz': [best_km['calinski_harabasz'] if best_km is not None else np.nan,
                          best_agglo['calinski_harabasz'] if best_agglo is not None else np.nan,
                          best_dbscan['calinski_harabasz'] if best_dbscan is not None else np.nan],
})
print(comparacao.to_string(index=False))

# Gráfico 13: Comparação de métricas entre algoritmos
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Comparação de Métricas — Melhores Modelos por Algoritmo', fontsize=14, fontweight='bold')

metrics_names = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
algo_colors = ['#2196F3', '#4CAF50', '#FF9800']
for i, metric in enumerate(metrics_names):
    vals = comparacao[metric].values
    bars = axes[i].bar(comparacao['Algoritmo'], vals, color=algo_colors, alpha=0.8,
                       edgecolor='white', linewidth=1.5)
    axes[i].set_title(metric, fontweight='bold')
    axes[i].set_ylabel(metric)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            axes[i].annotate(f'{val:.3f}', (bar.get_x() + bar.get_width() / 2, val),
                             textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUTPUT_DIR, '13_comparacao_metricas.png'), bbox_inches='tight')
plt.close()
print("[OK] Gráfico salvo: 13_comparacao_metricas.png")

# ========================= GRÁFICOS DE CLUSTERS NO PCA =======================
print("\n--- Gráficos de Clusters no espaço PCA ---")

def plot_clusters_pca(X_pca, labels, titulo, filename):
    """Plota clusters no espaço PCA 2D."""
    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = sorted(set(labels))
    cluster_colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_labels), 3)))

    for idx, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        name = f'Cluster {label}' if label != -1 else 'Ruído'
        color = 'gray' if label == -1 else cluster_colors[idx]
        alpha = 0.3 if label == -1 else 0.6
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color], label=name, alpha=alpha, s=20, edgecolor='none')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f"[OK] Gráfico salvo: {filename}")

# K-Means melhor
if best_km is not None:
    best_km_labels = kmeans_models[best_km['config']]['labels']
    plot_clusters_pca(X_pca, best_km_labels,
                      f"K-Means — {best_km['config']}", '10_clusters_kmeans.png')

# Agglomerative melhor
if best_agglo is not None:
    best_agglo_labels = agglo_models[best_agglo['config']]['labels']
    plot_clusters_pca(X_pca, best_agglo_labels,
                      f"Agglomerative — {best_agglo['config']}", '11_clusters_agglomerative.png')

# DBSCAN melhor
if best_dbscan is not None:
    best_dbscan_labels = dbscan_models[best_dbscan['config']]['labels']
    plot_clusters_pca(X_pca, best_dbscan_labels,
                      f"DBSCAN — {best_dbscan['config']}", '12_clusters_dbscan.png')

# ========================= SEÇÃO 5: INTERPRETAÇÃO ============================
print("\n" + "=" * 70)
print("6. INTERPRETAÇÃO — CLUSTERS × RÓTULOS ORIGINAIS")
print("=" * 70)

def interpretar_clusters(labels, risk_labels, nome_algo, config_name):
    """Cruza clusters encontrados com rótulos reais e exibe tabela de contingência."""
    print(f"\n{'='*50}")
    print(f"  {nome_algo} — {config_name}")
    print(f"{'='*50}")

    # Tabela de contingência
    ct = pd.crosstab(labels, risk_labels, margins=True, margins_name='Total')
    print(f"\nTabela de Contingência (Cluster × RiskLevel):")
    print(ct.to_string())

    # Percentual por cluster
    ct_pct = pd.crosstab(labels, risk_labels, normalize='index') * 100
    print(f"\nDistribuição percentual por cluster (%):")
    print(ct_pct.round(1).to_string())

    # Perfil de cada cluster (média das features originais)
    df_interp = df_clean.copy()
    df_interp['Cluster'] = labels
    df_interp['RiskLevel'] = risk_labels
    perfil = df_interp.groupby('Cluster')[features_numericas].mean()
    print(f"\nPerfil médio de cada cluster:")
    print(perfil.round(2).to_string())

    return ct, ct_pct

# Interpretar os 3 melhores modelos
results_interp = {}

if best_km is not None:
    ct_km, ct_pct_km = interpretar_clusters(
        best_km_labels, risk_labels, 'K-Means', best_km['config'])
    results_interp['K-Means'] = {'ct': ct_km, 'ct_pct': ct_pct_km, 'labels': best_km_labels}

if best_agglo is not None:
    ct_agglo, ct_pct_agglo = interpretar_clusters(
        best_agglo_labels, risk_labels, 'Agglomerative', best_agglo['config'])
    results_interp['Agglomerative'] = {'ct': ct_agglo, 'ct_pct': ct_pct_agglo, 'labels': best_agglo_labels}

if best_dbscan is not None:
    ct_db, ct_pct_db = interpretar_clusters(
        best_dbscan_labels, risk_labels, 'DBSCAN', best_dbscan['config'])
    results_interp['DBSCAN'] = {'ct': ct_db, 'ct_pct': ct_pct_db, 'labels': best_dbscan_labels}

# Gráfico 14: Heatmap do crosstab (melhor modelo geral)
# Determinar melhor modelo geral pelo Silhouette
melhor_geral_nome = comparacao.loc[comparacao['Silhouette'].idxmax(), 'Algoritmo']
print(f"\n\n[*] Melhor modelo geral (por Silhouette): {melhor_geral_nome}")

fig, axes_interp = plt.subplots(1, len(results_interp), figsize=(6 * len(results_interp), 5))
if len(results_interp) == 1:
    axes_interp = [axes_interp]

for idx, (algo_name, data) in enumerate(results_interp.items()):
    ct_pct_plot = data['ct_pct']
    # Reordenar colunas
    col_order = [c for c in ['low risk', 'mid risk', 'high risk'] if c in ct_pct_plot.columns]
    ct_pct_plot = ct_pct_plot[col_order]

    sns.heatmap(ct_pct_plot, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=0.5, ax=axes_interp[idx], vmin=0, vmax=100,
                cbar_kws={'label': '%'})
    axes_interp[idx].set_title(f'{algo_name}', fontweight='bold', fontsize=12)
    axes_interp[idx].set_xlabel('RiskLevel (real)')
    axes_interp[idx].set_ylabel('Cluster')

plt.suptitle('Distribuição de RiskLevel por Cluster (%)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUTPUT_DIR, '14_crosstab_heatmap.png'), bbox_inches='tight')
plt.close()
print("[OK] Gráfico salvo: 14_crosstab_heatmap.png")

# ========================= RESUMO FINAL =====================================
print("\n" + "=" * 70)
print("RESUMO FINAL")
print("=" * 70)
print(f"\nDataset: Maternal Health Risk Data Set")
print(f"Instâncias utilizadas: {len(df_clean)} (após remoção de {n_hr_outliers} outliers extremos)")
print(f"Features: {features_numericas}")
print(f"\nMelhor modelo por algoritmo:")
if best_km is not None:
    print(f"  K-Means:       {best_km['config']} | Sil={best_km['silhouette']:.4f}")
if best_agglo is not None:
    print(f"  Agglomerative: {best_agglo['config']} | Sil={best_agglo['silhouette']:.4f}")
if best_dbscan is not None:
    print(f"  DBSCAN:        {best_dbscan['config']} | Sil={best_dbscan['silhouette']:.4f}")
print(f"\n[*] Melhor modelo geral: {melhor_geral_nome}")

print(f"\nGráficos salvos em: {OUTPUT_DIR}")
print(f"Total de gráficos: 15")
print("\n[OK] Pipeline concluído com sucesso!")

# ========================= EXPORTAR RESULTADOS EM CSV ========================
kmeans_df.to_csv(os.path.join(OUTPUT_DIR, 'resultados_kmeans.csv'), index=False)
agglo_df.to_csv(os.path.join(OUTPUT_DIR, 'resultados_agglomerative.csv'), index=False)
dbscan_df.to_csv(os.path.join(OUTPUT_DIR, 'resultados_dbscan.csv'), index=False)
comparacao.to_csv(os.path.join(OUTPUT_DIR, 'comparacao_melhores.csv'), index=False)
print("[OK] Tabelas de resultados exportadas em CSV na pasta graficos/")

