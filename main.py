import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
# Carregar os dados
df = pd.read_csv("dataset"".csv")
 
# Pré-processamento
df = df.replace('Unknown', np.nan)
df = df.dropna(axis=1, how='all')
 
if 'target' not in df.columns:
    raise ValueError("Coluna 'target' não encontrada no dataset.")
 
X = df.drop('target', axis=1)
y = df['target']
 
# Converter colunas categóricas
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
 
# Tratar valores faltantes
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
 
# Escalonamento
scaler = StandardScaler()
X = scaler.fit_transform(X)
 
# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Lista para guardar os resultados
resultados = []
 
# Função para treinar a MLP com diferentes configurações
def treinar_mlp(hidden_layer_sizes=(50,), activation='relu', learning_rate='constant', max_iter=1000):
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=42
    )
    mlp.fit(X_train, y_train)
 
    # Avaliação
    y_pred = mlp.predict(X_test)
    epocas_usadas = mlp.n_iter_
    acc = accuracy_score(y_test, y_pred)
 
    # Salva o resultado
    resultados.append({
        'arquitetura': hidden_layer_sizes,
        'ativacao': activation,
        'taxa_aprendizado': learning_rate,
        'acuracia': acc,
        'epocas': epocas_usadas
    })
 
    # Resultados formatados
    print("="*60)
    print(f"Arquitetura: {hidden_layer_sizes}")
    print(f"Ativação: {activation}")
    print(f"Taxa de aprendizado: {learning_rate}")
    print(f"Max_iter configurado: {max_iter}")
    print(f"Épocas usadas até parada: {epocas_usadas}")
    print(f"Acurácia: {acc:.6f}")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred, digits=6))
    print("="*60)
 
# Testar variações da arquitetura e parâmetros
arquiteturas = [
    (10,), (20,), (50,), (100,), (200,),
    (10, 10), (20, 20), (50, 50)
]
ativacoes = ['relu', 'logistic']
taxas_aprendizado = ['constant', 'adaptive']
max_iter = 1000
 
for arquitetura in arquiteturas:
    for ativacao in ativacoes:
        for taxa in taxas_aprendizado:
            treinar_mlp(hidden_layer_sizes=arquitetura, activation=ativacao, learning_rate=taxa, max_iter=max_iter)
 
# --- Relatório Final ---
 
# Melhor configuração geral
melhor_resultado = max(resultados, key=lambda x: x['acuracia'])
 
# Melhor arquitetura
arquiteturas_unicas = set([r['arquitetura'] for r in resultados])
melhores_arq = {}
for arq in arquiteturas_unicas:
    accs = [r['acuracia'] for r in resultados if r['arquitetura'] == arq]
    melhores_arq[arq] = np.mean(accs)
melhor_arquitetura = max(melhores_arq.items(), key=lambda x: x[1])
 
# Melhor ativação
ativacoes_unicas = set([r['ativacao'] for r in resultados])
melhores_ativ = {}
for ativ in ativacoes_unicas:
    accs = [r['acuracia'] for r in resultados if r['ativacao'] == ativ]
    melhores_ativ[ativ] = np.mean(accs)
melhor_ativacao = max(melhores_ativ.items(), key=lambda x: x[1])
 
# Melhor taxa de aprendizado
taxas_unicas = set([r['taxa_aprendizado'] for r in resultados])
melhores_taxa = {}
for taxa in taxas_unicas:
    accs = [r['acuracia'] for r in resultados if r['taxa_aprendizado'] == taxa]
    melhores_taxa[taxa] = np.mean(accs)
melhor_taxa = max(melhores_taxa.items(), key=lambda x: x[1])
 
# Mostrar análise final
print("\n\n========= RELATÓRIO FINAL =========")
print(f"Melhor combinação geral:")
print(f"Arquitetura: {melhor_resultado['arquitetura']}")
print(f"Ativação: {melhor_resultado['ativacao']}")
print(f"Taxa de aprendizado: {melhor_resultado['taxa_aprendizado']}")
print(f"Acurácia: {melhor_resultado['acuracia']:.4f}")
 
print("\nMelhor arquitetura média:")
print(f"{melhor_arquitetura[0]} com média de acurácia {melhor_arquitetura[1]:.4f}")
 
print("\nMelhor ativação média:")
print(f"{melhor_ativacao[0]} com média de acurácia {melhor_ativacao[1]:.4f}")
 
print("\nMelhor taxa de aprendizado média:")
print(f"{melhor_taxa[0]} com média de acurácia {melhor_taxa[1]:.4f}")
print("=====================================")