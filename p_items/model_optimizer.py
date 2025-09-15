# model_optimizer.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_final_model():
    """
    Entrena el modelo de Bosques Aleatorios utilizando los parámetros óptimos
    encontrados previamente y lo guarda en un archivo.
    """
    # Cargamos el dataset y separamos los datos
    df_model_wine = pd.read_csv("dataset/wine_ready.csv")
    X = df_model_wine.drop('quality_binary', axis=1)
    y = df_model_wine['quality_binary']

    # Dividimos los datos para un entrenamiento y testeo final
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Creamos el modelo de Bosques Aleatorios con los parámetros óptimos
    # No se usa GridSearchCV ya que los parámetros ya son conocidos
    best_model = RandomForestClassifier(
        n_estimators=170,
        max_depth=20,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )

    # Entrenamos el modelo con el conjunto de entrenamiento
    print("Entrenando el modelo con los parámetros óptimos...")
    best_model.fit(X_train, y_train)
    print("Entrenamiento finalizado.")

    # Guardamos el modelo entrenado
    joblib.dump(best_model, 'best_random_forest_model.pkl')
    
    print("\n--- Modelo Guardado ---")
    print("El modelo final, entrenado con los parámetros óptimos,")
    print("ha sido guardado como 'best_random_forest_model.pkl'.")

if __name__ == '__main__':
    train_final_model()