"""
Avalia um modelo nos ultimos 20% de cada classe (mesma validacao do train.py).
Eduardo Sochodolak e Andrew Bertelli

Uso:  python avaliar.py <pasta_do_modelo> [pasta_de_dados]
      python avaliar.py ../libras
      python avaliar.py pessoas_fundo ../libras/data
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
VAL_FRACTION = 0.2

pasta_modelo = sys.argv[1]
pasta_dados = sys.argv[2] if len(sys.argv) > 2 else os.path.join(pasta_modelo, "data")
modelo = tf.keras.models.load_model(os.path.join(pasta_modelo, "gesture_model.keras"))
classes = json.load(open(os.path.join(pasta_modelo, "class_names.json")))


def fotos_validacao(classe):
    pasta = os.path.join(pasta_dados, classe)
    fotos = sorted(f for f in os.listdir(pasta) if f.lower().endswith(".jpg"))
    corte = int(len(fotos) * (1 - VAL_FRACTION))
    return [os.path.join(pasta, f) for f in fotos[corte:]]


def carregar(caminho):
    img = tf.image.decode_jpeg(tf.io.read_file(caminho), channels=3)
    return tf.image.resize(img, (IMG_SIZE, IMG_SIZE))


avaliadas = [c for c in classes if os.path.isdir(os.path.join(pasta_dados, c))]
caminhos, reais = [], []
for classe in avaliadas:
    fotos = fotos_validacao(classe)
    caminhos += fotos
    reais += [classes.index(classe)] * len(fotos)

lote = tf.data.Dataset.from_tensor_slices(caminhos).map(carregar).batch(32)
previstos = np.argmax(modelo.predict(lote, verbose=0), axis=1)
reais = np.array(reais)

matriz = np.zeros((len(classes), len(classes)), dtype=int)
for r, p in zip(reais, previstos):
    matriz[r, p] += 1

print(f"Modelo: {pasta_modelo}  |  Dados: {pasta_dados}")
print(f"Acuracia: {(previstos == reais).mean():.1%} ({(previstos == reais).sum()}/{len(reais)})")
print("\nMatriz de confusao (linhas = real, colunas = previsto):")
print("      " + " ".join(f"{c[:5]:>5}" for c in classes))
for classe in avaliadas:
    i = classes.index(classe)
    print(f"{classe[:5]:>5} " + " ".join(f"{v:>5}" for v in matriz[i]))
