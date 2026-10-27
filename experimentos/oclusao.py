"""
Pergunta 2 - Que partes da imagem o modelo usa para decidir?
Eduardo Sochodolak e Andrew Bertelli

Teste 1 (oclusao): passa um quadrado cinza por cima da foto e mede quanto
a confianca na classe certa cai. Onde cai mais e o que o modelo olha.
Teste 2 (espelho): espelha as fotos de validacao, simulando a outra mao.

Uso:  python oclusao.py      (gera oclusao.png nesta pasta)
"""
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_SIZE = 224
TAMPA = 48
PASSO = 16
SINAIS = ["A", "D", "I", "V", "W", "Y"]

aqui = os.path.dirname(os.path.abspath(__file__))
pasta_libras = os.path.join(aqui, "..", "libras")
modelo = tf.keras.models.load_model(os.path.join(pasta_libras, "gesture_model.keras"))
classes = json.load(open(os.path.join(pasta_libras, "class_names.json")))
pasta_dados = os.path.join(pasta_libras, "data")


def carregar(caminho):
    img = tf.image.decode_jpeg(tf.io.read_file(caminho), channels=3)
    return tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy()


def fotos_da_classe(classe):
    pasta = os.path.join(pasta_dados, classe)
    return [os.path.join(pasta, f) for f in sorted(os.listdir(pasta)) if f.lower().endswith(".jpg")]


def foto_mais_confiante(classe):
    fotos = fotos_da_classe(classe)[::10]
    imagens = np.stack([carregar(f) for f in fotos])
    probs = modelo.predict(imagens, verbose=0)[:, classes.index(classe)]
    return imagens[int(np.argmax(probs))]


def mapa_de_oclusao(imagem, classe):
    alvo = classes.index(classe)
    posicoes = range(0, IMG_SIZE - TAMPA + 1, PASSO)
    tampadas = []
    for y in posicoes:
        for x in posicoes:
            copia = imagem.copy()
            copia[y:y + TAMPA, x:x + TAMPA] = 127
            tampadas.append(copia)
    probs = modelo.predict(np.stack(tampadas), verbose=0)
    original = modelo.predict(imagem[None], verbose=0)[0, alvo]
    queda = (original - probs[:, alvo]).reshape(len(posicoes), len(posicoes))
    pior = int(np.argmax(original - probs[:, alvo]))
    return queda, original, classes[int(np.argmax(probs[pior]))]


figura, eixos = plt.subplots(2, len(SINAIS), figsize=(3 * len(SINAIS), 6.4))
print("Teste 1 - oclusao")
for i, classe in enumerate(SINAIS):
    imagem = foto_mais_confiante(classe)
    queda, original, virou = mapa_de_oclusao(imagem, classe)
    mapa = tf.image.resize(queda[..., None], (IMG_SIZE, IMG_SIZE)).numpy()[..., 0]
    eixos[0, i].imshow(imagem.astype("uint8"))
    eixos[0, i].set_title(f"{classe} ({original:.0%})")
    eixos[1, i].imshow(imagem.astype("uint8"))
    eixos[1, i].imshow(np.clip(mapa, 0, 1), cmap="jet", alpha=0.5, vmin=0, vmax=1)
    eixos[1, i].set_title(f"tapando o pior: {virou}")
    print(f"  {classe}: confianca {original:.0%}, queda maxima {queda.max():.0%}, vira '{virou}'")
for eixo in eixos.ravel():
    eixo.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(aqui, "oclusao.png"), dpi=110)

print("\nTeste 2 - espelho (simula a outra mao)")
acertos_normal, acertos_espelho, total = 0, 0, 0
for classe in classes:
    fotos = fotos_da_classe(classe)
    validacao = np.stack([carregar(f) for f in fotos[int(len(fotos) * 0.8):]])
    alvo = classes.index(classe)
    acertos_normal += (np.argmax(modelo.predict(validacao, verbose=0), axis=1) == alvo).sum()
    acertos_espelho += (np.argmax(modelo.predict(validacao[:, :, ::-1], verbose=0), axis=1) == alvo).sum()
    total += len(validacao)
print(f"  normal:    {acertos_normal / total:.1%}")
print(f"  espelhada: {acertos_espelho / total:.1%}")
print("\nFigura salva em oclusao.png")
