# -*- coding: utf-8 -*-
"""
Grupo 1 - Segmentação de Imagens: Técnicas Baseadas em Descontinuidades
Disciplina: Tópicos Especiais em Visão Computacional - UFPI/Picos

Este script implementa e demonstra, sobre imagens sintéticas geradas
localmente, as três técnicas de detecção de descontinuidades discutidas
na Seção 10.2 do livro-texto (Gonzalez & Woods, 3ª ed.):

    1) Detecção de pontos isolados  (Seção 10.2.2) -> máscara Laplaciana
    2) Detecção de linhas           (Seção 10.2.3) -> máscaras direcionais
    3) Detecção de bordas           (Seção 10.2.5) -> Roberts, Prewitt, Sobel

Bibliotecas de terceiros usadas (apenas para operações genéricas de
arranjo/filtragem/plot - a lógica de cada técnica é implementada
explicitamente com as máscaras do livro):
    numpy        -> manipulação de arrays/imagens
    scipy.ndimage.correlate -> aplica a máscara (correlação espacial,
                     igual ao "R = soma w_k * z_k" da Eq. 10.2-3)
    matplotlib   -> visualização dos resultados

Para rodar:
    pip install numpy scipy matplotlib
    python deteccao_descontinuidades.py

As figuras geradas são salvas na pasta ./saida/
"""

import os
import numpy as np
from scipy.ndimage import correlate
import matplotlib.pyplot as plt

SAIDA = "saida"
os.makedirs(SAIDA, exist_ok=True)
rng = np.random.default_rng(7)


# ---------------------------------------------------------------------
# 1) DETECÇÃO DE PONTOS ISOLADOS (Seção 10.2.2 / Exemplo 10.1)
# ---------------------------------------------------------------------
# Máscara Laplaciana (Fig. 10.4a) - soma dos coeficientes = 0
MASCARA_LAPLACIANA = np.array([
    [1,  1, 1],
    [1, -8, 1],
    [1,  1, 1],
], dtype=float)


def detectar_pontos_isolados(img, limiar_relativo=0.9):
    """Detecta pontos isolados com a máscara Laplaciana (Eq. 10.2-7/10.2-8).

    Parameters
    ----------
    img : ndarray 2D
        Imagem de entrada (níveis de cinza, float).
    limiar_relativo : float
        Fração do maior valor absoluto de resposta usada como limiar T
        (no livro, T é escolhido como uma fração do maior |R(x,y)|).

    Returns
    -------
    R : ndarray
        Resposta da máscara em cada pixel (R = soma w_k z_k).
    pontos : ndarray de bool
        Máscara binária g(x,y) com os pontos detectados.
    """
    R = correlate(img, MASCARA_LAPLACIANA, mode="nearest")
    T = limiar_relativo * np.max(np.abs(R))
    pontos = np.abs(R) >= T
    return R, pontos


# ---------------------------------------------------------------------
# 2) DETECÇÃO DE LINHAS (Seção 10.2.3 / Exemplo 10.3)
# ---------------------------------------------------------------------
# As quatro máscaras direcionais da Fig. 10.6 (peso 2 na direção
# preferencial; coeficientes somam zero)
MASCARAS_LINHA = {
    "horizontal (0°)": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=float),
    "+45°": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=float),
    "vertical (90°)": np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=float),
    "-45°": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=float),
}


def detectar_linhas(img, direcao="+45°", limiar_relativo=0.55, manter_so_positivos=True):
    """Detecta linhas em uma direção específica (Exemplo 10.3).

    Como a máscara Laplaciana direcional produz o efeito de "linha dupla"
    (valores positivos de um lado, negativos do outro - Seção 10.2.3),
    por padrão mantemos apenas os valores positivos antes de limiarizar.
    """
    mask = MASCARAS_LINHA[direcao]
    R = correlate(img, mask, mode="nearest")
    Rp = np.clip(R, 0, None) if manter_so_positivos else np.abs(R)
    T = limiar_relativo * Rp.max()
    linhas = Rp >= T
    return R, linhas


def direcao_dominante(img):
    """Para cada pixel, retorna qual das 4 direções tem maior |resposta|
    (regra de decisão discutida no Exemplo 10.3: se |Rk| > |Rj| para
    todo j != k, o pixel pertence, com maior probabilidade, a uma linha
    na direção k)."""
    respostas = {nome: correlate(img, m, mode="nearest") for nome, m in MASCARAS_LINHA.items()}
    nomes = list(respostas.keys())
    pilha = np.stack([np.abs(respostas[n]) for n in nomes], axis=0)
    idx = np.argmax(pilha, axis=0)
    return idx, nomes


# ---------------------------------------------------------------------
# 3) DETECÇÃO DE BORDAS: ROBERTS, PREWITT E SOBEL (Seção 10.2.5)
# ---------------------------------------------------------------------
OPERADORES = {
    "Roberts": (
        np.array([[-1, 0], [0, 1]], dtype=float),
        np.array([[0, -1], [1, 0]], dtype=float),
    ),
    "Prewitt": (
        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float),
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
    ),
    "Sobel": (
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float),
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
    ),
}


def gradiente(img, operador="Sobel"):
    """Calcula gx, gy e a magnitude M(x,y) = sqrt(gx^2 + gy^2) (Eq. 10.2-10)
    para o operador de gradiente escolhido (Roberts, Prewitt ou Sobel)."""
    gx_mask, gy_mask = OPERADORES[operador]
    gx = correlate(img, gx_mask, mode="nearest")
    gy = correlate(img, gy_mask, mode="nearest")
    M = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, M


def magnitude_aproximada(gx, gy):
    """Versão computacionalmente mais leve da magnitude (Eq. 10.2-20),
    sem raiz quadrada nem potenciação: M(x,y) ~= |gx| + |gy|."""
    return np.abs(gx) + np.abs(gy)


# ---------------------------------------------------------------------
# Utilitários para gerar as imagens sintéticas de demonstração
# ---------------------------------------------------------------------
def imagem_ponto_isolado(n=140):
    bg = np.full((n, n), 0.55)
    yy, xx = np.mgrid[0:n, 0:n]
    bg += 0.03 * np.sin(xx / 11.0) * np.cos(yy / 13.0)  # leve textura suave
    img = bg.copy()
    img[65, 78] = 1.0  # pixel isolado, bem mais claro que o fundo
    return img


def imagem_linhas(n=160):
    img = np.full((n, n), 0.08)

    def traca(p0, p1, largura=1):
        y0, x0 = p0
        y1, x1 = p1
        passos = int(max(abs(y1 - y0), abs(x1 - x0))) * 3
        for t in np.linspace(0, 1, passos):
            y = int(round(y0 + t * (y1 - y0)))
            x = int(round(x0 + t * (x1 - x0)))
            for dy in range(-(largura // 2), largura // 2 + 1):
                for dx in range(-(largura // 2), largura // 2 + 1):
                    if 0 <= y + dy < n and 0 <= x + dx < n:
                        img[y + dy, x + dx] = 1.0

    traca((20, 20), (140, 140))   # diagonal (+45°)
    traca((30, 130), (30, 10))    # horizontal
    traca((10, 100), (150, 100))  # vertical
    return img


def imagem_formas(n=180):
    Y, X = np.mgrid[0:n, 0:n]
    img = np.full((n, n), 0.25)
    img[30:90, 30:90] = 0.65                              # quadrado
    circulo = (Y - 120) ** 2 + (X - 60) ** 2 <= 38 ** 2
    img[circulo] = 0.9                                     # círculo
    for y in range(20, 150):                                # triângulo
        meia_base = int((y - 20) * 0.55)
        img[y, max(0, 140 - meia_base):min(n, 140 + meia_base)] = 0.45
    img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)  # ruído leve
    return img


# ---------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------
def demo_pontos():
    img = imagem_ponto_isolado()
    R, pontos = detectar_pontos_isolados(img, limiar_relativo=0.9)
    ys, xs = np.where(pontos)
    print(f"[Pontos] {len(ys)} ponto(s) isolado(s) detectado(s) em: "
          f"{list(zip(ys.tolist(), xs.tolist()))}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Entrada")
    axes[1].imshow(img, cmap="gray"); axes[1].scatter(xs, ys, s=80,
                    facecolors="none", edgecolors="red", linewidths=2)
    axes[1].set_title("Ponto(s) detectado(s)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "01_pontos_isolados.png"), dpi=150)
    plt.close(fig)


def demo_linhas():
    img = imagem_linhas()
    R, linhas = detectar_linhas(img, direcao="+45°", limiar_relativo=0.55)
    print(f"[Linhas] pixels detectados na direção +45°: {linhas.sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Entrada")
    destaque = np.stack([img * 0.4] * 3, axis=-1)
    ys, xs = np.where(linhas)
    destaque[ys, xs] = [1, 0.15, 0.15]
    axes[1].imshow(destaque); axes[1].set_title("Linhas a +45° detectadas")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "02_linhas_direcionais.png"), dpi=150)
    plt.close(fig)


def demo_bordas():
    img = imagem_formas()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Entrada (com ruído)")
    for ax, nome in zip(axes[1:], OPERADORES.keys()):
        gx, gy, M = gradiente(img, nome)
        M = (M - M.min()) / (M.max() - M.min() + 1e-9)
        ax.imshow(M, cmap="gray")
        ax.set_title(f"{nome}  |M(x,y)|")
        print(f"[Bordas] {nome}: resposta máxima = {M.max():.3f}, "
              f"média = {M.mean():.4f}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "03_bordas_roberts_prewitt_sobel.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    demo_pontos()
    demo_linhas()
    demo_bordas()
    print(f"\nFiguras salvas em ./{SAIDA}/")
