# -*- coding: utf-8 -*-
"""
Grupo 1 - Segmentação de Imagens: Técnicas Baseadas em Descontinuidades
Disciplina: Tópicos Especiais em Visão Computacional - UFPI/Picos

Este script implementa e demonstra as técnicas de detecção de
descontinuidades e de aguçamento do livro-texto (Gonzalez & Woods, 3ª ed.):

    1) Detecção de pontos isolados  (Seção 10.2.2) -> máscara Laplaciana
    2) Detecção de linhas           (Seção 10.2.3) -> máscaras direcionais
    3) Detecção de bordas           (Seção 10.2.5) -> Roberts, Prewitt, Sobel
    4) Algoritmo de Canny           (Seção 10.2.6) -> suavização + gradiente
                                     + supressão não-máxima + histerese
    5) Filtros de aguçamento        (Seção 3.6)    -> Laplaciano, máscara de
                                     nitidez, high-boost e gradiente
    6) Laplaciano para bordas       (Seções 3.6.1 e 10.2) -> LoG + cruzamento
                                     por zero

As demonstrações 1 a 4 usam imagens sintéticas geradas localmente; a 5 usa
uma imagem real (Lua, do scikit-image) e, se ele não estiver instalado,
cai para a imagem sintética de formas.

O algoritmo de Canny é implementado "na mão" (sem cv2.Canny), reutilizando
o operador de Sobel já definido para a Seção 10.2.5, exatamente para deixar
explícitas as cinco etapas.

Bibliotecas de terceiros usadas (apenas para operações genéricas de
arranjo/filtragem/plot - a lógica de cada técnica é implementada
explicitamente com as máscaras do livro):
    numpy        -> manipulação de arrays/imagens
    scipy.ndimage.correlate -> aplica a máscara (correlação espacial,
                     igual ao "R = soma w_k * z_k" da Eq. 10.2-3)
    matplotlib   -> visualização dos resultados
    scikit-image -> (opcional) só para carregar a imagem real da Lua

Para rodar:
    pip install numpy scipy matplotlib scikit-image
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
# 4) ALGORITMO DE CANNY (Seção 10.2.6)
# ---------------------------------------------------------------------
# O Canny foi construído a partir do modelo de borda em degrau. As cinco
# etapas abaixo seguem a formulação clássica de Canny (1986), extensão
# natural do que já foi visto em 10.2.5 (gradiente/Sobel):
#
#   1) suavização Gaussiana (reduz ruído antes de derivar - Seção 10.2.5);
#   2) gradiente por Sobel -> magnitude M(x,y) e direção theta(x,y);
#   3) supressão não-máxima (afina a borda ao longo da direção do
#      gradiente, mantendo só o pico local - resolve o problema da
#      "borda grossa" de derivadas de 1ª ordem discutido na Seção 10.2.1);
#   4) limiarização dupla (separa bordas "fortes" e "fracas");
#   5) ligação de bordas por histerese (uma borda fraca só sobrevive se
#      conectada a uma borda forte).

def suavizar_gaussiana(img, sigma=1.0, tamanho=9):
    """Etapa 1: suaviza a imagem com um núcleo Gaussiano 2-D antes de
    calcular derivadas, reduzindo a sensibilidade ao ruído citada na
    Seção 10.2.1 (Exemplo 10.4)."""
    eixo = np.arange(tamanho) - tamanho // 2
    xx, yy = np.meshgrid(eixo, eixo)
    nucleo = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    nucleo /= nucleo.sum()
    return correlate(img, nucleo, mode="nearest")


def supressao_nao_maxima(M, direcao):
    """Etapa 3: para cada pixel, mantém M(x,y) somente se for o maior
    valor entre os dois vizinhos na direção do gradiente (perpendicular
    à borda); caso contrário, zera. Isso afina a borda "grossa" que
    detectores de 1ª ordem produzem (ver conclusões da Seção 10.2.1)."""
    n, m = M.shape
    Mf = np.zeros_like(M)
    angulo = np.rad2deg(direcao) % 180  # direção da borda em [0, 180)

    Mp = np.pad(M, 1, mode="edge")
    for i in range(n):
        for j in range(m):
            a = angulo[i, j]
            ip, jp = i + 1, j + 1  # índices na matriz com padding
            if a < 22.5 or a >= 157.5:          # ~horizontal (0°)
                viz1, viz2 = Mp[ip, jp + 1], Mp[ip, jp - 1]
            elif a < 67.5:                       # ~+45°
                viz1, viz2 = Mp[ip - 1, jp + 1], Mp[ip + 1, jp - 1]
            elif a < 112.5:                       # ~vertical (90°)
                viz1, viz2 = Mp[ip - 1, jp], Mp[ip + 1, jp]
            else:                                 # ~-45°
                viz1, viz2 = Mp[ip - 1, jp - 1], Mp[ip + 1, jp + 1]

            if M[i, j] >= viz1 and M[i, j] >= viz2:
                Mf[i, j] = M[i, j]
    return Mf


def limiarizacao_dupla(M, percentil_alto=92, percentil_baixo=75):
    """Etapa 4: classifica cada pixel em borda forte, fraca ou não-borda.
    Em vez de usar uma fração do maior valor de M (como o limiar T das
    seções 10.2.2/10.2.3), usamos percentis da distribuição dos valores
    já afinados pela supressão não-máxima (a maioria é zero); isso evita
    que um único pixel de resposta muito alta distorça o limiar relativo,
    problema mais comum em Canny do que nos detectores mais simples."""
    nao_nulos = M[M > 0]
    alto = np.percentile(nao_nulos, percentil_alto)
    baixo = np.percentile(nao_nulos, percentil_baixo)
    fortes = M >= alto
    fracas = (M >= baixo) & (M < alto)
    return fortes, fracas


def ligacao_por_histerese(fortes, fracas):
    """Etapa 5: uma borda fraca só é mantida se estiver 8-conectada a
    uma borda forte (direta ou transitivamente); o restante é descartado."""
    bordas = fortes.copy()
    candidatos = fracas.copy()
    mudou = True
    while mudou:
        mudou = False
        ys, xs = np.where(candidatos)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 1), min(bordas.shape[0], y + 2)
            x0, x1 = max(0, x - 1), min(bordas.shape[1], x + 2)
            if bordas[y0:y1, x0:x1].any():
                bordas[y, x] = True
                candidatos[y, x] = False
                mudou = True
    return bordas


def canny(img, sigma=1.5, tamanho_nucleo=9, percentil_alto=90, percentil_baixo=72):
    """Executa as cinco etapas do algoritmo de Canny (Seção 10.2.6) e
    retorna (bordas_binarias, magnitude_apos_supressao)."""
    suave = suavizar_gaussiana(img, sigma=sigma, tamanho=tamanho_nucleo)
    gx, gy, M = gradiente(suave, operador="Sobel")
    direcao = np.arctan2(gy, gx)
    M_fina = supressao_nao_maxima(M, direcao)
    fortes, fracas = limiarizacao_dupla(M_fina, percentil_alto, percentil_baixo)
    bordas = ligacao_por_histerese(fortes, fracas)
    return bordas, M_fina


# ---------------------------------------------------------------------
# 5) FILTROS ESPACIAIS DE AGUÇAMENTO (Seção 3.6)
# ---------------------------------------------------------------------
# Fig. 3.37(a): Laplaciano de 4 vizinhos (MASCARA_LAPLACIANA acima é a de
# 8 vizinhos, Fig. 3.37b)
MASCARA_LAPLACIANA_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)


def laplaciano(img, mascara=MASCARA_LAPLACIANA):
    """Laplaciano discreto (Eq. 3.6-6 e extensão com diagonais)."""
    return correlate(img, mascara, mode="nearest")


def agucar_laplaciano(img, mascara=MASCARA_LAPLACIANA):
    """Seção 3.6.2, Eq. 3.6-7: g = f + c * (Laplaciano de f).
    c = -1 se o coeficiente central da máscara é negativo (Fig. 3.37 a, b);
    c = +1 se é positivo (Fig. 3.37 c, d). Retorna (g, laplaciano)."""
    c = -1.0 if mascara[1, 1] < 0 else 1.0
    lap = laplaciano(img, mascara)
    return np.clip(img + c * lap, 0, 255), lap


def ajustar_para_exibicao(x):
    """Soma o valor mínimo e reescala para [0, 255] (Eq. 2.6-10 e 2.6-11).
    Só para exibir o Laplaciano, que tem valores negativos."""
    return (x - x.min()) / (x.max() - x.min() + 1e-9) * 255


def mascara_nitidez(img, k=1.0, sigma=3.0, tamanho=5):
    """Seção 3.6.3. Eq. 3.6-8: máscara = f - f_borrada;
    Eq. 3.6-9: g = f + k * máscara.
    k = 1: máscara de nitidez; k > 1: high-boost; k < 1: atenua a máscara.
    Padrão do Exemplo 3.16: gaussiana 5x5 com sigma = 3.
    Retorna (g, máscara)."""
    mascara = img - suavizar_gaussiana(img, sigma=sigma, tamanho=tamanho)
    return np.clip(img + k * mascara, 0, 255), mascara


def gradiente_agucamento(img):
    """Seção 3.6.4: magnitude aproximada do gradiente de Sobel,
    M ~ |gx| + |gy| (Eq. 3.6-12)."""
    gx, gy, _ = gradiente(img, operador="Sobel")
    return magnitude_aproximada(gx, gy)


# ---------------------------------------------------------------------
# 6) LAPLACIANO PARA BORDAS: LoG + CRUZAMENTO POR ZERO
# ---------------------------------------------------------------------
def cruzamentos_por_zero(L, limiar):
    """Marca o pixel se o Laplaciano muda de sinal entre vizinhos opostos
    (horizontal, vertical e diagonais) com diferença maior que o limiar.
    O limiar descarta cruzamentos fracos causados por ruído."""
    H, W = L.shape
    Lp = np.pad(L, 1, mode="edge")
    bordas = np.zeros(L.shape, dtype=bool)
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a = Lp[1 - dy:1 - dy + H, 1 - dx:1 - dx + W]
        b = Lp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        bordas |= (a * b < 0) & (np.abs(a - b) > limiar)
    return bordas


def bordas_laplaciano(img, sigma=0.0, tamanho=11, limiar_relativo=0.2):
    """Bordas por cruzamento por zero do Laplaciano. Com sigma > 0 é o LoG
    (suaviza com gaussiana antes de derivar). Retorna (bordas, laplaciano)."""
    base = suavizar_gaussiana(img, sigma=sigma, tamanho=tamanho) if sigma > 0 else img
    L = laplaciano(base)
    return cruzamentos_por_zero(L, limiar_relativo * np.abs(L).max()), L


# ---------------------------------------------------------------------
# Utilitários para gerar as imagens de demonstração
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


def carregar_imagem_real():
    """Imagem da Lua do scikit-image (níveis 0-255); sem ele, usa a imagem
    sintética de formas."""
    try:
        from skimage import data
        return data.moon().astype(float)
    except Exception:
        print("[aviso] scikit-image indisponível: usando imagem sintética.")
        return imagem_formas() * 255


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


def demo_canny():
    img = imagem_formas()
    _, _, M_sobel = gradiente(img, operador="Sobel")
    bordas, M_fina = canny(img, sigma=1.5, tamanho_nucleo=9,
                           percentil_alto=90, percentil_baixo=72)
    print(f"[Canny] pixels de borda mantidos após histerese: {bordas.sum()} "
          f"(de {M_fina.size} pixels)")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Entrada (com ruído)")
    axes[1].imshow(M_sobel / M_sobel.max(), cmap="gray")
    axes[1].set_title("Sobel |M(x,y)| — sem afinamento")
    axes[2].imshow(bordas, cmap="gray")
    axes[2].set_title("Canny (suavização + NMS + histerese)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "04_canny.png"), dpi=150)
    plt.close(fig)


def demo_agucamento():
    f = suavizar_gaussiana(carregar_imagem_real(), sigma=1.2, tamanho=7)  # levemente borrada
    g_lap, lap = agucar_laplaciano(f)
    g_nit, _ = mascara_nitidez(f, k=1.0)
    g_hb, _ = mascara_nitidez(f, k=4.5)
    M = gradiente_agucamento(f)

    paineis = [
        (f, "Original (levemente borrada)", 255),
        (ajustar_para_exibicao(lap), "Laplaciano (com ajuste)", 255),
        (g_lap, "f − ∇²f  (c = −1, máscara 8-viz.)", 255),
        (g_nit, "Máscara de nitidez (k = 1)", 255),
        (g_hb, "High-boost (k = 4,5)", 255),
        (M, "Gradiente de Sobel  |gx|+|gy|", 0.6 * M.max()),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.1))
    for ax, (im, titulo, vmax) in zip(axes.ravel(), paineis):
        ax.imshow(im, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(titulo, fontsize=9, fontweight="bold", color="#8B2E3C")
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(SAIDA, "05_agucamento.png"), dpi=200)
    plt.close(fig)
    print("[Aguçamento] figura salva: 05_agucamento.png")


def demo_laplaciano_bordas():
    img = imagem_formas()
    b_sem, L_sem = bordas_laplaciano(img, sigma=0.0, limiar_relativo=0.2)
    b_log, _ = bordas_laplaciano(img, sigma=2.0, limiar_relativo=0.2)
    print(f"[Laplaciano] cruzamentos por zero sem suavização: {b_sem.sum()} | "
          f"LoG (sigma=2): {b_log.sum()}")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Entrada (com ruído)")
    axes[1].imshow(ajustar_para_exibicao(L_sem), cmap="gray")
    axes[1].set_title("Laplaciano (com ajuste)")
    axes[2].imshow(b_sem, cmap="gray"); axes[2].set_title("Cruzamento por zero, sem suavizar")
    axes[3].imshow(b_log, cmap="gray"); axes[3].set_title("LoG (σ = 2) + cruzamento por zero")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "06_laplaciano_bordas.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    demo_pontos()
    demo_linhas()
    demo_bordas()
    demo_canny()
    demo_agucamento()
    demo_laplaciano_bordas()
    print(f"\nFiguras salvas em ./{SAIDA}/")