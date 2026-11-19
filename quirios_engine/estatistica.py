"""
Estatística para decidir com evidência (sem dependência do scipy).

    comparar_pareado(diferencas, correcao)  -> ganho médio, IC 95%, p-valor

Com correcao = n_teste / n_treino, é o teste t de reamostragem corrigido de
Nadeau & Bengio (2003): na validação cruzada repetida as dobras se
sobrepõem, e o teste t comum subestima a variância e acha "vantagem" onde
não há. Com correcao = 0, é o teste t pareado comum (amostras independentes,
como sorteios avaliados num conjunto de teste fixo).
"""
import math

import numpy as np


def _betacf(a, b, x, iters=300, eps=3e-16):
    """Fração contínua da beta incompleta (Lentz modificado)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c; c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c; c = c if abs(c) > tiny else tiny
        dl = d * c
        h *= dl
        if abs(dl - 1) < eps:
            break
    return h


def beta_incompleta(a, b, x):
    """I_x(a, b) regularizada."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(ln) * _betacf(a, b, x) / a
    return 1 - math.exp(ln) * _betacf(b, a, 1 - x) / b


def t_sf(t, gl):
    """P(T > t) para T ~ t de Student com gl graus de liberdade."""
    if t < 0:
        return 1 - t_sf(-t, gl)
    return 0.5 * beta_incompleta(gl / 2, 0.5, gl / (gl + t * t))


def t_critico(gl, confianca=0.95):
    """t tal que P(|T| < t) = confianca (bissecção)."""
    alvo = (1 - confianca) / 2
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_sf(mid, gl) > alvo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def comparar_pareado(diferencas, correcao=0.0, confianca=0.95):
    """Diferenças pareadas (ex.: acurácia quântico - clássico por dobra).

    Devolve dict com ganho médio, IC, estatística t e p-valor UNILATERAL
    (hipótese: ganho > 0)."""
    d = np.asarray(diferencas, float)
    k = len(d)
    media = float(d.mean()) if k else 0.0
    if k < 2:
        return {"ganho": media, "ic": (media, media), "t": 0.0, "p_valor": 1.0, "n": k}
    var = float(d.var(ddof=1)) * (1 / k + correcao)
    if var <= 0:
        p = 0.0 if media > 0 else 1.0
        return {"ganho": media, "ic": (media, media), "t": math.inf if media > 0 else 0.0,
                "p_valor": p, "n": k}
    ep = math.sqrt(var)
    t = media / ep
    tc = t_critico(k - 1, confianca)
    return {"ganho": media, "ic": (media - tc * ep, media + tc * ep), "t": t,
            "p_valor": float(t_sf(t, k - 1)), "n": k}
