"""
Mini modelo de linguagem NATIVO em rede tensorial, treinado do zero na CPU.

    python -m quirios_engine.llm_nativo --passos 600

Comprimir um modelo já treinado é difícil (os pesos não têm a estrutura da
rede tensorial; ver llm_tensorial.py). Aqui o modelo NASCE em rede tensorial:
um transformer pequeno (atenção causal + MLP) que aprende a escrever
caractere a caractere no WikiText-2. Quatro versões, mesmo treino, mesmos
dados, mesma semente:

    denso                            a referência, largura cheia
    TT (rede tensorial)              todas as camadas lineares dos blocos em TT
    posto baixo                      as mesmas camadas como U V, mesmos parâmetros da TT
    denso estreito                   largura reduzida até ter os mesmos parâmetros da TT

A métrica é bits por caractere no texto de teste (menor é melhor; quanto o
modelo "se surpreende" com o próximo caractere).
"""
import argparse
import math
import time

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as e:                      # pragma: no cover
    raise ImportError('O LLM nativo exige PyTorch: pip install -e ".[ia]" datasets') from e

from .ia_tensorial import CamadaPostoBaixo, CamadaTT, contar_params, posto_equivalente


def fabrica(tipo, posto=16, d=3):
    def linear(a, b):
        if tipo == "tt":
            return CamadaTT(a, b, posto, d)
        if tipo == "posto_baixo":
            return CamadaPostoBaixo(a, b, posto_equivalente(a, b, posto, d))
        return nn.Linear(a, b)
    return linear


class Bloco(nn.Module):
    def __init__(self, largura, cabecas, linear):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(largura), nn.LayerNorm(largura)
        self.qkv, self.saida = linear(largura, 3 * largura), linear(largura, largura)
        self.fc1, self.fc2 = linear(largura, 4 * largura), linear(4 * largura, largura)
        self.cabecas = cabecas

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=-1)
        f = lambda t: t.view(B, T, self.cabecas, C // self.cabecas).transpose(1, 2)
        a = nn.functional.scaled_dot_product_attention(f(q), f(k), f(v), is_causal=True)
        x = x + self.saida(a.transpose(1, 2).reshape(B, T, C))
        return x + self.fc2(nn.functional.gelu(self.fc1(self.ln2(x))))


class MiniGPT(nn.Module):
    def __init__(self, vocab, largura=192, camadas=3, cabecas=4, contexto=128, linear=None):
        super().__init__()
        linear = linear or fabrica("denso")
        self.emb, self.pos = nn.Embedding(vocab, largura), nn.Embedding(contexto, largura)
        self.blocos = nn.ModuleList([Bloco(largura, cabecas, linear) for _ in range(camadas)])
        self.ln = nn.LayerNorm(largura)
        self.contexto = contexto

    def forward(self, ids):
        x = self.emb(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for b in self.blocos:
            x = b(x)
        return self.ln(x) @ self.emb.weight.T                         # saída amarrada à entrada


def dados_caracteres():
    from .llm_tensorial import texto_wikitext
    treino, teste = texto_wikitext("train"), texto_wikitext("test")
    simbolos = sorted(set(treino))
    idx = {c: i + 1 for i, c in enumerate(simbolos)}                    # 0 = símbolo desconhecido
    cod = lambda t: torch.tensor([idx.get(c, 0) for c in t], dtype=torch.long)
    return cod(treino), cod(teste), ["?"] + simbolos


def treinar(modelo, ids, passos, lote=16, seq=128, lr=2e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)
    aquece = max(1, passos // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / aquece if s < aquece else 0.5 * (1 + math.cos(math.pi * (s - aquece) / max(1, passos - aquece))))
    modelo.train()
    dev = next(modelo.parameters()).device
    t0 = time.perf_counter()
    for _ in range(passos):
        ini = torch.randint(0, len(ids) - seq - 1, (lote,), generator=g)
        x = torch.stack([ids[i:i + seq + 1] for i in ini.tolist()]).to(dev)
        perda = nn.functional.cross_entropy(modelo(x[:, :-1]).reshape(-1, modelo.emb.num_embeddings), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        sched.step()
    modelo.eval()
    return time.perf_counter() - t0


@torch.inference_mode()
def bits_por_caractere(modelo, ids, max_chars=60000):
    ids = ids[:max_chars].to(next(modelo.parameters()).device)
    soma, n, T = 0.0, 0, modelo.contexto
    for i in range(0, len(ids) - 1, T):
        x = ids[i:i + T + 1]
        if len(x) < 2:
            break
        soma += float(nn.functional.cross_entropy(modelo(x[None, :-1])[0], x[1:], reduction="sum"))
        n += len(x) - 1
    return soma / n / math.log(2)


@torch.inference_mode()
def escrever(modelo, simbolos, inicio="The ", n=160, temperatura=0.7, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = {c: i for i, c in enumerate(simbolos)}
    ids = [idx.get(c, 0) for c in inicio]
    dev = next(modelo.parameters()).device
    for _ in range(n):
        x = torch.tensor([ids[-modelo.contexto:]], device=dev)
        p = torch.softmax(modelo(x)[0, -1].float().cpu() / temperatura, -1)
        ids.append(int(torch.multinomial(p, 1, generator=g)))
    return "".join(simbolos[i] for i in ids)


def _largura_equivalente(vocab, alvo, camadas, cabecas, contexto):
    """Maior largura (múltipla de cabecas*4) de um MiniGPT denso com até `alvo` parâmetros."""
    melhor = cabecas * 4
    for L in range(cabecas * 4, 1025, cabecas * 4):
        if contar_params(MiniGPT(vocab, L, camadas, cabecas, contexto)) > alvo:
            break                                                # cresce com L: daqui para frente só passa
        melhor = L
    return melhor


def ligacao_equivalente(vocab, alvo, largura, camadas, cabecas, contexto, d=3, maximo=256):
    """Maior ligação TT de um MiniGPT de `largura` com até `alvo` parâmetros (None se nem a ligação 2 cabe)."""
    melhor = None
    for r in range(2, maximo + 1):
        if contar_params(MiniGPT(vocab, largura, camadas, cabecas, contexto, fabrica("tt", r, d))) > alvo:
            break
        melhor = r
    return melhor


def experimento(passos=600, largura=192, camadas=3, cabecas=4, contexto=128, posto=16, d=3, lote=16, seed=0,
                dispositivo="cpu", ao_progresso=None, largura_tt_larga=None):
    """largura_tt_larga: também treina uma TT MAIS LARGA com os mesmos parâmetros da TT comum
    (a rede tensorial deixa ter camadas largas por quase nada; aqui se mede se isso compensa)."""
    treino, teste, simbolos = dados_caracteres()
    V = len(simbolos)
    torch.manual_seed(seed)
    tt_params = contar_params(MiniGPT(V, largura, camadas, cabecas, contexto, fabrica("tt", posto, d)))
    estreita = _largura_equivalente(V, tt_params, camadas, cabecas, contexto)
    configs = [("denso", largura, fabrica("denso")),
               (f"rede tensorial TT (ligação {posto})", largura, fabrica("tt", posto, d)),
               ("posto baixo (mesmos parâmetros)", largura, fabrica("posto_baixo", posto, d)),
               (f"denso estreito (largura {estreita}, mesmos parâmetros)", estreita, fabrica("denso"))]
    if largura_tt_larga:
        if largura_tt_larga % cabecas:
            raise ValueError("largura_tt_larga deve ser múltipla de cabecas")
        r = ligacao_equivalente(V, tt_params, largura_tt_larga, camadas, cabecas, contexto, d)
        if r is None:
            raise ValueError(f"nem a ligação 2 cabe em {tt_params} parâmetros com largura {largura_tt_larga}")
        configs.append((f"rede tensorial TT larga (largura {largura_tt_larga}, ligação {r}, mesmos parâmetros)",
                        largura_tt_larga, fabrica("tt", r, d)))
    linhas = []
    for nome, L, fab in configs:
        torch.manual_seed(seed)
        m = MiniGPT(V, L, camadas, cabecas, contexto, fab).to(dispositivo)
        t = treinar(m, treino, passos, lote, contexto, seed=seed)
        linhas.append({"modelo": nome, "parametros": contar_params(m), "bits_por_caractere": bits_por_caractere(m, teste),
                       "s_treino": t, "amostra": escrever(m, simbolos)})
        if ao_progresso:
            ao_progresso(linhas[-1])
        del m
    base = linhas[0]["parametros"]
    for l in linhas:
        l["compressao"] = base / l["parametros"]
    return {"passos": passos, "largura": largura, "camadas": camadas, "contexto": contexto, "lote": lote,
            "posto": posto, "nucleos": d, "vocabulario": V, "threads": torch.get_num_threads(),
            "dispositivo": str(dispositivo), "linhas": linhas}


def texto_experimento(res):
    onde = "GPU" if str(res.get("dispositivo", "cpu")).startswith("cuda") else f"CPU {res['threads']} threads"
    out = [f"Mini modelo de linguagem caractere a caractere (WikiText-2), largura {res['largura']}, {res['camadas']} "
           f"blocos, {res['passos']} passos de treino iguais para todos, {onde}; bits por caractere no teste "
           f"(menor é melhor)"]
    for l in res["linhas"]:
        out.append(f"  {l['modelo']:48s} {l['parametros']:>9,d} parâm. ({l['compressao']:5.1f}x menor)  "
                   f"{l['bits_por_caractere']:.3f} bits/caractere  ({l['s_treino'] / 60:.1f} min)")
    for l in res["linhas"]:
        out.append(f"\n  [{l['modelo']}] {l['amostra']!r}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.llm_nativo",
                                 description="Mini modelo de linguagem nativo em rede tensorial, comparação justa.")
    ap.add_argument("--passos", type=int, default=600)
    ap.add_argument("--largura", type=int, default=192)
    ap.add_argument("--camadas", type=int, default=3)
    ap.add_argument("--posto", type=int, default=16)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--dispositivo", default="cpu", help="cpu ou cuda (GPU)")
    ap.add_argument("--tt-larga", type=int, default=None, help="também treina uma TT com esta largura e os mesmos parâmetros")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    mostra = lambda l: print(f"  {l['modelo']}: {l['bits_por_caractere']:.3f} bits/caractere "
                             f"({l['s_treino'] / 60:.1f} min)", flush=True)
    res = experimento(a.passos, a.largura, a.camadas, posto=a.posto, lote=a.lote, dispositivo=a.dispositivo,
                      ao_progresso=mostra, largura_tt_larga=a.tt_larga)
    print(texto_experimento(res))
    if a.json:
        import json
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
