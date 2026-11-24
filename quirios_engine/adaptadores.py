"""
Ajuste fino de modelos de linguagem com adaptadores em rede tensorial, contra o LoRA.

    pip install -e ".[ia]"
    python -m quirios_engine.adaptadores --passos 300                    # SmolLM2-135M, CPU
    python -m quirios_engine.adaptadores --modelo Qwen/Qwen2.5-0.5B-Instruct --dispositivo cuda

Encolher um modelo pronto é o caso mais difícil para a rede tensorial (os
pesos treinados não têm a estrutura dela). Ajustar um modelo pronto a uma
tarefa nova é o caso oposto: os pesos originais ficam congelados e só uma
correção pequena aprende, y = W x + escala * dW x. Hoje a correção padrão é o
LoRA (Hu et al. 2021), dW = B A de posto r. O adaptador TT escreve dW como uma
rede tensorial (a CamadaTT do QURIOS), que cobre a matriz inteira com muito
menos parâmetros. A ideia vem da pesquisa recente (LoRETTA, Yang et al. 2024;
TT-LoRA, 2024); aqui ela é medida com comparação justa:

    - os mesmos dados, os mesmos passos, a mesma ordem dos exemplos;
    - para cada método, a melhor taxa de aprendizado de uma grade, escolhida
      numa VALIDAÇÃO separada (nunca no teste);
    - um LoRA com o MESMO número de parâmetros treináveis do TT, além do LoRA comum;
    - a perplexidade final num teste que nenhum deles viu.

Tarefa padrão: conversas do dia a dia (everyday-conversations, formato de chat
do próprio modelo).
"""
import argparse
import json
import math
import time

try:
    import torch
    from torch import nn
except ImportError as e:                      # pragma: no cover
    raise ImportError('Os adaptadores exigem PyTorch: pip install -e ".[ia]" transformers datasets') from e

from . import llm_tensorial as lt
from .ia_tensorial import CamadaTT, params_camada_tt


class Adaptador(nn.Module):
    """Uma nn.Linear congelada mais uma correção treinável: LoRA (B A) ou rede tensorial (TT).
    As duas começam em dW = 0, então o modelo ajustado parte exatamente do original."""

    def __init__(self, base, tipo="tt", posto=8, d=4, escala=None):
        super().__init__()
        if tipo not in ("tt", "lora"):
            raise ValueError("tipo deve ser 'tt' ou 'lora'")
        a, b = base.in_features, base.out_features
        self.base, self.tipo, self.posto, self.d = base, tipo, posto, d
        for p in base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        if tipo == "lora":
            self.A = nn.Parameter(torch.randn(posto, a, device=dev) / math.sqrt(a))
            self.B = nn.Parameter(torch.zeros(b, posto, device=dev))
            self.escala = 2.0 if escala is None else escala         # alfa = 2 r, o valor comum
        else:
            self.tt = CamadaTT(a, b, posto, d, vies=False).to(dev)
            with torch.no_grad():
                self.tt.nucleos[-1].zero_()                           # dW = 0 no começo, como no LoRA
            self.escala = 1.0 if escala is None else escala

    def delta(self, x):
        if self.tipo == "lora":
            return (x @ self.A.T) @ self.B.T
        return self.tt(x)

    def forward(self, x):
        return self.base(x) + self.escala * self.delta(x)

    def treinaveis(self):
        return [p for n, p in self.named_parameters() if not n.startswith("base.")]


def instalar(modelo, tipo="tt", posto=8, d=4, alvos=lt.ALVOS, escala=None):
    """Congela o modelo inteiro e põe um adaptador em cada camada linear escolhida dos blocos."""
    for p in modelo.parameters():
        p.requires_grad_(False)
    n = 0
    for _, sub, attr, lin in list(lt._lineares(modelo, alvos)):
        setattr(sub, attr, Adaptador(lin, tipo, posto, d, escala))
        n += 1
    return n


def treinaveis(modelo):
    return [p for m in modelo.modules() if isinstance(m, Adaptador) for p in m.treinaveis()]


def params_treinaveis(modelo, tipo, posto, d=4, alvos=lt.ALVOS):
    """Quantos parâmetros um tipo de adaptador teria neste modelo (sem instalar)."""
    total = 0
    for _, _, _, lin in lt._lineares(modelo, alvos):
        a, b = lin.in_features, lin.out_features
        total += posto * (a + b) if tipo == "lora" else params_camada_tt(a, b, posto, d)
    return total


def posto_lora_equivalente(modelo, posto_tt, d=4, alvos=lt.ALVOS):
    """Posto do LoRA com (aproximadamente) os mesmos parâmetros treináveis do adaptador TT."""
    soma_ab = sum(lin.in_features + lin.out_features for _, _, _, lin in lt._lineares(modelo, alvos))
    return max(1, round(params_treinaveis(modelo, "tt", posto_tt, d, alvos) / soma_ab))


def ajustar(modelo, ids, passos, lote=4, seq=256, lr=1e-3, seed=0, ao_progresso=None):
    """Treina só os adaptadores (AdamW, aquecimento linear e cosseno). A ordem dos exemplos
    depende só de `seed`: todos os métodos veem exatamente os mesmos dados."""
    params = treinaveis(modelo)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    aquece = max(1, passos // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / aquece if s < aquece else 0.5 * (1 + math.cos(math.pi * (s - aquece) / max(1, passos - aquece))))
    g = torch.Generator().manual_seed(seed)
    dev = lt._disp(modelo)
    modelo.train()
    t0 = time.perf_counter()
    perdas = []
    for s in range(passos):
        ini = torch.randint(0, len(ids) - seq - 1, (lote,), generator=g)
        x = torch.stack([ids[i:i + seq + 1] for i in ini.tolist()]).to(dev)
        logits = modelo(x[:, :-1]).logits
        perda = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        perdas.append(float(perda.detach()))
        if ao_progresso and (s % 25 == 0 or s == passos - 1):
            ao_progresso(s, perdas[-1], time.perf_counter() - t0)
    modelo.eval()
    return time.perf_counter() - t0, perdas


def _memoria_gb(dispositivo):
    if str(dispositivo).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2 ** 30
    return None


def demo(nome=lt.MODELO_PADRAO, configs=(("lora", 8), ("tt", 8), ("tt", 4), ("lora", "mesmos")), passos=300,
         lote=4, seq=256, lrs=(3e-4, 1e-3, 3e-3), d=4, alvos=lt.ALVOS, max_tokens=16384, seed=0, dispositivo="cpu",
         ao_progresso=None, ao_linha=None):
    """configs: (tipo, posto); ("lora", "mesmos") é o LoRA com os mesmos parâmetros do primeiro TT da lista."""
    tok, modelo = lt.carregar(nome, dispositivo)
    ids = lt.tokens(tok, lt.texto_conversas(tok))
    corte = int(len(ids) * 0.9)
    treino, val = ids[:corte], ids[corte:]                       # validação: só para escolher a taxa
    teste = lt.tokens(tok, lt.texto_conversas(tok, divisao="test_sft"))
    ppl = lambda m, x: lt.perplexidade(m, x, max_tokens=max_tokens)
    linhas = [{"modelo": "original, sem ajuste", "tipo": None, "posto": None, "treinaveis": 0,
               "ppl_validacao": ppl(modelo, val), "ppl_teste": ppl(modelo, teste)}]
    tt_ref = next((p for t, p in configs if t == "tt"), 8)
    reais = [(t, posto_lora_equivalente(modelo, tt_ref, d, alvos) if p == "mesmos" else p, p == "mesmos")
             for t, p in configs]
    del modelo
    lt.liberar_memoria()

    def resultado():
        return {"modelo_base": nome, "tarefa": "everyday-conversations (chat)", "passos": passos, "lote": lote,
                "seq": seq, "lrs": list(lrs), "nucleos": d, "camadas": list(alvos), "tokens_avaliados": max_tokens,
                "tokens_treino": len(treino), "dispositivo": str(dispositivo), "linhas": linhas}

    for tipo, posto, mesmos in reais:
        tentativas = []
        for lr in lrs:
            torch.manual_seed(seed)
            _, m = lt.carregar(nome, dispositivo)
            instalar(m, tipo, posto, d, alvos)
            n = sum(p.numel() for p in treinaveis(m))
            if str(dispositivo).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            prog = (lambda s, l, dt, r=f"{tipo} {posto} lr {lr:g}": ao_progresso(r, s, l, dt)) if ao_progresso else None
            t, perdas = ajustar(m, treino, passos, lote, seq, lr, seed, prog)
            tentativas.append({"lr": lr, "ppl_validacao": ppl(m, val), "ppl_teste": ppl(m, teste), "s_treino": t,
                               "perda_final": float(sum(perdas[-20:]) / len(perdas[-20:])),
                               "memoria_gb": _memoria_gb(dispositivo), "treinaveis": n})
            del m
            lt.liberar_memoria()
        validas = [x for x in tentativas if math.isfinite(x["ppl_validacao"])] or tentativas
        melhor = min(validas, key=lambda x: x["ppl_validacao"])
        rotulo = ("LoRA" if tipo == "lora" else "adaptador TT") + f" posto {posto}" + (
            " (mesmos parâmetros do TT)" if mesmos else "")
        linhas.append({"modelo": rotulo, "tipo": tipo, "posto": posto, "mesmos_parametros_do_tt": mesmos,
                       "treinaveis": melhor["treinaveis"], "lr": melhor["lr"], "ppl_validacao": melhor["ppl_validacao"],
                       "ppl_teste": melhor["ppl_teste"], "s_treino": melhor["s_treino"],
                       "memoria_gb": melhor["memoria_gb"], "tentativas": tentativas})
        if ao_linha:
            ao_linha(resultado())
    return resultado()


def texto_demo(res):
    onde = "GPU" if res["dispositivo"].startswith("cuda") else "CPU"
    out = [f"Ajuste fino de {res['modelo_base']} em {res['tarefa']}, {res['passos']} passos iguais para todos, {onde}; "
           f"perplexidade no teste (menor é melhor); taxa de cada método escolhida na validação"]
    for l in res["linhas"]:
        extra = (f"  lr {l['lr']:g}, {l['s_treino'] / 60:.1f} min" if l.get("lr") else "")
        out.append(f"  {l['modelo']:44s} {l['treinaveis']:>10,d} parâm. treináveis  "
                   f"perplexidade {l['ppl_teste']:7.3f}{extra}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.adaptadores",
                                 description="Adaptadores em rede tensorial contra LoRA, com comparação justa.")
    ap.add_argument("--modelo", default=lt.MODELO_PADRAO)
    ap.add_argument("--passos", type=int, default=300)
    ap.add_argument("--lote", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--nucleos", type=int, default=4)
    ap.add_argument("--tokens-teste", type=int, default=16384)
    ap.add_argument("--dispositivo", default="cpu")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    prog = lambda r, s, l, dt: print(f"  [{r}] passo {s}: perda {l:.3f} ({dt / 60:.1f} min)", flush=True)
    res = demo(a.modelo, passos=a.passos, lote=a.lote, seq=a.seq, lrs=tuple(a.lrs), d=a.nucleos,
               max_tokens=a.tokens_teste, dispositivo=a.dispositivo, ao_progresso=prog)
    print(texto_demo(res))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
