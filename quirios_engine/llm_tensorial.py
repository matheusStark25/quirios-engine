"""
Um modelo de linguagem de verdade comprimido por redes tensoriais, na CPU.

    pip install -e ".[ia]" transformers datasets
    python -m quirios_engine.llm_tensorial demo --posto 32 --passos 300
    python -m quirios_engine.llm_tensorial demo --metodo todos          # TT x posto baixo, mesmos parâmetros

O modelo (SmolLM2-135M-Instruct, Apache-2.0) tem 30 blocos; as camadas
lineares da atenção e da MLP somam 106 milhões dos 135 milhões de
parâmetros. Cada uma vira uma rede tensorial (CamadaTT, TT-SVD dos pesos) e o
modelo é "curado" por destilação: aprende a imitar a distribuição de
próximas palavras do modelo original num texto público (WikiText-2, treino).

Para caber em pouca RAM, as respostas do original são calculadas UMA vez
(as k palavras mais prováveis em cada posição) e o próprio modelo é
comprimido no lugar: nunca há duas cópias na memória.

Medidas, sempre contra o original e contra a fatoração de posto baixo com os
mesmos parâmetros: perplexidade no WikiText-2 (teste; menor é melhor),
tamanho, velocidade de geração na CPU e respostas lado a lado.
"""
import argparse
import json
import math
import time

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as e:                      # pragma: no cover
    raise ImportError('O LLM tensorial exige PyTorch: pip install -e ".[ia]" transformers datasets') from e

from .ia_tensorial import CamadaPostoBaixo, CamadaTT, contar_params, posto_equivalente

MODELO_PADRAO = "HuggingFaceTB/SmolLM2-135M-Instruct"
ALVOS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ALVOS_MLP = ("gate_proj", "up_proj", "down_proj")
PERGUNTAS = ("What is the capital of France?",
             "Explain in one sentence what a quantum computer is.",
             "Give me three tips to sleep better.")


def _disp(modelo):
    return next(modelo.parameters()).device


def carregar(nome=MODELO_PADRAO, dispositivo="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(nome)
    try:
        modelo = AutoModelForCausalLM.from_pretrained(nome, dtype=torch.float32)
    except TypeError:                                            # transformers < 4.56 chama de torch_dtype
        modelo = AutoModelForCausalLM.from_pretrained(nome, torch_dtype=torch.float32)
    modelo = modelo.to(dispositivo)
    modelo.eval()
    return tok, modelo


def texto_wikitext(divisao):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=divisao)
    return "\n\n".join(t for t in ds["text"] if t.strip())


def texto_conversas(tok, conjunto="HuggingFaceTB/everyday-conversations-llama3.1-2k", divisao="train_sft"):
    """Conversas no formato de chat do próprio modelo: curar com elas preserva o modo
    conversa (curar só com texto enciclopédico fez o modelo responder vazio)."""
    from datasets import load_dataset
    ds = load_dataset(conjunto, split=divisao)
    return "".join(tok.apply_chat_template(m, tokenize=False) for m in ds["messages"])


def tokens(tok, texto):
    return torch.tensor(tok(texto)["input_ids"], dtype=torch.long)


@torch.inference_mode()
def perplexidade(modelo, ids, janela=512, max_tokens=16384):
    """exp(perda média) em janelas sem sobreposição dos primeiros max_tokens tokens."""
    modelo.eval()
    ids = ids[:max_tokens].to(_disp(modelo))
    soma, n = 0.0, 0
    for i in range(0, len(ids) - 1, janela):
        x = ids[i:i + janela + 1]
        if len(x) < 2:
            break
        out = modelo(x[None, :-1])
        perda = nn.functional.cross_entropy(out.logits[0], x[1:], reduction="sum")
        soma += float(perda)
        n += len(x) - 1
    return math.exp(soma / n)


def alvos_do_professor(modelo, ids, passos, lote=2, seq=192, k=32, seed=0):
    """Janelas de treino e, para cada posição, as k palavras mais prováveis do
    modelo ORIGINAL com suas probabilidades (renormalizadas)."""
    g = torch.Generator().manual_seed(seed)
    inicios = torch.randint(0, len(ids) - seq - 1, (passos, lote), generator=g)
    janelas, top_p, top_i = [], [], []
    dev = _disp(modelo)
    with torch.inference_mode():
        for s in range(passos):
            x = torch.stack([ids[i:i + seq] for i in inicios[s].tolist()])
            p = torch.softmax(modelo(x.to(dev)).logits, -1)
            v, i = p.topk(k, -1)
            janelas.append(x)                                          # alvos ficam na CPU (cabem na RAM)
            top_p.append((v / v.sum(-1, keepdim=True)).half().cpu())
            top_i.append(i.int().cpu())
    return janelas, top_p, top_i


def _lineares(modelo, alvos):
    for i, bloco in enumerate(modelo.model.layers):
        for sub in (bloco.self_attn, bloco.mlp):
            for nome in alvos:
                lin = getattr(sub, nome, None)
                if isinstance(lin, nn.Linear):
                    yield f"{i}.{nome}", sub, nome, lin


@torch.inference_mode()
def covariancias_de_entrada(modelo, ids, alvos=ALVOS, n_seq=16, seq=256, seed=0):
    """X^T X das ENTRADAS de cada camada linear, em textos de calibração (uma passada)."""
    g = torch.Generator().manual_seed(seed)
    inicios = torch.randint(0, len(ids) - seq, (n_seq,), generator=g)
    cov, ganchos = {}, []
    for nome, _, _, lin in _lineares(modelo, alvos):
        def gancho(mod, ent, out, nome=nome):
            x = ent[0].reshape(-1, ent[0].shape[-1])
            cov[nome] = cov.get(nome, 0) + x.T @ x
        ganchos.append(lin.register_forward_hook(gancho))
    try:
        for i in inicios.tolist():
            modelo(ids[i:i + seq][None])
    finally:
        for h in ganchos:
            h.remove()
    return {k: (v / (n_seq * seq)).cpu() for k, v in cov.items()}     # libera a GPU para o próximo modelo


def postobaixo_com_dados(lin, posto, cov, eps=1e-4):
    """SVD branqueado pelas ativações (SVD-LLM, Wang et al. 2024): minimiza o erro
    na SAÍDA da camada, ||W X - U V X||, e não nos pesos. W C = U S V^T com
    C C^T = E[x x^T]; a camada fica U S^(1/2) e S^(1/2) V^T C^-1."""
    W = lin.weight.detach().cpu().double()                      # sempre na CPU, em float64: o mesmo caminho
    cov = cov.detach().cpu().double()                           # com ou sem GPU (a camada nova nasce na CPU)
    S = cov + eps * torch.eye(cov.shape[0], dtype=torch.float64) * cov.diagonal().mean()
    C = torch.linalg.cholesky(S)
    U, s, Vt = torch.linalg.svd(W @ C, full_matrices=False)
    raiz = s[:posto].sqrt()
    c = CamadaPostoBaixo(lin.in_features, lin.out_features, posto, vies=lin.bias is not None)
    c.U.data.copy_((U[:, :posto] * raiz).float())
    c.V.data.copy_(torch.linalg.solve_triangular(C, raiz[:, None] * Vt[:posto], upper=False, left=False).float())
    if lin.bias is not None:
        c.vies.data.copy_(lin.bias.detach())
    return c


def comprimir_llm(modelo, metodo="tt", posto=32, d=3, alvos=ALVOS, cov=None, fator=None):
    """Troca, no lugar, as camadas lineares escolhidas de todos os blocos.
    metodo "tt": CamadaTT (TT-SVD dos pesos), ligação `posto`;
    "svd": posto baixo com os mesmos parâmetros da TT;
    "svd_dados": posto baixo branqueado pelas ativações (precisa de `cov`), com
    `fator` vezes menos parâmetros por camada (ou os mesmos da TT, se fator=None)."""
    if metodo not in ("tt", "svd", "svd_dados"):
        raise ValueError("metodo deve ser 'tt', 'svd' ou 'svd_dados'")
    trocadas = 0
    for nome, sub, attr, lin in list(_lineares(modelo, alvos)):
        a, b = lin.in_features, lin.out_features
        r = (round(a * b / fator / (a + b)) if fator else posto_equivalente(a, b, posto, d))
        r = max(1, min(r, a, b))
        if metodo == "tt":
            nova = CamadaTT.de_linear(lin, posto, d)
        elif metodo == "svd":
            nova = CamadaPostoBaixo.de_linear(lin, r)
        else:
            nova = postobaixo_com_dados(lin, r, cov[nome])
        setattr(sub, attr, nova)
        trocadas += 1
    return trocadas


def curar(modelo, janelas, top_p, top_i, lr=3e-4, peso_ce=0.2, ao_progresso=None, info=None):
    """Destilação: o modelo comprimido aprende a distribuição do original
    (entropia cruzada com as k palavras mais prováveis) + um pouco da
    próxima palavra real. Só os parâmetros das camadas comprimidas treinam.

    Guarda contra divergência: se a perda (média móvel) passa do dobro da
    melhor já vista, ou vira NaN/infinito, os pesos voltam ao último ponto
    bom, o otimizador é zerado e a taxa de aprendizado cai para 30%.
    Quantas vezes isso aconteceu vai para info["reducoes_lr"]."""
    for p in modelo.parameters():
        p.requires_grad_(False)
    treinaveis = []
    for m in modelo.modules():
        if isinstance(m, (CamadaTT, CamadaPostoBaixo)):
            for p in m.parameters():
                p.requires_grad_(True)
                treinaveis.append(p)
    opt = torch.optim.Adam(treinaveis, lr=lr)
    n, aquece = len(janelas), max(1, len(janelas) // 10)            # aquecimento linear, depois cosseno
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / aquece if s < aquece else 0.5 * (1 + math.cos(math.pi * (s - aquece) / max(1, n - aquece))))
    modelo.train()
    dev = _disp(modelo)
    seguro = [p.detach().clone() for p in treinaveis]               # último ponto bom
    media = melhor = None
    reducoes = 0
    t0 = time.perf_counter()
    for s, (x, p, i) in enumerate(zip(janelas, top_p, top_i)):
        x, p, i = x.to(dev), p.to(dev), i.to(dev)
        logp = torch.log_softmax(modelo(x).logits, -1)
        destil = -(p.float() * logp.gather(-1, i.long())).sum(-1).mean()
        ce = nn.functional.nll_loss(logp[:, :-1].reshape(-1, logp.shape[-1]), x[:, 1:].reshape(-1))
        perda = destil + peso_ce * ce
        v = float(perda.detach())
        media = v if media is None else 0.8 * media + 0.2 * v
        if not math.isfinite(v) or (melhor is not None and media > 2 * melhor):
            with torch.no_grad():                                   # divergiu: volta e desacelera
                for q, b in zip(treinaveis, seguro):
                    q.copy_(b)
            opt.state.clear()
            sched.base_lrs = [b * 0.3 for b in sched.base_lrs]
            media, reducoes = melhor, reducoes + 1
            sched.step()
            continue
        if melhor is None or media < melhor:
            melhor = media
            with torch.no_grad():
                for q, b in zip(treinaveis, seguro):
                    b.copy_(q)
        opt.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(treinaveis, 1.0)
        opt.step()
        sched.step()
        if ao_progresso and (s % 25 == 0 or s == len(janelas) - 1):
            ao_progresso(s, v, time.perf_counter() - t0)
    if len(janelas) and melhor is not None:                         # o último passo também é conferido
        with torch.no_grad():
            x, p, i = janelas[-1].to(dev), top_p[-1].to(dev), top_i[-1].to(dev)
            v = float(-(p.float() * torch.log_softmax(modelo(x).logits, -1).gather(-1, i.long())).sum(-1).mean())
            if not math.isfinite(v) or v > 2 * melhor:
                for q, b in zip(treinaveis, seguro):
                    q.copy_(b)
                reducoes += 1
    del seguro
    modelo.eval()
    for p in modelo.parameters():
        p.requires_grad_(True)
    if info is not None:
        info["reducoes_lr"] = reducoes
    return time.perf_counter() - t0


@torch.inference_mode()
def responder(modelo, tok, pergunta, max_novos=48):
    x = tok.apply_chat_template([{"role": "user", "content": pergunta}], add_generation_prompt=True,
                                return_tensors="pt", return_dict=True)["input_ids"].to(_disp(modelo))
    t0 = time.perf_counter()
    out = modelo.generate(x, max_new_tokens=max_novos, do_sample=False, pad_token_id=tok.eos_token_id)
    dt = time.perf_counter() - t0
    novos = out[0][x.shape[1]:]
    return tok.decode(novos, skip_special_tokens=True).strip(), len(novos) / dt


def medir(nome, modelo, tok, ids_teste, perguntas=PERGUNTAS, janela=512, max_tokens=16384):
    respostas, vel = [], []
    for q in perguntas:
        r, v = responder(modelo, tok, q)
        respostas.append({"pergunta": q, "resposta": r})
        vel.append(v)
    return {"modelo": nome, "parametros": contar_params(modelo), "mb_bf16": 2 * contar_params(modelo) / 2 ** 20,
            "perplexidade": perplexidade(modelo, ids_teste, janela, max_tokens),
            "tokens_por_s": float(np.median(vel)), "respostas": respostas}


def posto_tt_equivalente(entrada, saida, fator, d=3):
    """Ligação TT cujo número de parâmetros mais se aproxima de entrada*saida/fator
    (para comparar TT e posto baixo com o mesmo tamanho)."""
    from .ia_tensorial import params_camada_tt
    alvo = entrada * saida / fator
    return min(range(2, 1025), key=lambda r: abs(params_camada_tt(entrada, saida, r, d) - alvo))


def demo(nome=MODELO_PADRAO, metodos=("tt",), posto=32, d=3, passos=300, lote=2, seq=192, lr=3e-4, max_tokens=16384,
         seed=0, ao_progresso=None, saida=None, alvos=ALVOS_MLP, fator=None, dados_cura="wikitext", dispositivo="cpu",
         perguntas=PERGUNTAS):
    """dados_cura: "wikitext" (texto enciclopédico), "conversas" (chat) ou "misto"."""
    torch.manual_seed(seed)
    tok, modelo = carregar(nome, dispositivo)
    ids_teste = tokens(tok, texto_wikitext("test"))
    partes = {"wikitext": [texto_wikitext("train")], "conversas": [texto_conversas(tok)],
              "misto": [texto_wikitext("train"), texto_conversas(tok)]}[dados_cura]
    ids_treino = torch.cat([tokens(tok, t) for t in partes])
    linhas = [medir("original", modelo, tok, ids_teste, perguntas, max_tokens=max_tokens)]
    janelas, top_p, top_i = alvos_do_professor(modelo, ids_treino, passos, lote, seq, seed=seed)
    cov = covariancias_de_entrada(modelo, ids_treino.to(dispositivo), alvos, seed=seed) if "svd_dados" in metodos else None
    del modelo
    for metodo in metodos:
        _, modelo = carregar(nome, dispositivo)
        comprimir_llm(modelo, metodo, posto, d, alvos, cov=cov, fator=fator)
        modelo.to(dispositivo)                                    # as camadas novas nascem na CPU
        rotulo = {"tt": f"TT ligação {posto}", "svd": "posto baixo (mesmos parâmetros)",
                  "svd_dados": f"posto baixo guiado por dados ({fator or 1:g}x por camada)"}[metodo]
        ppl_sem = perplexidade(modelo, ids_teste, max_tokens=max_tokens)
        info = {}
        t = curar(modelo, janelas, top_p, top_i, lr=lr, info=info,
                  ao_progresso=(lambda s, l, dt, m=metodo: ao_progresso(m, s, l, dt)) if ao_progresso else None)
        linha = medir(rotulo, modelo, tok, ids_teste, perguntas, max_tokens=max_tokens)
        linha.update(perplexidade_sem_cura=ppl_sem, s_cura=t, passos_cura=passos, reducoes_lr=info["reducoes_lr"])
        linhas.append(linha)
        if saida:
            torch.save({k: v.detach().to(torch.bfloat16) for k, v in modelo.state_dict().items()},
                       f"{saida}_{metodo}.pt")
        del modelo
    base = linhas[0]
    for l in linhas:
        l["compressao"] = base["parametros"] / l["parametros"]
        l["aceleracao"] = l["tokens_por_s"] / base["tokens_por_s"]
    return {"modelo_base": nome, "posto": posto, "nucleos": d, "passos_cura": passos, "tokens_avaliados": max_tokens,
            "dados_cura": dados_cura, "dispositivo": str(dispositivo), "fator": fator,
            "camadas": list(alvos), "threads": torch.get_num_threads(), "linhas": linhas}


def texto_demo(res):
    out = [f"{res['modelo_base']} na CPU ({res['threads']} threads); perplexidade no WikiText-2 (teste, "
           f"{res['tokens_avaliados']} tokens; menor é melhor)"]
    for l in res["linhas"]:
        extra = (f" (sem cura: {l['perplexidade_sem_cura']:.1f}; cura de {l['passos_cura']} passos em "
                 f"{l['s_cura'] / 60:.1f} min"
                 + (f"; divergiu e desacelerou {l['reducoes_lr']}x" if l.get("reducoes_lr") else "")
                 + ")") if "perplexidade_sem_cura" in l else ""
        out.append(f"  {l['modelo']:32s} {l['parametros'] / 1e6:6.1f} M parâm. ({l['compressao']:.2f}x menor, "
                   f"{l['mb_bf16']:.0f} MB)  perplexidade {l['perplexidade']:6.2f}{extra}  "
                   f"{l['tokens_por_s']:.1f} tokens/s ({l['aceleracao']:.2f}x)")
    for i, q in enumerate(res["linhas"][0]["respostas"]):
        out.append(f"\n  P: {q['pergunta']}")
        for l in res["linhas"]:
            out.append(f"    [{l['modelo']}] {l['respostas'][i]['resposta'][:220]!r}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quirios_engine.llm_tensorial",
                                 description="Modelo de linguagem comprimido por redes tensoriais, com comparação justa.")
    sub = ap.add_subparsers(dest="acao", required=True)
    p = sub.add_parser("demo")
    p.add_argument("--modelo", default=MODELO_PADRAO)
    p.add_argument("--metodo", choices=["tt", "svd", "svd_dados", "todos"], default="svd_dados")
    p.add_argument("--fator", type=float, default=None, help="vezes menos parâmetros por camada (svd_dados)")
    p.add_argument("--posto", type=int, default=32)
    p.add_argument("--nucleos", type=int, default=3)
    p.add_argument("--passos", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alvos", choices=["mlp", "tudo"], default="mlp", help="só a MLP ou também a atenção")
    p.add_argument("--tokens-teste", type=int, default=16384)
    p.add_argument("--saida", default=None, help="prefixo para gravar os pesos comprimidos (bf16)")
    p.add_argument("--json", default=None, help="grava o resultado completo em JSON")
    p.add_argument("--dados-cura", choices=["wikitext", "conversas", "misto"], default="wikitext")
    p.add_argument("--dispositivo", default="cpu", help="cpu ou cuda (GPU)")
    a = ap.parse_args(argv)
    metodos = ("svd_dados", "tt") if a.metodo == "todos" else (a.metodo,)
    prog = lambda m, s, l, dt: print(f"  [{m}] cura passo {s}: perda {l:.3f} ({dt / 60:.1f} min)", flush=True)
    res = demo(a.modelo, metodos, a.posto, a.nucleos, a.passos, lr=a.lr, max_tokens=a.tokens_teste,
               ao_progresso=prog, saida=a.saida, alvos=ALVOS_MLP if a.alvos == "mlp" else ALVOS, fator=a.fator,
               dados_cura=a.dados_cura, dispositivo=a.dispositivo)
    print(texto_demo(res))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
