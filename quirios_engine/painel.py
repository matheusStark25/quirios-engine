"""
Painel ao vivo da evolução quântica, servido pela API em /painel.

Uma sessão roda a EvolucaoQuantica numa thread e publica um evento por
geração (entropia da população, melhor acurácia, ligações entre genes). A
página consome os eventos por streaming (fetch + text/event-stream, com o
cabeçalho X-API-Key, que a API sempre exige) e desenha tudo em SVG, sem
bibliotecas externas -- funciona offline, numa sala de reunião.
"""
import json
import threading
import time
import uuid

import numpy as np

MAX_SESSOES_ATIVAS = 3
DATASETS_PAINEL = ("sintetico", "breast_cancer", "wine", "iris")


def carregar_dados(dataset, n_qubits, n_amostras=96, seed=0):
    """(X, y) já em [0, 1] com n_qubits features."""
    if dataset == "sintetico":                       # XOR: o circuito padrão é fraco aqui
        r = np.random.default_rng(seed)
        X = r.uniform(0, 1, (n_amostras, n_qubits))
        return X, ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(float)
    from .benchmark import _preparar, carregar_dataset
    X, y = carregar_dataset(dataset)
    nq = min(n_qubits, X.shape[1])
    X, _ = _preparar(X, X[:1], nq)
    sel = np.random.default_rng(seed).choice(len(y), min(n_amostras, len(y)), replace=False)
    return X[sel], y[sel].astype(float)


class Sessao:
    def __init__(self, dataset, n_qubits, geracoes, populacao, emaranhamento=True, seed=0):
        if dataset not in DATASETS_PAINEL:
            raise ValueError(f"dataset deve ser um de {DATASETS_PAINEL}")
        self.id = uuid.uuid4().hex[:12]
        self.config = {"dataset": dataset, "n_qubits": n_qubits, "geracoes": geracoes,
                       "populacao": populacao, "emaranhamento": emaranhamento, "seed": seed}
        self.eventos, self.terminou = [], False
        self._cond = threading.Condition()
        self.criada = time.time()

    def publicar(self, evento):
        with self._cond:
            self.eventos.append(json.loads(json.dumps(evento, default=float)))
            self._cond.notify_all()

    def rodar(self):
        from .evolucao_quantica import EvolucaoQuantica
        c = self.config
        try:
            X, y = carregar_dados(c["dataset"], c["n_qubits"], seed=c["seed"])
            self.publicar({"tipo": "inicio", "amostras": len(y), "n_qubits": X.shape[1], **c})
            eq = EvolucaoQuantica(X.shape[1], populacao=c["populacao"], geracoes=c["geracoes"], seed=c["seed"],
                                  emaranhamento=c["emaranhamento"], dobras=4, repeticoes=2,
                                  ao_progresso=self.publicar)
            res = eq.executar(X, y)
            self.publicar({"tipo": "fim", "resumo": eq.resumo(res),
                           "melhor": {"descricao": res["melhor"]["descricao"],
                                      "acuracia": res["melhor"]["acuracia"], "portas_2q": res["melhor"]["custo"],
                                      "simulavel_classicamente": res["melhor"]["simulavel_classicamente"]},
                           "padrao": {"acuracia": res["padrao"]["acuracia"], "portas_2q": res["padrao"]["custo"]},
                           "n_qubits_gerador": res["n_qubits_gerador"], "genomas_possiveis": res["genomas_possiveis"],
                           "tempo_s": res["tempo_s"]})
        except Exception as e:                       # o painel mostra o erro em vez de travar
            self.publicar({"tipo": "erro", "mensagem": f"{type(e).__name__}: {e}"})
        finally:
            with self._cond:
                self.terminou = True
                self._cond.notify_all()

    def fluxo(self, keepalive_s=10.0):
        """Gerador de eventos no formato text/event-stream."""
        i = 0
        while True:
            with self._cond:
                if i >= len(self.eventos) and not self.terminou:
                    self._cond.wait(timeout=keepalive_s)
                novos, fim = self.eventos[i:], self.terminou     # depois de terminar, nada novo chega
            for ev in novos:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            i += len(novos)
            if fim:
                return
            if not novos:
                yield ": mantendo a conexão\n\n"


class GerenciadorSessoes:
    def __init__(self):
        self._sessoes, self._trava = {}, threading.Lock()

    def ativas(self):
        with self._trava:
            return sum(1 for s in self._sessoes.values() if not s.terminou)

    def criar(self, **kw):
        if self.ativas() >= MAX_SESSOES_ATIVAS:
            raise RuntimeError(f"já há {MAX_SESSOES_ATIVAS} evoluções ao vivo rodando; aguarde uma terminar")
        s = Sessao(**kw)
        with self._trava:
            self._sessoes[s.id] = s
        threading.Thread(target=s.rodar, name=f"painel-{s.id}", daemon=True).start()
        return s

    def obter(self, sid):
        with self._trava:
            return self._sessoes.get(sid)


HTML = r"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QURIOS · Evolução quântica ao vivo</title>
<style>
:root{--fundo:#0b0f1a;--cartao:#121829;--borda:#22304d;--texto:#e6ecff;--sutil:#8b98b8;--a1:#7aa2ff;--a2:#5be3c0;--a3:#ffb86b;--a4:#ff6b8b}
*{box-sizing:border-box}body{margin:0;background:var(--fundo);color:var(--texto);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:24px 24px 8px}h1{margin:0;font-size:22px}header p{margin:4px 0 0;color:var(--sutil)}
.controles{display:flex;flex-wrap:wrap;gap:10px;padding:12px 24px;align-items:end}
label{display:flex;flex-direction:column;font-size:12px;color:var(--sutil);gap:4px}
select,input{background:var(--cartao);color:var(--texto);border:1px solid var(--borda);border-radius:8px;padding:7px 9px;font:inherit}
button{background:var(--a1);color:#081028;border:0;border-radius:8px;padding:9px 16px;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.grade{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;padding:12px 24px 24px}
.cartao{background:var(--cartao);border:1px solid var(--borda);border-radius:12px;padding:14px 16px}
.cartao h2{margin:0 0 2px;font-size:14px}.cartao .sub{color:var(--sutil);font-size:12px;margin-bottom:8px}
.numero{font-size:30px;font-weight:700}.linha{display:flex;gap:18px;flex-wrap:wrap}
svg{width:100%;height:170px}.barra{height:10px;border-radius:5px;background:var(--a2)}
.lig{display:grid;grid-template-columns:minmax(0,1fr) minmax(90px,40%);gap:6px 12px;align-items:center;font-size:13px}
#log{font:12px/1.5 ui-monospace,Consolas,monospace;color:var(--sutil);max-height:180px;overflow:auto;white-space:pre-wrap}
.ok{color:var(--a2)}.alerta{color:var(--a3)}.erro{color:var(--a4)}
</style></head><body>
<header><h1>QURIOS · Evolução quântica ao vivo</h1>
<p>A população de circuitos vive em superposição num circuito gerador; medir sorteia indivíduos, a elite gira as amplitudes e o emaranhamento aprende quais genes andam juntos.</p></header>
<div class="controles">
<label>Dados<select id="dataset"><option value="sintetico">Sintético (XOR)</option><option value="breast_cancer">Breast Cancer</option><option value="wine">Wine</option><option value="iris">Iris</option></select></label>
<label>Qubits do modelo<input id="qubits" type="number" value="6" min="2" max="8"></label>
<label>Gerações<input id="geracoes" type="number" value="6" min="1" max="20"></label>
<label>População<input id="populacao" type="number" value="12" min="4" max="48"></label>
<label>Emaranhamento<select id="emar"><option value="1">ligado</option><option value="0">desligado (Han–Kim)</option></select></label>
<label>Chave da API<input id="chave" type="password" placeholder="obrigatória" autocomplete="off"></label>
<button id="iniciar">Iniciar evolução</button></div>
<div class="grade">
<div class="cartao"><h2>Entropia da população</h2><div class="sub">bits · começa com todas as arquiteturas em superposição</div><svg id="g_entropia"></svg></div>
<div class="cartao"><h2>Melhor acurácia (validação cruzada)</h2><div class="sub">linha tracejada = circuito padrão do QURIOS</div><svg id="g_acc"></svg></div>
<div class="cartao"><h2>Ligação máxima entre genes</h2><div class="sub">informação mútua na população · só nasce com emaranhamento</div><svg id="g_lig"></svg></div>
<div class="cartao"><h2>Ligações aprendidas</h2><div class="sub">pares de genes que o circuito aprendeu a escolher juntos</div><div id="ligacoes" class="lig"></div></div>
<div class="cartao"><h2>Estado</h2><div class="linha"><div><div class="sub">geração</div><div class="numero" id="n_ger">–</div></div><div><div class="sub">melhor</div><div class="numero" id="n_acc">–</div></div><div><div class="sub">portas 2q</div><div class="numero" id="n_2q">–</div></div></div><div id="resultado" class="sub"></div></div>
<div class="cartao"><h2>Registro</h2><div id="log"></div></div></div>
<script>
const $=id=>document.getElementById(id);let dados=[],padrao=null;
function grafico(svg,pontos,{cor="#7aa2ff",min=0,max=1,ref=null,fmt=v=>v.toFixed(2),total=6}){
 const W=svg.clientWidth||320,H=170,m={l:38,r:10,t:10,b:22};svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
 const x=i=>m.l+(W-m.l-m.r)*i/Math.max(1,total),y=v=>m.t+(H-m.t-m.b)*(1-(v-min)/(max-min||1));let s="";
 for(let k=0;k<=4;k++){const v=min+(max-min)*k/4;s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="#22304d"/><text x="4" y="${y(v)+4}" fill="#8b98b8" font-size="10">${fmt(v)}</text>`}
 for(let i=0;i<=total;i++)s+=`<text x="${x(i)-3}" y="${H-6}" fill="#8b98b8" font-size="10">${i}</text>`;
 if(ref!==null)s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(ref)}" y2="${y(ref)}" stroke="#ffb86b" stroke-dasharray="5 4"/>`;
 if(pontos.length){s+=`<polyline fill="none" stroke="${cor}" stroke-width="2.5" points="${pontos.map((v,i)=>x(i)+","+y(v)).join(" ")}"/>`;
  pontos.forEach((v,i)=>s+=`<circle cx="${x(i)}" cy="${y(v)}" r="3.5" fill="${cor}"/>`)}
 svg.innerHTML=s}
function log(t,c=""){const d=$("log");d.innerHTML+=`<span class="${c}">${t}</span>\n`;d.scrollTop=d.scrollHeight}
function desenhar(total){const e=dados.map(d=>d.entropia_bits),a=dados.map(d=>d.melhor_acuracia),l=dados.map(d=>d.ligacao_max_bits);
 const amin=Math.min(...a,padrao??1)-0.05;grafico($("g_entropia"),e,{cor:"#7aa2ff",min:0,max:Math.max(16,...e),fmt:v=>v.toFixed(0),total});
 grafico($("g_acc"),a,{cor:"#5be3c0",min:Math.max(0,amin),max:1,ref:padrao,fmt:v=>(100*v).toFixed(0)+"%",total});
 grafico($("g_lig"),l,{cor:"#ffb86b",min:0,max:Math.max(0.3,...l),total})}
async function iniciar(){const b=$("iniciar");b.disabled=true;dados=[];padrao=null;$("log").innerHTML="";$("ligacoes").innerHTML="";$("resultado").textContent="";
 const h={"Content-Type":"application/json"};if($("chave").value)h["X-API-Key"]=$("chave").value;
 const corpo={dataset:$("dataset").value,n_qubits:+$("qubits").value,geracoes:+$("geracoes").value,populacao:+$("populacao").value,emaranhamento:$("emar").value==="1"};
 const total=corpo.geracoes;desenhar(total);
 try{const r=await fetch("/painel/sessoes",{method:"POST",headers:h,body:JSON.stringify(corpo)});const j=await r.json();
  if(!r.ok){log("erro: "+(j.detail||r.status),"erro");b.disabled=false;return}
  const s=await fetch(`/painel/sessoes/${j.id}/eventos`,{headers:h});const leitor=s.body.getReader();const dec=new TextDecoder();let buf="";
  while(true){const {value,done}=await leitor.read();if(done)break;buf+=dec.decode(value,{stream:true});let k;
   while((k=buf.indexOf("\n\n"))>=0){const bloco=buf.slice(0,k);buf=buf.slice(k+2);if(!bloco.startsWith("data: "))continue;tratar(JSON.parse(bloco.slice(6)),total)}}
 }catch(e){log("erro: "+e,"erro")}b.disabled=false}
function tratar(ev,total){
 if(ev.tipo==="inicio")log(`início: ${ev.dataset}, ${ev.amostras} amostras, ${ev.n_qubits} qubits no modelo`);
 if(ev.tipo==="geracao"){dados.push(ev);padrao=ev.padrao_acuracia;desenhar(total);$("n_ger").textContent=`${ev.geracao}/${ev.total_geracoes}`;
  $("n_acc").textContent=(100*ev.melhor_acuracia).toFixed(1)+"%";$("n_2q").textContent=ev.melhor_portas_2q;
  $("ligacoes").innerHTML=(ev.ligacoes||[]).filter(l=>l.bits>1e-3).map(l=>`<div>${l.genes[0]} ↔ ${l.genes[1]}</div><div><div class="barra" style="width:${Math.min(100,200*l.bits)}%"></div></div>`).join("")||'<div class="sub">nenhuma ainda</div>';
  log(`geração ${ev.geracao}: ${(100*ev.melhor_acuracia).toFixed(1)}% · entropia ${ev.entropia_bits.toFixed(1)} bits · ${ev.genomas_distintos} distintos · ${ev.melhor_descricao}`)}
 if(ev.tipo==="fim"){const m=ev.melhor;$("resultado").innerHTML=`<span class="ok">melhor circuito: ${m.descricao}, ${(100*m.acuracia).toFixed(1)}% com ${m.portas_2q} portas de 2 qubits</span> · padrão: ${(100*ev.padrao.acuracia).toFixed(1)}% com ${ev.padrao.portas_2q}`+(m.simulavel_classicamente?' · <span class="alerta">sem emaranhamento: simulável classicamente</span>':"");
  log(`fim em ${ev.tempo_s}s · gerador de ${ev.n_qubits_gerador} qubits, ${ev.genomas_possiveis} arquiteturas em superposição`,"ok")}
 if(ev.tipo==="erro")log("erro: "+ev.mensagem,"erro")}
$("iniciar").onclick=iniciar;window.onresize=()=>desenhar(+$("geracoes").value);desenhar(6);
</script></body></html>"""
