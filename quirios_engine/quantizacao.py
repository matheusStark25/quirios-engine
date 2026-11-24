"""
Quantização de pesos guiada pelos dados, com correção de posto baixo.

É a régua que faltava: hoje um modelo de linguagem é encolhido guardando cada
peso com poucos bits (8, 4 ou 3) em vez de 16. O QURIOS compara as próprias
compressões (rede tensorial, posto baixo) contra isso e oferece três níveis:

    intN                  arredondamento por grupo (RTN), o padrão mais simples
    intN+dados            antes de arredondar, cada canal de entrada ganha uma
                          escala escolhida pelas ativações reais (a ideia do AWQ,
                          Lin et al. 2024); o critério é o erro na SAÍDA da
                          camada, tr(dW C dW^T) com C = E[x x^T], e não nos pesos
    intN+dados+correcao   e mais uma correção de posto baixo do erro que sobrou,
                          medida também na saída (o SVD branqueado do SVD-LLM);
                          custa cerca de meio bit a mais por peso

O tamanho é contado em bits de verdade: N bits por peso, 16 bits pela escala e
N pelo zero de cada grupo, 16 por escala de canal e por parâmetro da correção.
Os pesos ficam guardados como códigos inteiros; a camada reconstrói a matriz
na hora (e a guarda em inferência), então a qualidade medida é exatamente a
do formato guardado. O ganho medido aqui é de TAMANHO guardado (disco,
download, transmissão): para gastar menos memória e ganhar velocidade na
execução é preciso um núcleo de cálculo com inteiros, que esta versão não tem.
"""
import torch
from torch import nn

from .ia_tensorial import svd_branqueado

ALFAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _grupo(entrada, grupo):
    """Maior divisor de `entrada` que não passa de `grupo` (camadas de 576 usam grupos de 96)."""
    if not grupo or grupo >= entrada:
        return entrada
    return next(g for g in range(grupo, 0, -1) if entrada % g == 0)


def quantizar_grupos(W, bits, grupo=128):
    """Arredondamento assimétrico por grupo de `grupo` entradas consecutivas de cada linha.
    O zero sempre cabe na grade (como no GPTQ), e a escala é guardada em 16 bits."""
    if not 2 <= bits <= 8:
        raise ValueError("bits deve estar entre 2 e 8")
    saida, entrada = W.shape
    g = _grupo(entrada, grupo)
    Wg = W.float().reshape(saida, entrada // g, g)
    mn = Wg.amin(-1, keepdim=True).clamp(max=0)
    mx = Wg.amax(-1, keepdim=True).clamp(min=0)
    niveis = 2 ** bits - 1
    escala = ((mx - mn) / niveis).clamp_min(1e-8).half()
    e = escala.float()
    zero = torch.round(-mn / e).clamp(0, niveis)
    q = torch.clamp(torch.round(Wg / e) + zero, 0, niveis)
    return q.to(torch.uint8), escala, zero.to(torch.uint8)


def desquantizar(q, escala, zero):
    return ((q.float() - zero.float()) * escala.float()).reshape(q.shape[0], -1)


def erro_na_saida(dW, C):
    """tr(dW C dW^T): o erro quadrático médio que dW causa na saída, para entradas com E[x x^T] = C."""
    return float(((dW @ C) * dW).sum())


class CamadaQuantizada(nn.Module):
    """y = W' x + U (V x) + b, com W' = desquantizar(códigos) / escala_do_canal."""

    def __init__(self, q, escala, zero, bits, canal=None, U=None, V=None, vies=None):
        super().__init__()
        self.saida, self.entrada = q.shape[0], q.shape[1] * q.shape[2]
        self.bits = bits
        self.register_buffer("q", q)
        self.register_buffer("escala", escala)
        self.register_buffer("zero", zero)
        self.register_buffer("canal", canal)
        self.U = nn.Parameter(U.detach().clone().float()) if U is not None else None
        self.V = nn.Parameter(V.detach().clone().float()) if V is not None else None
        self.vies = nn.Parameter(vies.detach().clone().float()) if vies is not None else None
        self.alfa = 0.0
        self._W = None

    @classmethod
    def de_linear(cls, lin, bits=4, grupo=128, cov=None, dados=False, posto_correcao=0, alfas=ALFAS):
        """Quantiza uma nn.Linear treinada. Com `dados`, a escala dos canais é escolhida pelo
        erro na saída (precisa de `cov`); com `posto_correcao` > 0, soma a correção de posto baixo."""
        if (dados or posto_correcao) and cov is None:
            raise ValueError("a versão guiada pelos dados precisa das covariâncias das entradas (cov)")
        W = lin.weight.detach().float()
        canal, alfa = None, 0.0
        if dados:
            C = cov.to(W.device, torch.float32)
            rms = C.diagonal().clamp_min(1e-12).sqrt()
            melhor = None
            for a in alfas:
                s = rms ** a
                s = (s / (s.max() * s.min()).sqrt()).half().float()      # a escala guardada é de 16 bits
                q, e, z = quantizar_grupos(W * s, bits, grupo)
                err = erro_na_saida(W - desquantizar(q, e, z) / s, C)
                if melhor is None or err < melhor[0]:
                    melhor = (err, a, q, e, z, s)
            _, alfa, q, e, z, s = melhor
            canal = s.half() if alfa > 0 else None
            del C
        else:
            q, e, z = quantizar_grupos(W, bits, grupo)
        U = V = None
        if posto_correcao > 0:
            Wq = desquantizar(q, e, z)
            if canal is not None:
                Wq = Wq / canal.float()
            U, V = svd_branqueado(W - Wq, cov, min(posto_correcao, *W.shape))
        g = _grupo(W.shape[1], grupo)
        c = cls(q.reshape(W.shape[0], -1, g).cpu(), e.cpu(), z.cpu(), bits,
                canal.cpu() if canal is not None else None, U, V, lin.bias)
        c.alfa = alfa
        return c

    def peso(self):
        if not torch.is_grad_enabled() and self._W is not None and self._W.device == self.q.device:
            return self._W
        W = desquantizar(self.q, self.escala, self.zero)
        if self.canal is not None:
            W = W / self.canal.float()
        if not torch.is_grad_enabled():
            self._W = W
        return W

    def forward(self, x):
        y = nn.functional.linear(x, self.peso().to(x.dtype), self.vies)
        if self.U is not None:
            y = y + (x @ self.V.T) @ self.U.T
        return y

    def tamanho_bits(self):
        b = self.q.numel() * self.bits + self.escala.numel() * (16 + self.bits)
        if self.canal is not None:
            b += 16 * self.canal.numel()
        return b + sum(16 * p.numel() for p in (self.U, self.V, self.vies) if p is not None)

    def extra_repr(self):
        corr = f", correção posto {self.U.shape[1]}" if self.U is not None else ""
        return f"entrada={self.entrada}, saida={self.saida}, bits={self.bits}, alfa={self.alfa:g}{corr}"


def bits_do_modelo(modelo, bits_comum=16):
    """Tamanho do modelo em bits: camadas quantizadas pelo formato guardado, o resto em bf16."""
    total, vistos = 0, set()
    for m in modelo.modules():
        if isinstance(m, CamadaQuantizada):
            total += m.tamanho_bits()
            vistos.update(id(p) for p in m.parameters())
    return total + sum(bits_comum * p.numel() for p in modelo.parameters() if id(p) not in vistos)


def posto_para_bits(entrada, saida, bits_extra):
    """Posto da correção que custa cerca de `bits_extra` bits a mais por peso da camada."""
    return max(1, round(bits_extra * entrada * saida / (16 * (entrada + saida))))


def ler_metodo(metodo):
    """'int4' -> (4, False, False); 'int4+dados' -> (4, True, False); 'int3+dados+correcao' -> (3, True, True).
    Devolve None se `metodo` não é de quantização."""
    partes = metodo.split("+")
    if not partes[0].startswith("int"):
        return None
    extras = set(partes[1:])
    try:
        bits = int(partes[0][3:])
    except ValueError:
        raise ValueError(f"método de quantização inválido: {metodo!r}") from None
    if not 2 <= bits <= 8 or not extras <= {"dados", "correcao"} or len(extras) != len(partes) - 1:
        raise ValueError(f"método de quantização inválido: {metodo!r}")
    return bits, "dados" in extras, "correcao" in extras
