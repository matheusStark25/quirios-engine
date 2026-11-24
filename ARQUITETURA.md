# QURIOS — arquitetura lógica

O QURIOS é a camada entre o software de hoje e a computação quântica. Ele
resolve um problema que ainda está chegando — a IA e os dados passando a
viver também em processadores quânticos — mas funciona agora, em qualquer
computador, e prova cada número que mostra.

## O QURIOS melhora uma IA ou um sistema?

As duas coisas, e cada uma com medida:

| O que melhora | Como | Medido |
|---|---|---|
| **Modelos de IA** | Escreve as camadas no formato dos estados quânticos (redes tensoriais) | MNIST: 35× menos parâmetros e acurácia maior que o original (97,46% × 97,34%); na mesma acurácia, 9 a 22× menos parâmetros que a compressão clássica |
| **Velocidade de IA** | Camadas largas em rede tensorial | 4.096×4.096: 6× / 3,9× / 1,6× mais rápida (lotes 1 / 64 / 1000); 8.192×8.192: 30× / 6,6× / 3,5× |
| **Aprendizado com dados quânticos** | Memória quântica (duas cópias medidas juntas) | Simulação: 98% contra 49% do melhor clássico com o mesmo número de experimentos (6 qubits). **Em dois chips reais da IBM** (ibm_marrakesh e ibm_kingston, 4 qubits): 93% contra 69% em 100 estados, McNemar p = 0,00002, comprovantes assinados |
| **Circuitos quânticos** | Evolução dentro de um circuito quântico | Mesmo resultado do circuito padrão com 56% menos portas de 2 qubits |
| **Sistemas** | Decide, com estatística, onde cada tarefa roda (clássico ou quântico); carrega dados clássicos na QPU; protege e prova tudo | Foto 128×128 na QPU com 68× menos portas; 344 testes, incluindo ataques a cada camada de proteção |

Onde ele melhora a IA de hoje sem precisar de computador quântico: a
**quantização guiada pelos dados** (`quantizacao.py`) guarda um modelo de
linguagem em 4 bits por peso com metade da perda do arredondamento comum:
+11% contra +22% no Qwen2.5-0.5B (GPU T4), +7,5% contra +18% no SmolLM2-135M (CPU). O ganho é de tamanho guardado; falta
o núcleo de cálculo com inteiros para ganhar memória e velocidade na execução.

Onde ele **não** melhora, e a página de evidências mostra: fases de Ising
com informação local (o clássico vence); encolher um modelo de linguagem já
treinado com rede tensorial (na GPU, o Qwen 0,5B ficou 1,27× menor e a
perplexidade foi de 18,7 para 28,7, muito atrás da quantização); o adaptador de ajuste fino em rede tensorial, que
perdeu para o LoRA com os mesmos parâmetros na CPU e na GPU (descartado); e um modelo de linguagem
pequeno treinado do zero, em que a rede tensorial vence a fatoração clássica
(3,94 contra 4,24 bits por caractere com os mesmos parâmetros) mas perde para um
modelo denso mais estreito (3,71). As versões maiores desses testes rodam numa
GPU grátis, no Kaggle ou no Colab (`exemplos/kaggle_llm_tensorial.ipynb`).

## Princípios: além de política e de mercado

Mudanças de governo, de fornecedor ou de mercado não devem parar o QURIOS.
Por isso ele é construído sobre o que não pertence a ninguém:

1. **Sem dono único.** Os backends são plugáveis: simulação exata, tensor
   network (MPS, 100 qubits), Qiskit Aer, IBM Quantum e qualquer outro
   registrado (`backends_registrados`). Trocar de provedor de QPU é trocar um
   parâmetro, não reescrever o sistema.
2. **Padrões abertos.** Circuitos em OpenQASM 3, dados em JSON, agentes de IA
   por MCP, criptografia dos padrões públicos do NIST (FIPS 203 e 204).
3. **Funciona sem internet.** Tudo, menos a QPU real, roda no computador do
   usuário: os dados não precisam sair de casa.
4. **Seguro contra o computador quântico de amanhã.** Assinaturas ML-DSA e
   cifra com ML-KEM desde já: o que é guardado hoje não poderá ser aberto
   quando máquinas quânticas quebrarem RSA ("guarde agora, quebre depois").
5. **Prova, não promessa.** Cada execução na QPU gera um comprovante selado;
   cada comparação é justa (mesmo orçamento para clássico e quântico); um
   resultado injusto publicado antes foi retirado.

## As camadas

```
 Software clássico · Python · API HTTP · Painel · Agentes de IA (MCP)
                               │
 ┌─────────────────────────────▼──────────────────────────────┐
 │ PROTEÇÃO (protecao.py)                                      │
 │ identidade com escopos · contenção (taxa, força bruta)      │
 │ trilha de auditoria assinada · cofre pós-quântico           │
 │ guarda das ferramentas de agentes de IA                     │
 └─────────────────────────────┬──────────────────────────────┘
                               │
 INTELIGÊNCIA      orquestrador clássico × quântico (evidência estatística)
                   forja quântica: um circuito desenha a rede de IA
                   evolução quântica: um circuito desenha circuitos
                               │
 IA TENSORIAL      camadas TT · compressão de modelos · LLM nativo e comprimido
 (física quântica  quantização guiada pelos dados · adaptadores de ajuste fino
  na IA clássica)  (ia_tensorial, llm_nativo, llm_tensorial, quantizacao, adaptadores)
                               │
 VANTAGEM          memória quântica · dados quânticos · kernels quânticos
 QUÂNTICA
                               │
 PONTE PARA O      dados clássicos → MPS → circuito · QPU com comprovante
 HARDWARE          mitigação de erro · projeção para hardware futuro
                               │
 EXECUÇÃO          exato · MPS · Aer · IBM · plugins · tarefas assíncronas
```

## Como as peças se encaixam (um fluxo)

1. Um sistema clássico ou um agente de IA chega pela API ou pelo MCP; a
   **proteção** confere quem é, o que pode fazer e registra o pedido.
2. O **orquestrador** decide, com teste estatístico, se a tarefa vale a pena
   no quântico ou se o clássico resolve melhor.
3. Para IA, a **forja quântica** desenha a rede (camadas em rede tensorial,
   tamanho, ligação) e o modelo sai menor e com comprovante dos pesos.
4. Se o dado já é quântico (um sensor, um simulador, outra QPU), a **memória
   quântica** aprende com menos experimentos do que qualquer método clássico.
5. Para rodar no hardware, a **ponte** carrega os dados como estados
   quânticos com poucas portas, executa na QPU escolhida e sela o resultado.
6. Tudo o que foi guardado fica cifrado e assinado; a trilha de auditoria
   mostra o que aconteceu, e qualquer adulteração aparece.

Detalhes de segurança: [SEGURANCA.md](SEGURANCA.md). Resultados e comandos:
[README.pt-BR.md](README.pt-BR.md). Mudanças por versão: [CHANGELOG.md](CHANGELOG.md).
