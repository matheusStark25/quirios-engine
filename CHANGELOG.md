# Changelog

## Em andamento (depois da 0.8.0): IA medida contra o que o mercado usa

### O que a primeira rodada na GPU mostrou (Colab T4, 23/11/2026), sem enfeite
- Encolher o Qwen2.5-0.5B com rede tensorial: 1,27× menor, perplexidade de 18,7 para 28,7 (posto baixo guiado pelos dados: 28,4), e a conversa em português se perdeu. A rede tensorial saiu do corte com perplexidade 179 mil e a cura de 2.000 passos a trouxe ao nível do melhor clássico, mas isso **não compete** com a quantização de 4 bits, que encolhe ~4× perdendo pouco. Fica como pesquisa.
- Modelo nativo maior: denso 1,947 e TT (ligação 24, 6,1× menos parâmetros) 2,329 bits por caractere. A sessão caiu antes das duas linhas que decidem (posto baixo e denso estreito com os mesmos parâmetros).
- Corrigido: o texto do resultado dizia "na CPU (1 threads)" para velocidades medidas na GPU (os 33 a 39 tokens/s da parte 1 eram da T4). Agora diz onde a geração foi medida.

### Quantização guiada pelos dados (`quantizacao.py`)
- `intN`: arredondamento por grupos de 128 pesos, com o zero sempre na grade (o padrão do mercado). `intN+dados`: escala por canal de entrada escolhida pelas ativações (a ideia do AWQ), com o erro medido na saída da camada, tr(dW C dWᵀ). `intN+dados+correcao`: soma uma correção de posto baixo do erro que sobrou pelo SVD branqueado (~0,5 bit por peso).
- Tamanho contado em bits de verdade (códigos, escala e zero de cada grupo, escala de canal, correção). `llm_tensorial demo --metodos int8 int4 int4+dados ...` compara tudo lado a lado.
- Medido na CPU (SmolLM2-135M, MLP inteira, perplexidade no WikiText-2, original 22,18): 4 bits comum 26,24 (+18%), guiado pelos dados 25,09, **com correção 23,85 (+7,5%)**, a 4,7 bits por peso. Em 3 bits: comum 55,17 (+149%, respostas embaralhadas), **com dados e correção 32,40 (+46%)**, coerente. O ganho é de tamanho guardado: sem núcleo de cálculo com inteiros, a execução não fica mais leve nem mais rápida.
- **Repetido na GPU** (Colab T4, 24/11/2026, Qwen2.5-0.5B, todas as camadas lineares, original 18,72): 8 bits 18,71; 4 bits comum 22,90 (+22%), guiado pelos dados 21,24, **com correção 20,73 (+11%)** a 4,68 bits por peso; 3 bits comum 76,76 (+310%), **com dados e correção 30,30 (+62%)**. A perda dos 4 bits cai pela metade nos dois modelos. Na GPU a correção custa velocidade (0,73× do original), por ser uma multiplicação a mais sem núcleo próprio.
- `svd_branqueado` virou função própria em `ia_tensorial.py` e é usada pela compressão guiada pelos dados e pela correção.
- Corrigido: `parametros` dos modelos quantizados contava só os parâmetros de ponto flutuante (54,9 milhões em vez de 134,5); os códigos inteiros agora contam.

### Adaptadores de ajuste fino em rede tensorial (`adaptadores.py`)
- `Adaptador`: uma camada linear congelada mais uma correção treinável, LoRA (B A) ou rede tensorial (a `CamadaTT`, com o último núcleo zerado para começar em dW = 0, como o LoRA). `ajustar` treina só a correção com os mesmos dados na mesma ordem para todos; `demo` escolhe a taxa de cada método numa validação separada e inclui um LoRA com o mesmo número de parâmetros do TT (`posto_lora_equivalente`). `python -m quirios_engine.adaptadores --passos 300`.
- Medido na CPU (SmolLM2-135M, conversas do dia a dia, 80 passos, perplexidade no teste; original 3,939): LoRA posto 8 2,585 (2,44 milhões de parâmetros), **LoRA posto 3 com os mesmos parâmetros do TT 2,585**, TT posto 8 2,647 (856 mil), TT posto 4 2,669 (236 mil). Com os mesmos parâmetros **o LoRA ganhou** e treinou 7× mais rápido, e o menor LoRA possível (posto 1, 305 mil parâmetros) deu 2,580, melhor que o TT posto 4. **O adaptador TT perdeu em todas as comparações**: nesta tarefa curta até o posto 1 basta, e o formato da rede tensorial não tem o que oferecer.
- **Repetido na GPU** (Colab T4, 24/11/2026, Qwen2.5-0.5B, 300 passos, taxas 0,001/0,003/0,01; original 7,587): LoRA posto 8 2,421 (4,4 milhões de parâmetros), **LoRA posto 2 com os mesmos parâmetros do TT 2,379**, LoRA posto 1 2,385 (550 mil), TT posto 8 2,469 (883 mil), TT posto 4 2,484 (257 mil). O LoRA ganhou de novo em tudo e o TT treinou 1,5 a 2× mais devagar. **Caminho descartado**; o código fica para reprodução. A sessão caiu no último método (LoRA posto 1), que foi medido de novo numa máquina nova com a mesma semente (as perdas de treino repetiram passo a passo).

### Modelo nativo: a rede tensorial larga
- `llm_nativo --tt-larga 768`: treina também uma TT bem mais larga com os mesmos parâmetros da TT comum (`ligacao_equivalente` acha a maior ligação que cabe). É o teste da vantagem própria da rede tensorial: camadas largas por quase nada.

### Memória no Colab grátis (12,7 GB de RAM)
- Na primeira rodada do notebook novo (24/11/2026), o processo foi morto por falta de RAM (11,8 GB) no 4º método da quantização, depois de 8 bits (perplexidade 18,71 contra 18,72 do original), 4 bits (22,90) e 4 bits guiado pelos dados (21,24). Diagnóstico no próprio Colab: a correção sozinha fica estável (4,6 GB); a RAM vai se acumulando entre um modelo e outro, porque o malloc do Linux guarda o que o Python solta.
- `llm_tensorial.liberar_memoria()` devolve essa memória ao sistema (`malloc_trim`) depois de cada modelo, na quantização e nos adaptadores (15 cargas do modelo por rodada). Na GPU, as covariâncias das ativações (2,4 GB no Qwen 0,5B) ficam na placa, não na RAM. O SVD branqueado não monta mais a matriz identidade (190 MB a menos por camada grande, mesmo resultado).
- O notebook mostra a RAM do processo em cada linha de resultado.

### Notebook (Kaggle ou Colab), três partes
- Parte 1, quantização do Qwen2.5-0.5B (todas as camadas lineares); parte 2, adaptadores TT contra LoRA, incluindo o LoRA com os mesmos parâmetros e o menor LoRA possível; parte 3, modelo nativo em cinco versões. Cada parte grava o JSON depois de cada modelo, para nada se perder se a sessão cair.

## 0.8.0 — Fase 7: vantagem com memória quântica, prova na QPU e painel ao vivo

### Vantagem quântica medida: aprender com memória quântica (`memoria_quantica.py`)
- Tarefa: dizer se um estado ρ = (I + 0,9 P)/2ⁿ, com P de Pauli desconhecido, é simétrico por reversão temporal, a partir de cópias dele.
- **Com memória quântica**, duas cópias são medidas juntas na base de Bell. **Sem memória**, cada cópia é medida sozinha numa base de Pauli aleatória, com decodificação bayesiana exata sobre todos os Paulis. Os dois lados recebem o mesmo número de cópias.
- Resultado medido, com o mesmo número de cópias: com 6 qubits e 64 cópias, **98,0% com memória contra 49,0% sem** (chute é 50%). Com memória, 64 cópias bastam para 90% de 4 a 6 qubits; sem memória, nem 256 cópias bastam a partir de 4 qubits. No ensaio com o ruído do ibm_kingston (4 qubits, 64 cópias, 20 estados): 100% contra 55%.
- Circuitos Qiskit de preparo e de medida conferidos contra a teoria (erro de 10⁻¹⁶); `python -m quirios_engine.memoria_quantica` (ou `quirios-memoria-quantica`) e `python -m quirios_engine.qpu memoria`.

### Retratação: o "+18 pp" da v0.7 era uma comparação injusta
- Na v0.7, os modelos quânticos de `dados_quanticos.py` recebiam **valores esperados exatos** (como se medissem infinitas vezes), enquanto os clássicos recebiam medidas com 10 ou 100 shots. Com o **mesmo orçamento de medições** para os dois lados, o quântico fica **abaixo** do melhor clássico neste problema (ver README, "Dados com estrutura quântica").
- `comparar(..., justo=True)` passou a ser o padrão. O resultado antigo continua disponível com `justo=False` / `--teto`, rotulado como **teto teórico (comparação injusta)**.
- O motivo físico: a fase do Ising com campo transversal está em correlações **locais** (⟨ZZ⟩), que o laboratório mede diretamente. O próximo benchmark são fases com informação não local (ordem de string, QCNN).

### IA clássica com redes tensoriais (`ia_tensorial.py`, extra `[ia]`)
- `CamadaTT`: substitui `nn.Linear` por uma cadeia de núcleos tensor-train (MPO), o formato do backend MPS. `comprimir_modelo` troca as camadas de um modelo já treinado (TT-SVD) e `CamadaPostoBaixo` é a referência clássica justa (SVD com os mesmos parâmetros).
- MNIST (784-1024-1024-10): com 35× menos parâmetros a rede tensorial chega a 97,46%, acima do modelo denso original (97,34%); com 122× menos, 96,58%. Na mesma acurácia, usa 9 a 22× menos parâmetros que a fatoração de posto baixo. Com o mesmo número de parâmetros da TT de ligação 8, a fatoração clássica fica em 81,5% contra 97,4% da TT.
- A camada TT corta a cadeia ao meio e faz duas multiplicações grandes. Velocidade de uma camada larga na CPU (6 núcleos, ligação 8; menor ganho entre duas medições): 4096×4096 com 1.986× menos parâmetros e 6× / 3,9× / 1,6× mais rápida com lotes de 1 / 64 / 1000; 8192×8192 com 5.191× menos parâmetros e 30× / 6,6× / 3,5×. O treino com camadas TT no MNIST custa o mesmo que o denso (16 s por época).

### Forja quântica (`forja.py`)
- Um circuito quântico desenha a rede: largura e, para cada camada, rede tensorial (núcleos, ligação), posto baixo ou densa. Os projetos ficam em superposição no gerador da evolução quântica (13 qubits, 2.592 projetos); cada medida é um projeto, treinado do zero e medido numa validação separada; o vencedor sai com comprovante selado (SHA-256 dos pesos).
- O mesmo orçamento vai para uma busca genética clássica e uma aleatória. MNIST, meta de 30× menor: gerador quântico 89× menor com 96,80% no teste; genético 54× com 97,30%; aleatório 95× com 97,08% (denso: 98,02%). Os três chegam a projetos equivalentes; nesta escala o motor quântico empata com o clássico.
- O protocolo anterior (ajustar por 200 passos os pesos já comprimidos) favorecia a fatoração de posto baixo e chegava a 94,6% nos três motores; treinar cada projeto do zero corrigiu isso.
- `python -m quirios_engine.forja mnist --alvo 30 [--motor todos] [--modo-gerador ibm]`.
- `python -m quirios_engine.ia_tensorial mnist|fronteira|velocidade`.

### Modelos de linguagem com redes tensoriais (`llm_nativo.py`, `llm_tensorial.py`, notebook do Kaggle)
- **Nativo** (`llm_nativo.py`): um modelo de linguagem caractere a caractere (WikiText-2) treinado do zero em quatro versões com os mesmos dados e passos: denso, rede tensorial TT, posto baixo e denso estreito, as três últimas com o mesmo número de parâmetros. `python -m quirios_engine.llm_nativo --passos 600 [--dispositivo cuda]`.
- Medido na CPU (largura 192, 600 passos, bits por caractere): com 6,4× menos parâmetros, TT 3,94 contra 4,24 do posto baixo com os mesmos parâmetros; com 4×, empate (3,74 contra 3,75). Nos dois casos um modelo denso mais estreito foi melhor (3,71 e 3,65; denso completo 3,44). Nesta escala a TT vence a fatoração clássica, mas não o modelo comum estreito; a versão maior roda na GPU do Kaggle.
- **Comprimido** (`llm_tensorial.py`): troca as camadas de um modelo já treinado (SmolLM2, Qwen2.5...) por TT, posto baixo ou posto baixo guiado pelas ativações (branqueamento do SVD-LLM), e "cura" por destilação com textos, conversas ou os dois (`--dados-cura misto`). Mede perplexidade, tamanho, velocidade e respostas lado a lado.
- Medido na CPU (SmolLM2-135M, 1,5× menos parâmetros na MLP): cortar direto destrói o modelo (perplexidade de 23 para 10⁵–10⁸); com o branqueamento, 163 sem cura e 47 depois de 300 passos, mas as respostas de chat saíram vazias. Para encolher um modelo pronto sem perder qualidade é preciso curar na GPU: `exemplos/kaggle_llm_tensorial.ipynb` faz isso de graça no Kaggle (Qwen2.5-0.5B, 2.000 passos) e também treina uma versão maior do modelo nativo.
- **Cura segura**: o TT-SVD deixava os núcleos do meio com entradas minúsculas e o Adam mudava dezenas de por cento deles por passo, e a cura divergia. `ia_tensorial.equilibrar` dá a todos os núcleos a mesma escala sem mudar a matriz, e `curar` ganhou uma guarda: se a perda dispara, volta ao último ponto bom e reduz a taxa de aprendizado (`reducoes_lr` no resultado).
- **Pronto para GPU**: na GPU, a covariância das ativações ficava na placa e a matriz identidade na CPU, e a parte 1 do notebook cairia ao comprimir. O SVD branqueado agora roda sempre na CPU em float64 (o mesmo caminho testado aqui) e libera a GPU. `carregar` aceita transformers antigos (`torch_dtype`), e o notebook acha o pacote onde quer que o Kaggle monte o dataset.

### Proteção em camadas (`protecao.py`, API, MCP; ver SEGURANCA.md)
- **A memória quântica ajustada ao chip da IQM** (ensaio com o modelo de ruído do Garnet, 40 estados de teste + 20 de calibração, 64 cópias): de 75% contra 70% para **100% contra 45%** (McNemar 22 contra 0, p = 5×10⁻⁷). Duas causas:
  - a calibração do ruído: no Garnet sobra só 0,05 do sinal no lado sem memória, e o decodificador supunha 0,6;
  - o novo `preparo="produto"` (`memoria_quantica._preparar_produto`): o estado (I + εP)/2ⁿ como mistura de estados produto, cada qubit num autoestado do seu Pauli e a paridade sorteada. É exatamente o mesmo estado médio (teste até 10⁻¹²), e não usa nenhuma porta de 2 qubits no preparo: num Pauli de peso 4, o circuito com memória cai de 10 para 4.

  A `prova` completa usa as duas; `qpu memoria` continua com `--preparo cnot` por padrão, para os comprovantes publicados se reproduzirem iguais. No ensaio da IBM Kingston, o preparo novo dá 100% contra 62,5%.
- Notebook do Kaggle: instala o QURIOS direto do GitHub público e roda com "Save & Run All". Não precisa mais subir o pacote como dataset.
- O notebook de modelos de linguagem também roda no **Google Colab** (e em qualquer computador com GPU): descobre sozinho a pasta de saída (`/kaggle/working`, `/content` ou a pasta atual) e imprime o resultado completo no fim de cada parte, para não depender de baixar arquivo. Motivo: em 23/11/2026 duas rodadas com GPU T4 x2 ficaram mais de uma hora em "Queued" no Kaggle, sem começar nem gastar cota, um problema que outros usuários relatam desde 9/11. Botão "Abrir no Colab" no README.
- **Replicado num segundo chip** (19/11/2026, ibm_kingston, estados novos, critério fixado antes de rodar): 95% contra 60% em 40 estados, McNemar 15 contra 1, p = 0,0005, 8 s de QPU. Os dois chips juntos, 3 execuções e 100 estados: 93% contra 69%, McNemar 28 contra 4, p = 0,00002. No kingston, o ensaio com o modelo de ruído previu 97,5% contra 57,5% (sinal calibrado de 0,95 e 0,85), e o chip deu 95% contra 60% com o mesmo 0,95 e 0,85.
- **Vantagem da memória quântica com significância estatística num chip real** (19/11/2026, ibm_marrakesh). Juntando as duas execuções: 91,7% contra 75% em 60 estados, McNemar pareado 13 contra 3, p = 0,021. A segunda execução sozinha deu 97,5% contra 77,5% (40 estados, p = 0,021, 9 s de QPU). Novo: `memoria_na_qpu(..., calibracao=K)` põe K estados de calibração no mesmo job, fora do teste, e mede o sinal que o ruído deixa, separado para cada lado e por máxima verossimilhança numa grade (deu 0,85 com memória e 0,65 sem). O resultado traz o teste exato de McNemar (`qpu.mcnemar`), o certo quando os dois lados respondem sobre os mesmos casos. Sem `calibracao` o comportamento é o de antes. A `prova` completa passou a usar 40 estados de teste + 20 de calibração.
- Conferido e registrado: a calibração **não** explica a melhora da segunda execução. Os mesmos dados, decodificados com o valor fixo de antes (0,6), dão exatamente o mesmo resultado. A diferença entre as execuções vem dos estados sorteados e da variação do chip.
- **Primeira execução numa QPU real** (19/11/2026, ibm_marrakesh, Heron de 156 qubits, 8 jobs, 22 s de QPU, comprovantes assinados em `resultados/qpu_real/`): memória quântica 80% contra 70% sem memória (4 qubits, 64 cópias, 20 estados; vantagem presente, mas sem significância estatística, e menor que no ensaio, 95% contra 55%); dados preparados no chip: empate, 80% contra 80%; evolução sorteada no chip: 92,5% contra 88,7% do padrão, com o campeão sem portas de 2 qubits.
- Corrigido: na qiskit-ibm-runtime 0.50 o tempo de QPU saiu de `job.metrics()` e foi para `job.usage()`, então os comprovantes saíam sem o tempo. Agora o QURIOS lê dos dois lugares. Para a prova de 19/11, o tempo de cada job foi consultado na IBM depois e gravado num complemento assinado (`uso_qpu_ibm_*.json`), sem tocar nos comprovantes originais. O resumo dizia "sem QPU (ensaio)" quando faltava o tempo; agora diz "tempo de QPU não informado".
- `conta_ibm` e `conta_iqm`: no Windows, colar (Ctrl+V) num campo escondido do terminal chegava vazio. Agora abre uma janelinha onde a colagem funciona, ou lê da área de transferência e a limpa depois. Só aceita texto com formato de chave: recusa o comando ou o CRN colados por engano, o que de fato aconteceu na primeira tentativa.
- **Segunda empresa de hardware: IQM** (Finlândia). `BackendQiskit(modo="iqm")` fala com a IQM Resonance pelo `iqm-client`, e `conta_iqm` salva o token sem mostrá-lo (em `~/.quirios/iqm_token`, entregue à IQM só pela variável `IQM_TOKEN` do processo). A IQM não tem o Sampler da IBM: `JobPorRun` manda os circuitos por `backend.run`, em jobs de até 200 circuitos, e devolve o resultado no mesmo formato. O tempo de execução no processador vem da linha do tempo da IQM e entra no comprovante. `qpu prova --modo iqm` (com `--dispositivo garnet:mock` para testar sem gastar créditos) e o ensaio `--modo ruidoso --dispositivo iqm_garnet`, com o ruído do chip Garnet. O pacote da IQM exige Qiskit 2.1 e por isso vive num ambiente separado (extra `iqm`).
- Corrigido: o resumo da `prova` quebrava no experimento `dados` (a acurácia é gravada como [média, desvio]). Achado no ensaio com o ruído da IQM.
- **Prova na QPU real num comando** (`python -m quirios_engine.qpu prova`): escolhe a QPU de menor fila da conta salva, pede uma confirmação só, roda memória quântica, dados preparados na QPU e evolução sorteada na QPU, assina e confere cada comprovante, grava um resumo assinado e a chave pública em `resultados/qpu_real/`. Um experimento que falha não apaga os outros. `--modo ruidoso` faz o mesmo em ensaio. O README ganhou "Situação hoje", dizendo onde cada número foi medido (a memória quântica ainda não rodou numa QPU real).
- **Sem chave o servidor não sobe** (decisão do dono do projeto): o modo aberto deixou de existir. Sem `QURIOS_CHAVES_API` com uma chave valendo, ou sem `QURIOS_API_KEY` de 16 caracteres ou mais, o `uvicorn` para na partida (código 3) com a mensagem de como criar a chave. Segunda trava: se a partida fosse pulada, toda rota protegida responde 401.
- **Identidade**: `CofreDeChaves` com chaves de 256 bits guardadas só como hash SHA-256, escopos `leitura` / `treino` / `admin`, validade e revogação, comparação em tempo constante. Cada rota da API exige o escopo certo.
- **Contenção**: limite de taxa por chave e por IP (`LimitadorTaxa`), bloqueio de força bruta por IP (`BloqueioForcaBruta`), limite de tamanho do corpo (413).
- **Integridade**: `TrilhaAuditoria` encadeada por hash, com a cabeça assinada por ML-DSA-65; `verificar_trilha` detecta registro alterado, apagado, reordenado, truncado ou assinatura falsa. A API registra cada requisição (quem, rota, status, tempo).
- **Sigilo**: `Cofre` com ML-KEM-768 (FIPS 203) + AES-256-GCM. Os modelos gravados pela API ficam cifrados, ligados ao id do modelo e assinados; o texto aberto nunca vai ao disco. `VQCClassifier.para_json` e `carregar(conteudo=...)` para isso.
- **Navegador**: CSP, `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy`, HSTS em HTTPS.
- **Agentes de IA**: `Guarda` em todas as ferramentas do servidor MCP: só as liberadas (`QURIOS_MCP_FERRAMENTAS`), orçamento de chamadas por hora, limites de amostras e features, nomes validados (caminhos e instruções disfarçadas de nome são recusados), tudo auditado.
- `python -m quirios_engine.protecao chave|assinatura|cofre|auditoria` (ou `quirios-protecao`). `/saude` mostra o estado de cada camada.
- Revisão antes do commit (cada item com um teste que falhava antes):
  - uma rota com separador de linha Unicode (`GET /%E2%80%A8`, sem chave nenhuma) fazia a trilha legítima parecer adulterada: registros agora em ASCII e só `\n` separa registros;
  - corpo enviado em pedaços (chunked) passava sem `Content-Length` e furava o limite de tamanho: agora é recusado (411);
  - limitador e bloqueio guardavam IPs para sempre (memória sem limite com IPv6): no máximo 10.000 identidades;
  - `DELETE /modelos/{id}` deixava o arquivo cifrado e as assinaturas no disco: agora apaga tudo;
  - a trilha relia o arquivo inteiro a cada requisição (mais lenta a cada dia): agora lê só o fim, fora do laço de eventos;
  - escopo desconhecido no arquivo de chaves dava erro 500: agora 403;
  - a `Guarda` só conferia o tamanho da primeira amostra: agora confere todas.
- `ARQUITETURA.md`: o que o QURIOS melhora (IA e sistemas), com medidas, os princípios (sem fornecedor único, padrões abertos, funciona sem internet, criptografia pós-quântica desde já, prova em vez de promessa) e as camadas.

### Dados clássicos dentro da QPU (`redes_tensoriais.py`)
- `vetor_para_mps` + `mps_para_circuito`: um vetor clássico vira um circuito de preparo com uma porta pequena por qubit (preparo sequencial de MPS), conferido com fidelidade 1.
- Foto real 128×128 (14 qubits): 240 CNOTs com fidelidade 0,952, contra 16.369 CNOTs do carregamento genérico (68× menos). O genérico quadruplica a cada 2 qubits; o MPS cresce em linha reta.

### Conta da IBM sem expor a chave (`conta_ibm.py`)
- `python -m quirios_engine.conta_ibm` pede a API key sem mostrá-la e salva a conta do Qiskit; `--verificar` lista as QPUs por tamanho da fila.

### Pipeline 100% quântico (`dados_quanticos.py`, `ansatz.py`)
- Nova codificação **`"ising"`** no `Ansatz`: o próprio circuito prepara o estado fundamental a partir de (J, h), com um cronograma tipo QAOA de 6 passos (RZZ + RX) otimizado (`otimizar_cronograma`, `CRONOGRAMA_ISING`). Fidelidade média de 0,93 com 8 spins.
- `fases_ising(preparo="circuito")` (padrão na linha de comando) gera os dados pelo mesmo circuito que roda na QPU. `preparo="exato"` mantém a diagonalização como referência.
- O preparo adiabático ingênuo tinha fidelidade de 0,006–0,45 (erro de Trotter com passo grande). Foi substituído pelo cronograma otimizado.

### Execução na QPU com comprovante (`qpu.py`)
- `python -m quirios_engine.qpu evolucao|dados|verificar --modo ruidoso|ibm`.
  - `evolucao`: a população da evolução quântica é **sorteada na QPU**, um job por geração.
  - `dados`: os estados de Ising são **preparados e medidos na QPU** (2 jobs): bases Z/X para o modelo clássico e o circuito do kernel projetado para o quântico, com o mesmo orçamento de medições.
- **Comprovante** em JSON: ids dos jobs, backend, horários, tempo de QPU, SHA-256 das contagens brutas e circuitos em OpenQASM 3, selado com SHA-256 e, opcionalmente, assinado com ML-DSA-65. `verificar` detecta qualquer alteração.
- No modo `ibm` a linha de comando mostra quantos jobs e shots vai usar e pede confirmação. A chave da IBM nunca passa pelo QURIOS: a conta é salva uma vez com `QiskitRuntimeService.save_account`.
- `BackendQiskit.amostrar_lote` (vários circuitos num job) e `registro_jobs` (cada job enviado, com hash das contagens e `job.metrics()`).

### Painel ao vivo (`painel.py`, API)
- `GET /painel`: página que acompanha uma evolução quântica geração a geração — entropia da população caindo, acurácia contra o circuito padrão, ligações entre genes nascendo. SVG sem bibliotecas externas (funciona offline).
- `POST /painel/sessoes`, `GET /painel/sessoes/{id}/eventos` (text/event-stream) e `GET /painel/sessoes/{id}`. A chave vai no cabeçalho `X-API-Key`, nunca na URL. No máximo 3 sessões ativas (429 acima disso).
- `EvolucaoQuantica(ao_progresso=...)`: callback por geração.

### Medição justa (`backends.py`)
- **`BackendShots(shots)`**: envolve qualquer backend e aplica o ruído de amostragem de um número finito de shots (binomial nas marginais e nos valores esperados). É o que torna a comparação de dados quânticos justa.

### Testes de ponta a ponta e de estresse (`tests/e2e/`)
- `pytest -m e2e`: servidor HTTP real em subprocesso (autenticação, treino, orquestração, tarefas assíncronas com MPS, evolução quântica, campeões, dados quânticos, projeção, painel por streaming, persistência assinada após reinício), todas as linhas de comando, comprovante de QPU assinado e adulterado, e o servidor MCP pelo protocolo stdio real.
- `pytest -m stress -s`: requisições concorrentes, 100 tarefas assíncronas, evoluções concorrentes na mesma chave, cargas grandes, limite de sessões do painel, 240 entradas malformadas, MPS com 100 qubits e registro concorrente entre processos.
- A suíte padrão (`pytest`) continua rápida: e2e e estresse só rodam quando pedidos.

### Corrigido (encontrado pelos testes de ponta a ponta e de estresse)
- **O servidor caía (access violation) com trabalho pesado em paralelo.** O OpenBLAS que vem com o numpy 2.x no Windows quebra quando várias threads entram ao mesmo tempo nas rotinas multithread dele. Reproduzido fora do QURIOS, só com numpy: `eigh` 256×256 em 4 threads, e `vdot` de vetores complexos de 2^16 (o gradiente adjunto da evolução quântica). Rotinas como `eigvalsh`, `svd`, `solve` e `qr` não quebram. A API e o pool de tarefas agora usam o BLAS em 1 thread (`tarefas.blas_em_uma_thread`, via threadpoolctl), já que paralelizam por requisição, e o `eigh` de `dados_quanticos` é serializado por uma trava.
- **Entradas com NaN ou infinito** (o JSON do Python as aceita) faziam o erro 422 falhar ao repetir a entrada, e a conexão caía sem resposta. Pior: NaN passava pela validação de X na codificação por ângulo, porque `NaN < 0` é falso. Agora toda entrada da API recusa NaN e infinito, e o erro 422 é sempre um JSON válido.
- **Escritas concorrentes no registro de campeões perdiam atualizações**: duas evoluções na mesma chave liam o arquivo, e a segunda gravação apagava a primeira. Agora há uma trava (entre threads e entre processos, por arquivo `.lock`) e o registro é relido sob a trava antes de gravar.
- **No Windows, a gravação do registro falhava com "Acesso negado"** quando outro processo (um leitor do registro, antivírus ou indexador) estava com o arquivo aberto no instante da troca atômica. A troca e a leitura agora tentam de novo por alguns milissegundos.
- **Um modelo persistido adulterado impedia o servidor de iniciar**. Agora ele é recusado, o servidor sobe com os demais, e `/saude` lista os rejeitados em `modelos_rejeitados`.

## 0.7.0 — Fase 6: o circuito quântico conduz a própria evolução

### Evolução quântica (`evolucao_quantica.py`)
- **`EvolucaoQuantica`**: a população de arquiteturas vive em **superposição** num circuito gerador, com 16 qubits no espaço padrão para as 15.552 arquiteturas possíveis.
  - **Indivíduo:** medir o gerador sorteia um genoma (regra de Born).
  - **Mutação:** rotações aleatórias nos ângulos do gerador.
  - **Herança:** o gerador é ajustado para aumentar a amplitude da elite.
  - **Ligação entre genes:** portas RZZ emaranham os qubits do gerador, e o circuito aprende quais genes andam juntos.
- **Gradiente exato por diferenciação adjunta**: 27× mais rápido que o parameter-shift, com igualdade testada até 1e-10.
- **Temperatura quântica**: um termo de entropia evita o colapso prematuro da população.
- **Ligações aprendidas**: informação mútua entre genes na população sorteada. Começa em 0 e só cresce com emaranhamento (um gerador sem RZZ produz genes independentes).
- As RZZ começam com ângulos aleatórios. Com todas em zero, o gradiente delas é identicamente nulo (ponto de sela), e a distribuição inicial continua uniforme do mesmo jeito.
- O gerador pode ser medido em qualquer backend (`backend_gerador=`), inclusive Aer com ruído e a QPU da IBM, via o novo `BackendQiskit.amostrar(circuito, shots)`.
- Usa o mesmo registro de campeões e a mesma promoção segura da `BuscaEvolutiva` (agora `_promover`).
- `evoluir_continuamente(..., motor="quantico")` e `--motor quantico|classico` na linha de comando. **O padrão passou a ser o motor quântico.**

### Dados com estrutura quântica (`dados_quanticos.py`)
- `fases_ising()` gera sistemas de Ising com campo transversal desordenado, com duas visões:
  - o **estado fundamental**, que os modelos quânticos recebem por amplitude;
  - **medidas** com poucos shots, que os modelos clássicos recebem.
- `comparar()`: clássicos (logística, RFF, SVM) contra quânticos (VQC e kernel projetado), com teste pareado.
- Resultado perto da transição de fase, com 10 shots: **quântico 96,8% contra clássico 78,5% (+18 pp, p = 0,03)**. Com 100 shots, a diferença deixa de ser significativa (o gargalo de medição). **Retratado na 0.8.0:** o quântico recebia valores esperados exatos; com o mesmo número de medições ele fica abaixo do clássico.
- Linha de comando `python -m quirios_engine.dados_quanticos` (ou `quirios-dados-quanticos`).
- `VQCClassifier(constante_amplitude=None)` para dados que já são estados quânticos.

### Caminho IBM profundo (`ibm_estimador.py`, extra `[ibm]`)
- **`BackendEstimador`** usa o primitivo Estimator:
  - o novo Estimator da IBM, com processamento no cliente, quando disponível; senão o `EstimatorV2`;
  - a pilha de mitigação da IBM: resiliência 0/1/2 (TREX, ZNE), desacoplamento dinâmico XpXm e twirling.
- **Operator backpropagation** (`qiskit-addon-obp`): absorve classicamente as últimas fatias do circuito no observável. A QPU recebe um circuito mais raso, dentro de um orçamento de erro.
- O modo `"local"`, com o Estimator do Aer, testa o caminho inteiro sem conta, com ou sem o ruído de uma geração de hardware.
- `criar_backend("estimador-local" | "ibm-estimador")`.

### Evolução contínua agendada (`agendamento.py`)
- `python -m quirios_engine.agendamento mostrar|instalar|remover` (ou `quirios-agendar`) agenda a auto-evolução noturna:
  - no Windows, pelo Agendador de Tarefas;
  - no Linux/macOS, pelo crontab.
- O log fica em `evolucao_noturna.log`.

### Corrigido
- **O primeiro campeão era promovido sem evidência** ("primeiro campeão deste problema") e podia ser pior que o circuito padrão. Agora, enquanto não há campeão, o titular a ser batido é o circuito padrão do QURIOS, com a mesma regra de evidência. Num teste com Iris, o campeão promovido do jeito antigo acertava 86,7% contra 93,3% do padrão.
- **`entropia_emaranhamento` media o qubit errado** com 3 ou mais qubits: ela usava a convenção MSB, enquanto o resto do QURIOS usa LSB. No estado (|000⟩+|011⟩)/√2 ela dizia que o qubit 0 não estava emaranhado.
- Ela também montava a matriz de densidade completa (4^n), o que com 16 qubits pediria cerca de 68 GB. Agora calcula a matriz reduzida direto do vetor de estado (`matriz_reduzida`), em milissegundos.

### API e MCP
- `POST /evoluir` ganha `motor` (padrão `"quantico"`) e devolve entropia e ligações aprendidas.
- Novo `POST /dados-quanticos`.
- Novas ferramentas MCP: `evoluir_circuito(motor=...)` e `comparar_dados_quanticos`.

## 0.6.0 — Fase 5: núcleo quântico auto-evolutivo e tecnologia emergente

### Circuito genético (`ansatz.py`)
- O `Ansatz` virou um genoma com novos genes:
  - `rotacoes`: qualquer sequência de Rx/Ry/Rz, inclusive SU(2) completo;
  - `emaranhador`: CNOT, CZ ou **RZZ treinável** (porta fracionária nativa dos IBM Heron);
  - `codificacao`: ângulo, **IQP / mapa ZZ** (Havlíček et al. 2019) ou amplitude;
  - `leitura`: Z, Z+ZZ, **Pauli** (X, Y, Z por qubit) ou **Pauli2** (todos os Paulis de 2 qubits vizinhos).
- O genoma padrão reproduz exatamente o circuito da v0.2–v0.5.
- `esperanca_pauli()` calcula valores esperados exatos de qualquer Pauli.
- `portas_2q()` dá o custo do circuito em hardware.

### Auto-evolução (`evolucao.py`)
- **`BuscaEvolutiva`** faz busca de arquitetura (QAS) por torneio, cruzamento, mutação e elitismo.
  - A aptidão é a acurácia de validação cruzada menos uma penalidade por portas de 2 qubits.
  - Há dois modos: mapa de features (reservatório) ou kernel projetado.
- **`RegistroCampeoes`** guarda o campeão de cada problema e o histórico. A escrita é atômica, com assinatura ML-DSA opcional.
- **Promoção segura**: um genoma novo só vira campeão se vencer o atual nas mesmas dobras, de um destes jeitos:
  - com p < 0,05 no teste t de Nadeau-Bengio;
  - ou custando 25% menos portas sem perder acurácia.
  - O sistema nunca regride.
- `evoluir_continuamente()` e a linha de comando `python -m quirios_engine.evolucao` (ou `quirios-evolucao`) rodam o ciclo de auto-melhoria rodada após rodada.
- O orquestrador inclui o campeão automaticamente (`candidatos_padrao(..., registro=, chave=)`).

### Kernels quânticos (`kernels.py`)
- `KernelQuantico` oferece dois tipos:
  - **fidelidade**;
  - **projetado** (Huang et al. 2021), que usa só ⟨X⟩, ⟨Y⟩ e ⟨Z⟩ de cada qubit e por isso roda em qualquer backend, inclusive MPS com 100 qubits.
- `ClassificadorKernel`: kernel ridge que **se calibra sozinho**, escolhendo λ e γ pelo erro leave-one-out em fórmula fechada (sem nenhum circuito a mais).
- `alinhamento` (kernel-alvo centralizado), `concentracao` (diagnóstico de concentração exponencial, Thanasilp et al. 2024) e `treinar_alinhamento` (kernel treinável via SPSA).

### Medição e mitigação de erro (`backends.py`)
- `BackendQiskit` mede qualquer leitura de dois jeitos:
  - **bases X/Y/Z** agrupadas;
  - **sombras clássicas** (Huang, Kueng & Preskill 2020), com bases aleatórias por qubit.
- **Mitigação de leitura**: calibra a matriz de confusão por qubit e inverte as correlações de até 2 qubits.
- **ZNE (extrapolação para ruído zero)**: dobra as portas de 2 qubits (G → G G⁻¹ G) nas escalas 1, 3 e 5 e extrapola até 0.
- `portas_fracionarias=True` pede RZZ/RX nativos à IBM.
- **Plugins de provedores**: `criar_backend()` encontra backends publicados no grupo de entry points `quirios.backends`; `backends_registrados()` lista os disponíveis.

### Hardware do futuro (`futuro.py`)
- `BackendQiskit(modo="parametrico", erros=...)` simula qualquer taxa de erro. As gerações prontas são `hoje` (≈ Heron r2), `proxima`, `dez_vezes`, `cem_vezes` e `tolerante_a_falhas`.
- `projetar_hardware()` gera a **curva de prontidão**: erro das features e acurácia em cada geração.

### Agentes de IA e segurança
- **Servidor MCP** (`mcp_servidor.py`, extra `[mcp]`, SDK MCP 2.x): agentes de IA usam o QURIOS como ferramenta. As 8 ferramentas são: treinar, prever, orquestrar, evoluir, campeões, projetar hardware, diagnosticar kernel e status.
- **Assinatura pós-quântica** (`seguranca.py`, extra `[pqc]`): ML-DSA-65 (NIST FIPS 204) assina modelos salvos e o registro de campeões. O que for adulterado ou vier de outra chave não carrega.

### API
- `POST /evoluir` (tarefa assíncrona), `GET /campeoes` e `POST /projetar`.
- Os genes novos em `POST /modelos`; `incluir_kernel` e `chave_campeao` em `POST /orquestrar`.
- Variáveis de ambiente `QURIOS_REGISTRO`, `QURIOS_CHAVE_PQC` e `QURIOS_CHAVE_PQC_PUBLICA`.

### Alterado
- `BackendQiskit.recuperar(job_id, ansatz)` aceita o ansatz do envio; `n_qubits` continua funcionando.
- Comandos instaláveis: `quirios-mcp`, `quirios-evolucao` e `quirios-benchmark`.

## 0.5.0 — Fases 1 a 4 da infraestrutura

### Fase 1: fundação da infra
- **`Ansatz`** (`ansatz.py`): descrição única do circuito que todo backend
  executa, com vetor de parâmetros plano. Com 1 camada reproduz exatamente o
  circuito da v0.2–v0.4 (fidelidade 1, testado).
- **Contrato `Backend`**: `features(X, ansatz, params)` e `submeter(...)`.
  Backend novo = herdar e implementar `_calcular`. `criar_backend(nome)`.
- **`BackendMPS`**: tensor network (matrix product state) do Qiskit Aer, com
  valores esperados exatos e sem shots. Topologia linear ou anel de 1 camada
  roda 100 qubits em ~0,1 s/amostra; `max_bond` para truncar.
- **GPU opcional**: `device="GPU"` nos backends Aer, validado na criação;
  `dispositivos_disponiveis()`.
- **Tarefas assíncronas** (`tarefas.py`): `backend.submeter()` devolve uma
  `Tarefa` (status, resultado com timeout, cancelar). Jobs IBM recuperáveis
  em outro processo com `BackendQiskit.recuperar(job_id, n_qubits)`.
- `QuantumConfig` não monta mais a permutação densa (2^n) na criação, então
  configs de 50–100+ qubits funcionam com MPS/Qiskit.
- `VQCClassifier.salvar()` / `VQCClassifier.carregar()` em JSON (sem pickle).

### Fase 2: treino que escala
- **`SPSA`** (`otimizadores.py`): 2 circuitos por passo, qualquer número de
  parâmetros; ganhos de Spall com calibração automática do passo.
- **Parameter-shift geral**: funciona com qualquer ansatz e qualquer backend
  (antes só existia o caminho rápido de 1 camada).
- `VQCClassifier(backend=..., otimizador=..., n_camadas=..., codificacao=...,
  inicializacao=..., treino_por_camadas=..., lote=...)`;
  `circuitos_executados` conta o custo equivalente em hardware.
- **Treino por camadas** e **inicialização "pequena"** (mitigação de barren plateau).
- **Diagnóstico** (`diagnostico.py`): `variancia_gradiente()` (termômetro de
  barren plateau) e `estimar_circuitos()` (orçamento de QPU antes de treinar).

### Fase 3: mais dados por qubit
- **Data re-uploading**: d features em n qubits, distribuídas em ceil(d/n)
  camadas (ou reenviadas, se d = n).
- **Codificação por amplitude**: d features em ceil(log2 d) qubits.
- `Quantico` aceita mais features que qubits (re-uploading automático).

### Fase 4: orquestrador híbrido
- **`Orquestrador`** (`orquestrador.py`): avalia candidatos clássicos e
  quânticos nas mesmas dobras de uma validação cruzada repetida. O quântico
  só é escolhido se vencer o melhor clássico com p < alfa (teste t corrigido
  de Nadeau-Bengio). Orçamento de circuitos, relatório JSON, `submeter()`.
- **Roteador** `escolher_backend(config)`: vetor exato, MPS exato ou MPS
  truncado conforme o tamanho e o entrelaçamento.
- **Benchmark contínuo** (`benchmark.py`): `python -m quirios_engine.benchmark
  rodar|resumo`, com histórico JSONL e comparação pareada quântico vs clássico.
- `estatistica.py`: distribuição t e teste pareado sem scipy.

### API
- Campos novos em `POST /modelos`: `backend` (exato/mps/auto), `otimizador`,
  `codificacao`, `camadas_dados`, `inicializacao`, `treino_por_camadas`,
  `assincrono`.
- `POST /orquestrar`, `GET /tarefas`, `GET /tarefas/{id}`, `GET /backends`.
- `QURIOS_API_KEY` (cabeçalho `X-API-Key`) e `QURIOS_DIR_MODELOS` (modelos
  sobrevivem a reinícios).

### Corrigido
- **Modo `"ruidoso"` do `BackendQiskit` rodava sem ruído.** O modelo de ruído
  do ibm_kingston só era usado na transpilação; o `SamplerV2` do Aer era
  criado sem ele. Num circuito de teste, P(GHZ) dava 1,000 no lugar de 0,755.
  A correlação de 0,998 "no modelo de ruído" citada na v0.3 foi medida sem
  ruído.

### Alterado (incompatível)
- Backends: a assinatura passou de `features(X, theta, phi, alpha, config)`
  para `features(X, ansatz, params)`. `circuito_qiskit(x, theta, phi, alpha,
  config)` continua disponível por compatibilidade.

## 0.4.0

### Novo
- **API HTTP** (`quirios_engine.api`, extra `[api]`): treinar, listar, prever,
  medir entropia e remover modelos via FastAPI. `uvicorn quirios_engine.api:app`.
- `VQCClassifier.extrair_features(X)`: marginais de cada qubit + entropia de
  medição + pureza, para alimentar um classificador clássico (estava no README
  da v0.3 mas não existia).
- `VQCClassifier.estado_quantico(x)`: vetor de estado do circuito treinado,
  incluindo a camada pós-entrelaçamento (`alpha`).
- Suíte de testes em pytest (`tests/`), separada por módulo, com testes de
  regressão para os bugs abaixo e para a API.
- CI no GitHub Actions (Linux e Windows, Python 3.10 a 3.12) e build do pacote.
- `exemplos/` agora entra no sdist (`MANIFEST.in`).

### Corrigido
- **Topologia anel com `n_camadas` múltiplo de `n_qubits`** gerava pares
  `(i, i)`: um CNOT de um qubit nele mesmo, que não é unitário (permutação
  não bijetora). Exemplo: `QuantumConfig(2, "anel", 2)`. O offset agora cicla
  em `1..n-1`; para `n_camadas < n_qubits` os circuitos são idênticos aos da v0.3.
- Treinar/prever sem `fit()` agora dá `RuntimeError` com mensagem clara em vez
  de `AttributeError`.
- `exemplos/rodar_na_ibm.py` grava o JSON em UTF-8 (no Windows o padrão era cp1252).
- `__all__` passa a exportar `ClassificadorQURIOS`, `Brutos`, `RFF`,
  `Quantico`, os backends, `loss_e_grad`, `ajustar_leitura`, `logistica_l2`.

### Alterado
- Python mínimo: 3.10 (era 3.9 no `pyproject.toml`, 3.10 no README).
- Removida a dependência `scipy`, que não era usada.
- O script `tests/test_pacote_completo.py` foi substituído pela suíte pytest.

## 0.3.0
- Backends Qiskit (ideal, ruidoso com modelo do ibm_kingston, IBM real).
- `ClassificadorQURIOS` com geradores plugáveis e seleção por validação cruzada.
- `exemplos/rodar_na_ibm.py`.

## 0.2.0
- Leitura de todos os qubits, camada `Ry(alpha)` pós-entrelaçamento, leitura
  resolvida por regressão logística L2, topologia `blocos`, simulação rápida.
