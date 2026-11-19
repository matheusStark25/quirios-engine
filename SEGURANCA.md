# Segurança do QURIOS

Defesa em profundidade, como em bancos: várias camadas independentes, de modo
que a falha de uma não entrega o sistema, e tudo registrado de um jeito que
não dá para apagar sem ser notado. **Nenhum sistema é impossível de invadir**;
o objetivo é que uma invasão seja cara, contida e visível.

## Ameaças consideradas

| Ameaça | Camada que responde |
|---|---|
| Servidor colocado no ar sem nenhuma chave (por pressa ou esquecimento) | **Sem chave o QURIOS não sobe**: sem chave configurada, com chave única de menos de 16 caracteres ou com o arquivo de chaves sem nenhuma valendo, o servidor para na partida. Não existe modo aberto; se a partida fosse pulada, toda rota protegida responderia 401 |
| Alguém sem chave (ou um robô) tenta usar a API | Identidade: toda rota, menos `/saude` e a página do painel, exige `X-API-Key` |
| Adivinhar chaves por força bruta | Chaves de 256 bits; bloqueio do IP depois de 10 erros em 5 minutos (15 min) |
| Descobrir a chave pelo tempo de resposta | Comparação em tempo constante (`hmac.compare_digest`), sobre o hash |
| Vazamento do arquivo de chaves | Só o hash SHA-256 é guardado; a chave aparece uma vez, ao ser criada |
| Chave roubada usada para tudo | Escopos: `leitura` (consultar, prever), `treino` (treinar, evoluir), `admin` (apagar); validade e revogação |
| Inundar o servidor | Limite de requisições por chave e por IP; limite de tamanho do corpo; tarefas pesadas em fila |
| Mandar o corpo em pedaços (chunked) para furar o limite de tamanho | Pedido com corpo sem `Content-Length` é recusado (411) antes de ser lido |
| Trocar de IP sem parar (IPv6) para esgotar a memória | Limitador e bloqueio guardam no máximo 10.000 identidades; as inativas saem primeiro |
| Envenenar a trilha com caracteres que quebram linha (`/%E2%80%A8`) | Registros gravados em ASCII puro; só `\n` separa registros na verificação |
| Entradas malformadas, NaN, payloads gigantes | Validação estrita (NaN e infinito recusados), 240 entradas malformadas no teste de estresse sem nenhuma queda |
| Apagar os rastros de um ataque | Trilha de auditoria encadeada por hash, cabeça assinada com ML-DSA-65: alterar, apagar, reordenar ou truncar é detectado |
| Roubo dos modelos gravados | Cifra AES-256-GCM com chave embrulhada por ML-KEM-768; o texto aberto nunca vai ao disco |
| Trocar um modelo por outro | Cifra ligada ao id do modelo e assinatura ML-DSA do arquivo cifrado; o servidor recusa e lista em `/saude` |
| Computador quântico futuro quebrando a criptografia | ML-DSA (FIPS 204) e ML-KEM (FIPS 203), padrões pós-quânticos do NIST |
| **Uma IA usando o QURIOS para atacá-lo** (MCP) | Guarda: lista de ferramentas liberadas, orçamento de chamadas por hora, limites de tamanho, validação de nomes, auditoria. Nenhuma ferramenta executa código, abre caminhos de arquivo ou muda configuração a partir do que o agente envia: tudo é tratado como dado |
| Página do painel embutida em site malicioso | Cabeçalhos `Content-Security-Policy`, `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy`; HSTS quando servido por HTTPS |
| Chave da IBM exposta | Digitada só no terminal do dono (`conta_ibm`), nunca impressa nem guardada pelo QURIOS |

## Como ligar tudo

```bash
python -m quirios_engine.protecao chave criar --arquivo segredos/chaves.json --escopos leitura treino --dias 90
python -m quirios_engine.protecao assinatura gerar --pasta segredos      # par ML-DSA-65 (assinar)
python -m quirios_engine.protecao cofre gerar --pasta segredos/cofre     # par ML-KEM-768 (cifrar)

export QURIOS_CHAVES_API=segredos/chaves.json
export QURIOS_AUDITORIA=auditoria/trilha.jsonl
export QURIOS_CHAVE_PQC=segredos/quirios_mldsa65.pem
export QURIOS_CHAVE_PQC_PUBLICA=segredos/quirios_mldsa65.pub.pem
export QURIOS_COFRE_PUBLICA=segredos/cofre/quirios_mlkem768.pub.pem
export QURIOS_COFRE_PRIVADA=segredos/cofre/quirios_mlkem768.pem
export QURIOS_DIR_MODELOS=modelos/
uvicorn quirios_engine.api:app --host 127.0.0.1 --port 8000

curl http://127.0.0.1:8000/saude          # mostra o estado de cada camada
python -m quirios_engine.protecao auditoria verificar auditoria/trilha.jsonl --chave-publica segredos/quirios_mldsa65.pub.pem
```

Agentes de IA (MCP): `QURIOS_MCP_FERRAMENTAS` (lista do que é liberado),
`QURIOS_MCP_CHAMADAS_HORA`, `QURIOS_AUDITORIA`.

## O que é testado

`tests/test_protecao.py` e `tests/test_api_protecao.py` atacam cada camada, e
`tests/e2e/test_e2e.py` tenta subir o `uvicorn` de verdade sem chave (ele sai
com erro e a mensagem de como criar uma). Os ataques:
chave falsa e força bruta (bloqueio na 5ª tentativa no teste), escopos,
chave revogada e vencida, trilha alterada / apagada / reordenada / truncada /
com assinatura falsa, cifra com contexto trocado, chave errada e bytes
alterados, modelo adulterado no disco, corpo gigante (inteiro ou em pedaços),
30.000 IPs trocados, rota com separador de linha Unicode, escopo inventado no
arquivo de chaves, modelo apagado sem deixar cifrado nem assinatura, cabeçalhos, e agentes
pedindo ferramentas não liberadas, mandando dados gigantes ou "nomes" que são
caminhos de arquivo e instruções.

## O que fica para a implantação

O código não garante sozinho:

- HTTPS (TLS) na frente da API, com um proxy reverso. Atrás do proxy, rode o
  uvicorn com `--proxy-headers --forwarded-allow-ips <IP do proxy>`; sem isso
  todo cliente parece ter o IP do proxy, e o bloqueio de um atacante bloquearia
  todo mundo.
- Rede isolada e firewall; a API não deve ficar exposta direto na internet.
- Chaves num gerenciador de segredos ou HSM, não em pastas comuns, com rotação periódica.
- Backups cifrados e testados.
- Atualização de dependências e varredura de vulnerabilidades.
- Testes de invasão feitos por uma empresa externa antes de qualquer cliente real.
