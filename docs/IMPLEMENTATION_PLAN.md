# Plano Revisado de Implementação - Artigo ENIAC 2026

## 1. Diagnóstico de Viabilidade

### Premissas

- Hoje no ambiente: 2026-06-07, America/Sao_Paulo.
- Deadline informado: 2026-06-08 23:59 UTC-12, ou 2026-06-09 08:59 no horário de Sao Paulo.
- O prazo oficial deve ser confirmado manualmente no JEMS3 e/ou site oficial do evento, pois a busca publica nao retornou facilmente uma pagina oficial indexada com esses detalhes.

### Veredito

Se ainda nao existem codigo funcional, resultados e rascunho do artigo, o plano original completo nao e viavel com qualidade:

- 2 ambientes nao e viavel.
- 10 seeds por configuracao provavelmente nao e viavel.
- Ablacao L={1,3,5} completa provavelmente nao e viavel.
- Benchmark CPU vs GPU do simulador so entra se nao atrasar resultados principais.

O que ainda e viavel como submissao honesta:

- 1 ambiente: CartPole-v1.
- 2 agentes: DQN-MLP vs DQN-VQC.
- 3 a 5 seeds por agente, idealmente 5.
- 1 configuracao principal de VQC, preferencialmente angle encoding + data re-uploading + CNOT em anel + readout linear.
- Ablacao minima apenas se sobrar tempo: L=1 vs L=3 em 3 seeds, ou uma discussao qualitativa sem vender como resultado principal.
- Artigo em ate 12 paginas com foco Undergraduate: estudo controlado, limitacoes claras, sem alegar vantagem quantica.

## 2. Escopo Final Recomendado

### Pergunta de Pesquisa Principal

Com contagem de parametros aproximadamente equiparada, como um agente DQN-VQC se compara a um DQN classico em eficiencia amostral, estabilidade e tempo de treinamento no CartPole-v1?

### Contribuicoes

1. Implementacao de um agente DQN hibrido no qual o aproximador de Q-values usa um VQC treinavel.
2. Comparacao controlada com baseline MLP-DQN sob orcamento de parametros equiparado.
3. Analise empirica de recompensa, estabilidade entre seeds, parametros treinaveis e tempo de parede.
4. Discussao de limitacoes praticas de simulacao, profundidade de circuito e escalabilidade.

### Nao Prometer

- Vantagem quantica.
- Superioridade geral do VQC.
- Escalabilidade para hardware quantico real.
- Conclusoes universais sobre QRL.

## 3. Arquitetura Experimental

### Ambiente

Obrigatorio:

- `CartPole-v1`
- Estado: 4 dimensoes
- Acoes: 2
- Qubits recomendados: 4

Cortar:

- LunarLander, MiniGrid, Acrobot, FrozenLake, salvo se ja houver resultados prontos.

### Agentes

#### DQN-MLP

- Entrada: estado normalizado.
- Saida: Q-values para cada acao.
- Arquitetura ajustada para ficar dentro de +/-10% da contagem de parametros do VQC.
- Mesmo replay buffer, target network, epsilon schedule, batch size e otimizador do VQC sempre que possivel.

#### DQN-VQC

- Entrada: estado normalizado para `[-pi, pi]`.
- Encoding: angle encoding com `RY` ou `RX/RY`.
- Data re-uploading: repetir encoding entre camadas, se a implementacao ja estiver estavel.
- Ansatz: rotacoes treinaveis por qubit + CNOT em anel.
- Profundidade principal: `L=3`, salvo instabilidade; se instavel, usar `L=1`.
- Readout: expectativas de Pauli-Z por qubit + camada linear classica para numero de acoes.

## 4. Matriz Experimental Minima

Tabela obrigatoria:

| Experimento | Ambiente | Agente | Seeds | Observacao |
|---|---|---:|---:|---|
| Principal MLP | CartPole-v1 | DQN-MLP | 5 | Baseline parametrizado |
| Principal VQC | CartPole-v1 | DQN-VQC L=3 ou L=1 | 5 | Configuracao final |

Tabela opcional, se houver tempo:

| Experimento | Ambiente | Agente | Seeds | Observacao |
|---|---|---:|---:|---|
| Ablacao rasa | CartPole-v1 | DQN-VQC L=1 | 3 | Comparar com L=3 |
| Ablacao media | CartPole-v1 | DQN-VQC L=3 | 3 | Apenas se ja nao for o principal |

## 5. Metricas

Obrigatorias:

- Recompensa por episodio.
- Media movel de 100 episodios, ou janela menor se o numero de episodios for baixo.
- Media e desvio padrao entre seeds.
- Numero de parametros treinaveis.
- Tempo de parede por seed/configuracao.

Desejaveis:

- Passos ou episodios ate atingir limiar de resolucao.
- Area sob curva de aprendizado.
- Video/GIF de um episodio treinado para apresentacao, nao como resultado central do PDF.

## 6. Cronograma de Emergencia

### 2026-06-07 - Agora ate fim do dia

1. Confirmar deadline no JEMS3/site oficial.
2. Montar ambiente Python.
3. Implementar ou validar DQN-MLP em CartPole.
4. Implementar DQN-VQC minimo.
5. Rodar smoke tests:
   - 1 seed MLP por poucos episodios.
   - 1 seed VQC por poucos episodios.
   - salvar CSV de recompensas.
6. Criar esqueleto do artigo no template SBC.
7. Escrever Introducao, Metodologia e Configuracao Experimental com placeholders honestos para resultados.

### 2026-06-08 - Dia de resultados e escrita final

1. Rodar matriz minima com 3 a 5 seeds.
2. Gerar figuras:
   - curva MLP vs VQC;
   - tabela de parametros;
   - tabela de hiperparametros;
   - figura do circuito VQC.
3. Escrever Resultados e Discussao com interpretacao conservadora.
4. Escrever Limitacoes e Conclusao.
5. Revisar anonimato:
   - remover autores;
   - remover instituicao;
   - remover agradecimentos;
   - evitar caminhos locais identificaveis em figuras/metadados.
6. Compilar PDF e checar limite de 12 paginas.
7. Submeter antes do limite operacional interno: 2026-06-08 22:00 BRT, se possivel.

## 7. Criterios de Corte

Se o VQC nao aprende claramente:

- ainda submeter se houver comparacao honesta mostrando dificuldade, desde que a metodologia esteja correta;
- reposicionar o artigo como estudo empirico de estabilidade e custo, nao como demonstracao de desempenho;
- enfatizar que o VQC nao superou o baseline no regime estudado.

Se nao der para rodar 5 seeds:

- rodar 3 seeds e declarar explicitamente como limitacao;
- nao fazer teste estatistico forte;
- reportar intervalos e discutir variabilidade.

Se o artigo passar de 12 paginas:

1. Cortar segundo ambiente.
2. Cortar ablaçao.
3. Encurtar trabalhos relacionados.
4. Mover detalhes de hiperparametros para tabela compacta.

## 8. Estrutura do Artigo

1. Resumo e Abstract.
2. Introducao:
   - contexto de RL, DQN e QML;
   - motivacao de VQC como aproximador;
   - pergunta de pesquisa;
   - contribuicoes.
3. Fundamentacao e Trabalhos Relacionados:
   - DQN;
   - VQC/PQC;
   - QRL em Gym;
   - lacuna: comparacao parametrizada e ablaçoes controladas.
4. Metodologia:
   - DQN comum;
   - MLP baseline;
   - DQN-VQC;
   - normalizacao, encoding, ansatz, readout;
   - equiparacao de parametros.
5. Configuracao Experimental:
   - ambiente;
   - hiperparametros;
   - seeds;
   - hardware e simulador;
   - metricas.
6. Resultados e Discussao:
   - curva de aprendizado;
   - eficiencia amostral;
   - estabilidade;
   - tempo de parede;
   - interpretacao.
7. Limitacoes:
   - simulacao classica;
   - poucos qubits;
   - possivel barren plateau;
   - numero limitado de seeds;
   - ausencia de hardware real.
8. Conclusao e Trabalhos Futuros.

## 9. Checklist Final

- [ ] Deadline oficial confirmado.
- [ ] PDF no template SBC.
- [ ] PDF com ate 12 paginas.
- [ ] PDF duplo-cego.
- [ ] Titulo e resumo em ingles, se exigido.
- [ ] Resumo em portugues e abstract em ingles, se o corpo estiver em portugues.
- [ ] Tabela de hiperparametros.
- [ ] Tabela de parametros treinaveis.
- [ ] Curvas com media e desvio.
- [ ] Seeds documentados.
- [ ] Hardware e backend documentados.
- [ ] Nenhuma alegacao de vantagem quantica.
- [ ] Referencias conferidas em fontes oficiais.
- [ ] Submissao feita no JEMS3.
