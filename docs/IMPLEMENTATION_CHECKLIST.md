# Checklist de Implementacao

Este checklist existe para evitar progresso baseado em suposicoes. Cada etapa deve produzir um artefato verificavel antes da proxima.

## Fatos Verificados

- Repositorio local: `/home/daniel/artigo-reinforcement-learning-quantico`.
- O diretorio ainda nao e um repositorio Git.
- `python` nao esta disponivel no shell.
- `python3` esta disponivel: Python 3.12.3.
- No Python global, foram encontrados:
  - `PyYAML`;
  - nao encontrados: `torch`, `gymnasium`, `pennylane`, `numpy`, `pandas`, `matplotlib`.

## Etapa 0 - Planejamento Operacional

- [x] Criar plano revisado em `docs/IMPLEMENTATION_PLAN.md`.
- [x] Criar estrutura de pastas em `docs/FOLDER_STRUCTURE.md`.
- [x] Criar contexto para agentes: `agent.md`, `claude.md`, `codex.md`.
- [x] Registrar este checklist.

Gate de saida:

- [x] Existe checklist versionavel e verificavel.

## Etapa 1 - Projeto Python Minimo

- [x] Criar `requirements.txt`.
- [x] Criar `pyproject.toml` minimo.
- [x] Criar pacote `src/qrl` com `__init__.py`.
- [x] Criar modulos sem execucao pesada no import.

Gate de saida:

- [x] `.venv/bin/python -m compileall src scripts tests` executa sem erro.

## Etapa 2 - Componentes Isolados

- [x] Implementar `set_global_seed`.
- [x] Implementar `ReplayBuffer`.
- [x] Implementar `MLPQNetwork`.
- [x] Implementar `VQCQNetwork`.
- [x] Implementar contagem de parametros treinaveis.
- [x] Implementar normalizacao/clipping de observacoes para CartPole.

Gate de saida:

- [x] Forward do MLP retorna `[batch, n_actions]`.
- [x] Forward do VQC retorna `[batch, n_actions]`.
- [x] Replay buffer amostra shapes corretos.

## Etapa 3 - Loop DQN e CLI

- [x] Implementar criacao de ambiente Gymnasium.
- [x] Implementar epsilon-greedy.
- [x] Implementar target network.
- [x] Implementar atualizacao Bellman com Huber loss.
- [x] Salvar CSV com `run_id,agent,env,seed,episode,reward,steps,wall_time_sec`.
- [x] Salvar resumo JSON por run.
- [x] Criar `scripts/train.py`.
- [x] Criar `scripts/count_params.py`.
- [x] Criar `scripts/plot_results.py`.

Gate de saida:

- [x] Um treino curto com 1 seed e poucos episodios gera CSV em `outputs/results/`.

## Etapa 4 - Validacao

- [x] Criar testes de smoke.
- [x] Rodar `.venv/bin/python -m compileall src scripts tests`.
- [x] Rodar teste de forward MLP.
- [x] Rodar teste de forward VQC.
- [x] Rodar treino curto MLP.
- [x] Rodar treino curto VQC.

Gate de saida:

- [x] Validacoes reportadas sem inventar metricas de desempenho.

## Etapa 5 - Experimentos Reais

- [ ] Rodar MLP CartPole com 3 a 5 seeds.
- [ ] Rodar VQC CartPole com 3 a 5 seeds.
- [ ] Gerar figura de curva de aprendizado final.
- [x] Gerar tabela de parametros.
- [ ] Atualizar artigo apenas com resultados existentes.

Gate de saida:

- [ ] Existem CSVs, figuras e tabelas correspondentes aos numeros do texto.

## Validacoes Executadas

```bash
.venv/bin/python tests/smoke_test.py
.venv/bin/python scripts/count_params.py --configs configs/cartpole_mlp.yaml configs/cartpole_vqc.yaml
.venv/bin/python scripts/train.py --config configs/cartpole_mlp.yaml --episodes 2 --seeds 0 --batch-size 4 --learning-starts 4
.venv/bin/python scripts/train.py --config configs/cartpole_vqc.yaml --episodes 2 --seeds 0 --batch-size 4 --learning-starts 4
.venv/bin/python scripts/plot_results.py --csv outputs/results/cartpole_mlp_20260607T171341Z.csv outputs/results/cartpole_vqc_20260607T171347Z.csv --window 2 --output outputs/figures/smoke_learning_curve.png
```

Resultados de validacao:

- MLP configurado: 44 parametros treinaveis.
- VQC configurado: 46 parametros treinaveis.
- Smoke training MLP calculou loss.
- Smoke training VQC calculou loss.
- Figura de smoke gerada em `outputs/figures/smoke_learning_curve.png`.

## Criterios de Corte

- Se dependencias nao puderem ser instaladas a tempo, entregar codigo e comandos documentados.
- Se VQC estiver lento, reduzir para L=1 e 3 seeds.
- Se VQC nao aprender, manter resultado como evidencia empirica e discutir limitacoes.
- Se faltar tempo, priorizar CSV, figura principal, tabela de parametros e texto honesto.

## Decisao Apos Piloto

- Piloto MLP: 25 episodios, 514 passos, 0.61s.
- Piloto VQC: 25 episodios, 524 passos, 70.40s.
- Decisao inicial: usar `configs/cartpole_*_emergency.yaml` com 100 episodios e 3 seeds.
- Ajuste apos execucao longa: reduzir para 50 episodios e 3 seeds, com progresso incremental e CSV salvo apos cada seed. A execucao VQC de 100 episodios ficou longa demais sem artefato parcial.

## Resultados Principais Gerados

Configuracao principal recomendada:

- MLP L1 equivalente: `configs/cartpole_mlp_l1_final100.yaml`
- VQC L1: `configs/cartpole_vqc_l1_final100.yaml`
- Seeds: 0, 1, 2
- Episodios: 100
- Parametros: MLP 23, VQC 22

Artefatos:

- CSV MLP: `outputs/results/cartpole_mlp_l1_final100_20260607T175056Z.csv`
- CSV VQC: `outputs/results/cartpole_vqc_l1_final100_20260607T175108Z.csv`
- Curva: `outputs/figures/cartpole_l1_final100_learning_curve.png`
- Circuito: `outputs/figures/vqc_l1_circuit.png`
- Tabela agregada: `outputs/tables/cartpole_l1_final100_summary.csv`
- Tabela LaTeX: `outputs/tables/cartpole_l1_final100_summary.tex`
- Parametros: `outputs/tables/cartpole_l1_parameter_counts.csv`

Resumo numerico:

- MLP: recompensa final media dos ultimos 20 episodios `11.58 +/- 0.06`, melhor recompensa media `59.00`, tempo medio `1.19s`.
- VQC: recompensa final media dos ultimos 20 episodios `11.17 +/- 0.45`, melhor recompensa media `66.67`, tempo medio `97.00s`.
- Interpretacao: o experimento curto nao resolveu CartPole; serve como comparacao controlada de comportamento e custo, nao como demonstracao de vantagem.

## Artigo

- Esqueleto preenchido em `article/main.tex` e `article/sections/`.
- Compilacao validada com:

```bash
cd article
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

- PDF gerado: `article/main.pdf`.
- Observacao: ainda falta substituir pelo template oficial SBC e inserir referencias verificadas.

## Visualizacao

- Checkpoints passaram a ser salvos em `outputs/checkpoints/` para novos treinos.
- Script de renderizacao de checkpoint: `scripts/record_video.py`.
- Script de demonstracao heuristica: `scripts/record_cartpole_heuristic.py`.
- GIF do agente MLP treinado curto: `outputs/videos/cartpole_mlp_demo.gif`.
  - Episodio: 8 passos, recompensa 8.
  - Nao usar como evidencia de bom desempenho.
- GIF heuristico: `outputs/videos/cartpole_heuristic_demo.gif`.
  - Episodio: 500 passos, recompensa 500.
  - Usar apenas como demonstracao visual do ambiente, rotulada como heuristica.

## Revisao Externa e Correcao Experimental

- Claude foi executado com `claude --dangerously-skip-permissions --model opus`.
- `agy --yolo` falhou porque a flag nao existe nesta instalacao.
- `agy --dangerously-skip-permissions` ficou bloqueado por autenticacao Google e expirou.
- Prompt usado: `outputs/reviews/external_reviewer_prompt.md`.
- Resumo: `outputs/reviews/external_review_summary.md`.

Correcoes implementadas:

- Avaliacao gulosa apos treino em `src/qrl/training/dqn.py`.
- Baselines random e heuristico em `scripts/run_baselines.py`.
- Tabela baseada em summaries JSON em `scripts/summarize_json_results.py`.
- Epsilon corrigido para chegar a `0.05` nas configs `*_eval100`.
- Sanity check com MLP maior para provar que o pipeline aprende.

Resultados corrigidos principais:

- Random: `20.65 +/- 2.01`.
- Heuristica: `500.00 +/- 0.00`.
- DQN-MLP pequeno: greedy eval `9.42 +/- 0.20`, 23 parametros.
- DQN-VQC pequeno: greedy eval `9.45 +/- 0.18`, 22 parametros.
- MLP sanity maior: greedy eval `98.65`, 450 parametros, 1 seed.

## Cenario Quantum/Fuzzy

Tentativas DQN:

- `fuzzy_compact` exploratorio continha vazamento de politica e nao deve ser usado como resultado principal.
- `fuzzy_membership_4` seguro nao melhorou o DQN pequeno.
- Exploracao `fuzzy_guided` aumentou retorno durante exploracao inicial, mas nao melhorou a politica gulosa final.

Resultado positivo honesto:

- Foi implementada destilacao fuzzy-teacher em `scripts/train_fuzzy_distill.py`.
- MLP destilado com membership fuzzy limpa, 23 parametros, 3 seeds: greedy eval `440.72 +/- 102.68`.
- VQC L1 destilado com raw state, 22 parametros, 3 seeds: greedy eval `96.53 +/- 38.13`.
- VQC L1 destilado com membership fuzzy limpa, 22 parametros, 3 seeds: greedy eval `500.00 +/- 0.00`.
- VQC L3 destilado com raw state, 46 parametros, 3 seeds: greedy eval `321.87 +/- 188.69`.
- Tabela final: `outputs/tables/fuzzy_final_scenarios.csv`.
- Videos:
  - `outputs/videos/cartpole_fuzzy_distill_mlp_demo.gif`
  - `outputs/videos/cartpole_fuzzy_distill_vqc_demo.gif`
  - `outputs/videos/cartpole_fuzzy_membership_distill_vqc_l1_demo.gif`

Uso no artigo:

- Reportar como experimento separado de destilacao supervisionada de uma politica fuzzy.
- Nao apresentar como DQN aprendendo de recompensa.
- Nao apresentar como vantagem quantica.
- Claim defensavel: o VQC pequeno representa parcialmente uma politica fuzzy util e supera random/raw DQN-VQC quando recebe supervisao estruturada.
- Claim atualizado: com membership fuzzy descritiva e destilacao fuzzy-teacher, o VQC L1 de 22 parametros resolveu CartPole nas 3 seeds testadas. Isso e um resultado de representacao/supervisao fuzzy, nao vantagem quantica nem RL puro.
