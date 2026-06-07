# Artigo ENIAC 2026 - Quantum Reinforcement Learning com VQC

Repositório de planejamento, experimentos e escrita do artigo sobre uso de Variational Quantum Circuits (VQC) como aproximador de Q-values em agentes DQN.

## Estado Atual

- Data local registrada no ambiente: 2026-06-07, America/Sao_Paulo.
- Deadline informado pelo plano original: 2026-06-08 23:59 UTC-12, equivalente a 2026-06-09 08:59 BRT.
- Viabilidade: submissão completa ainda é possível apenas com escopo mínimo e execução disciplinada. O escopo com 2 ambientes, 10 seeds por configuração e ablação ampla não é realista se os resultados ainda não existem.

## Arquivos de Coordenação

- `docs/IMPLEMENTATION_PLAN.md`: plano revisado, viabilidade, cortes de escopo e cronograma final.
- `docs/FOLDER_STRUCTURE.md`: estrutura recomendada do repositório.
- `agent.md`: contexto comum para qualquer agente de IA trabalhando no projeto.
- `claude.md`: instruções compactas para uso com Claude.
- `codex.md`: instruções operacionais para uso com Codex.

## Regra de Escopo

Prioridade absoluta: artigo submetível, honesto e duplo-cego. Se houver conflito entre ambição experimental e prazo, cortar o segundo ambiente, reduzir ablação e manter apenas CartPole com comparação VQC vs MLP parametrizada.

## Setup Local

Use `python3`, nao `python`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Validacao Rapida

```bash
.venv/bin/python tests/smoke_test.py
.venv/bin/python scripts/count_params.py --configs configs/cartpole_mlp.yaml configs/cartpole_vqc.yaml
```

Contagem atual de parametros:

- MLP: 44 parametros treinaveis.
- VQC: 46 parametros treinaveis.

Hardware/PyTorch verificados na `.venv`:

- PyTorch: `2.12.0+cu130`.
- CUDA disponivel: sim.
- GPU: `NVIDIA GeForce RTX 5090`.
- Capability: `(12, 0)`.

## Treinos de Smoke

Estes comandos validam o caminho de treino, mas nao produzem resultado cientifico:

```bash
.venv/bin/python scripts/train.py --config configs/cartpole_mlp.yaml --episodes 2 --seeds 0 --batch-size 4 --learning-starts 4
.venv/bin/python scripts/train.py --config configs/cartpole_vqc.yaml --episodes 2 --seeds 0 --batch-size 4 --learning-starts 4
```

## Experimentos Reais

Comece com 3 seeds se o prazo estiver apertado:

```bash
.venv/bin/python scripts/train.py --config configs/cartpole_mlp.yaml --seeds 0 1 2
.venv/bin/python scripts/train.py --config configs/cartpole_vqc.yaml --seeds 0 1 2
```

Se houver tempo, rode 5 seeds conforme os arquivos de config.

## Artefatos Gerados

Conjunto principal recomendado para o artigo neste momento: configuracao rasa `L=1`, 100 episodios, 3 seeds.

Entradas:

- `configs/cartpole_mlp_l1_final100.yaml`
- `configs/cartpole_vqc_l1_final100.yaml`

Resultados:

- `outputs/results/cartpole_mlp_l1_final100_20260607T175056Z.csv`
- `outputs/results/cartpole_vqc_l1_final100_20260607T175108Z.csv`
- `outputs/tables/cartpole_l1_final100_summary.csv`
- `outputs/tables/cartpole_l1_final100_summary.tex`
- `outputs/tables/cartpole_l1_parameter_counts.csv`
- `outputs/figures/cartpole_l1_final100_learning_curve.png`
- `outputs/figures/vqc_l1_circuit.png`

Resumo L1:

- MLP: 23 parametros, recompensa final media dos ultimos 20 episodios `11.58 +/- 0.06`, tempo medio `1.19s`.
- VQC: 22 parametros, recompensa final media dos ultimos 20 episodios `11.17 +/- 0.45`, tempo medio `97.00s`.
- Interpretacao: nenhum agente resolveu CartPole neste orcamento curto; o VQC teve desempenho semelhante em recompensa, mas custo de simulacao muito maior.

## Visualizacao

Foram gerados GIFs para apresentacao visual do ambiente:

- Agente MLP treinado curto: `outputs/videos/cartpole_mlp_demo.gif`
  - Resultado do episodio renderizado: 8 passos, recompensa 8.
  - Serve para provar que o pipeline de renderizacao do agente funciona, mas nao e visualmente convincente.
- Politica heuristica de CartPole: `outputs/videos/cartpole_heuristic_demo.gif`
  - Resultado: 500 passos, recompensa 500.
  - Usar apenas como demonstracao visual do ambiente/jogo, claramente rotulada como heuristica e nao como resultado experimental do artigo.

Scripts:

```bash
.venv/bin/python scripts/record_video.py --config configs/cartpole_mlp_l1_pilot.yaml --checkpoint outputs/checkpoints/cartpole_mlp_l1_pilot_seed0_20260607T180022Z.pt --output outputs/videos/cartpole_mlp_demo.gif --seed 42 --max-steps 500 --fps 30
.venv/bin/python scripts/record_cartpole_heuristic.py --output outputs/videos/cartpole_heuristic_demo.gif --seed 42 --max-steps 500 --fps 30
```

## Avaliacao de Relevancia

O artigo e relevante se for apresentado como estudo empirico controlado de viabilidade, custo e estabilidade de DQN-VQC. Ele fica fraco se for apresentado como proposta de desempenho superior, porque os resultados atuais nao mostram isso. A mensagem tecnica correta e: sob orcamento de parametros equiparado e simulacao classica, o VQC teve recompensa semelhante ao MLP pequeno, mas custo de treinamento muito maior e sem resolver CartPole.

## Revisao Externa e Experimentos Corrigidos

Claude foi chamado via CLI com `--model opus` e apontou um problema importante: faltavam baseline aleatorio e avaliacao gulosa, e os agentes pequenos estavam abaixo do random. O Antigravity/Gemini nao foi executado porque `agy --yolo` nao existe nesta instalacao e o fallback `agy --dangerously-skip-permissions` exigiu autenticacao Google interativa.

Artefatos corrigidos:

- Random: `outputs/results/random_policy_20260607T180958Z.csv`
- Heuristica: `outputs/results/heuristic_policy_20260607T180958Z.csv`
- MLP pequeno corrigido: `outputs/results/cartpole_mlp_l1_eval100_20260607T181005Z.csv`
- VQC pequeno corrigido: `outputs/results/cartpole_vqc_l1_eval100_20260607T181047Z.csv`
- MLP sanity maior: `outputs/results/cartpole_mlp_sanity_solve_20260607T181006Z.csv`
- Tabela principal: `outputs/tables/cartpole_main_eval_summary.csv`
- Figura principal: `outputs/figures/cartpole_l1_eval100_with_baselines.png`

Resumo corrigido:

- Random: avaliacao/media final `20.65 +/- 2.01`.
- Heuristica: `500.00 +/- 0.00`.
- DQN-MLP pequeno, 23 parametros: greedy eval `9.42 +/- 0.20`, tempo medio `1.72s`.
- DQN-VQC pequeno, 22 parametros: greedy eval `9.45 +/- 0.18`, tempo medio `75.97s`.
- MLP sanity, 450 parametros: greedy eval `98.65` em 1 seed.

Interpretacao atual: o resultado principal e um regime de falha controlado. Os agentes pequenos ficam abaixo do baseline aleatorio; o VQC nao melhora desempenho e custa muito mais em simulacao. O sanity check com MLP maior mostra que o pipeline DQN aprende quando ha mais capacidade.

## Cenario Quantum/Fuzzy com Ganho

Foi implementado e otimizado um experimento separado de destilacao fuzzy-teacher:

- Script: `scripts/train_fuzzy_distill.py`
- Tabela final: `outputs/tables/fuzzy_final_scenarios.csv`
- MLP destilado com membership fuzzy limpa, 23 parametros, 3 seeds: greedy eval `440.72 +/- 102.68`.
- VQC L1 destilado com raw state, 22 parametros, 3 seeds: greedy eval `96.53 +/- 38.13`.
- VQC L1 destilado com membership fuzzy limpa, 22 parametros, 3 seeds: greedy eval `500.00 +/- 0.00`.
- VQC L3 destilado com raw state, 46 parametros, 3 seeds: greedy eval `321.87 +/- 188.69`.
- Random: `20.65 +/- 2.01`.
- Heuristica teacher: `500.00 +/- 0.00`.

Interpretacao correta: ha ganho no cenario fuzzy/quantum quando o VQC pequeno recebe supervisao de uma politica fuzzy teacher e uma representacao fuzzy descritiva sem vazamento de acao. Isso nao e DQN puro e nao prova vantagem quantica; mostra que um VQC compacto consegue representar uma politica fuzzy util e superar random/raw DQN-VQC.

Videos:

- `outputs/videos/cartpole_fuzzy_distill_mlp_demo.gif`
- `outputs/videos/cartpole_fuzzy_distill_vqc_demo.gif`
- `outputs/videos/cartpole_fuzzy_membership_distill_vqc_l1_demo.gif`
