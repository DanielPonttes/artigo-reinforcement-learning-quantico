# Estrutura Basica de Pastas

```text
.
├── article/
│   ├── main.tex
│   └── sections/
│       ├── 01_introduction.tex
│       ├── 02_related_work.tex
│       ├── 03_methodology.tex
│       ├── 04_experimental_setup.tex
│       ├── 05_results.tex
│       ├── 06_limitations.tex
│       └── 07_conclusion.tex
├── configs/
│   ├── cartpole_mlp.yaml
│   └── cartpole_vqc.yaml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── IMPLEMENTATION_PLAN.md
│   └── FOLDER_STRUCTURE.md
├── experiments/
│   └── notebooks/
├── outputs/
│   ├── figures/
│   ├── results/
│   ├── tables/
│   └── videos/
├── references/
├── scripts/
├── src/
│   └── qrl/
│       ├── agents/
│       ├── envs/
│       ├── models/
│       ├── training/
│       └── utils/
├── tests/
├── agent.md
├── claude.md
├── codex.md
└── README.md
```

## Papel de Cada Pasta

`article/`

Contem o artigo no template SBC. O arquivo `main.tex` deve ser anonimo para revisao duplo-cega. As secoes ficam quebradas para facilitar edicao por agentes e reduzir conflitos.

`configs/`

Arquivos de configuracao reproduziveis para cada experimento. Devem registrar ambiente, agente, seeds, hiperparametros, profundidade do circuito e backend quantico.

`data/`

Dados brutos e processados. Para este projeto, deve conter apenas artefatos pequenos e reproduziveis. Resultados grandes devem ficar fora do repositorio ou em storage separado.

`docs/`

Planejamento, estrutura, decisoes e notas de reproducibilidade.

`experiments/notebooks/`

Notebooks exploratorios. Resultados finais nao devem depender de notebooks; scripts em `scripts/` devem conseguir reproduzir tabelas e figuras.

`outputs/`

Artefatos gerados: CSVs de resultados, figuras, tabelas e videos/GIFs. Figuras finais usadas no artigo devem vir daqui.

`references/`

PDFs, BibTeX e notas de referencias. Conferir licenca antes de versionar PDFs.

`scripts/`

Entrypoints executaveis:

- treino;
- avaliacao;
- geracao de graficos;
- contagem de parametros;
- renderizacao de video.

`src/qrl/`

Codigo fonte do projeto:

- `agents/`: DQN, replay buffer, epsilon schedule.
- `envs/`: wrappers e normalizacao de estado.
- `models/`: MLP Q-network e VQC Q-network.
- `training/`: loop de treino, avaliacao e checkpointing.
- `utils/`: seeds, logging, metricas e contagem de parametros.

`tests/`

Testes pequenos de sanidade:

- shapes de modelos;
- contagem de parametros;
- forward pass do VQC;
- um passo de treino sem erro;
- reproducibilidade basica de seeds.
