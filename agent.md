# Contexto Comum para Agentes

## Objetivo

Finalizar uma submissao Undergraduate para o ENIAC 2026 sobre DQN com Variational Quantum Circuit (VQC) como aproximador de Q-values, comparado a um baseline MLP-DQN com contagem de parametros equiparada.

## Situacao Temporal

- Data local do ambiente quando este arquivo foi criado: 2026-06-07.
- Deadline informado pelo planejamento: 2026-06-08 23:59 UTC-12, equivalente a 2026-06-09 08:59 BRT.
- Confirmar prazo oficial no JEMS3/site do evento antes da submissao.

## Escopo Recomendado

Priorizar submissao minima:

- Ambiente: CartPole-v1.
- Agentes: DQN-MLP e DQN-VQC.
- Seeds: 5 se possivel, 3 se necessario.
- VQC: 4 qubits, angle encoding, CNOT em anel, readout por Pauli-Z + camada linear.
- Profundidade: L=3 por padrao; usar L=1 se estabilidade ou tempo forem problemas.

Nao investir em LunarLander, MiniGrid ou ablaçoes amplas se os resultados principais ainda nao estiverem prontos.

## Linha Editorial

O artigo deve ser honesto e conservador:

- nao alegar vantagem quantica;
- nao alegar superioridade geral;
- nao extrapolar simulacao para hardware real;
- discutir limitacoes de simulacao, barren plateaus, poucos qubits e numero de seeds;
- se o VQC perder para o MLP, apresentar isso como resultado valido.

## Duplo-Cego

Antes de gerar o PDF:

- remover nomes de autores;
- remover instituicao;
- remover agradecimentos;
- remover caminhos locais;
- evitar links identificaveis;
- limpar metadados do PDF se necessario.

## Prioridades

1. Codigo e resultados minimos reproduziveis.
2. Figuras e tabelas essenciais.
3. Texto claro no template SBC.
4. Revisao de anonimato e limite de paginas.

## Fontes de Verdade

- Plano revisado: `docs/IMPLEMENTATION_PLAN.md`.
- Estrutura do repositorio: `docs/FOLDER_STRUCTURE.md`.
- Resultados finais: arquivos em `outputs/`.
- Configuracoes finais: arquivos em `configs/`.

## Regras para Agentes

- Nao inventar resultados.
- Nao inserir numeros sem CSV/log correspondente.
- Sempre registrar seed, ambiente, agente, hiperparametros e backend.
- Manter mudancas pequenas e orientadas a submissao.
- Ao escrever, preferir frases diretas e tecnicas.
- Ao implementar, preferir scripts reproduziveis a notebooks.
