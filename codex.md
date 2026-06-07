# Instrucoes para Codex

Voce esta trabalhando em `/home/daniel/artigo-reinforcement-learning-quantico`.

## Objetivo Operacional

Produzir rapidamente um pacote submetivel:

- codigo minimo de experimentos;
- resultados em CSV;
- figuras e tabelas;
- artigo SBC anonimo.

## Ordem de Execucao

1. Ler `docs/IMPLEMENTATION_PLAN.md`.
2. Validar estrutura com `docs/FOLDER_STRUCTURE.md`.
3. Implementar primeiro o caminho minimo CartPole MLP.
4. Implementar VQC apenas no nivel necessario para comparar com MLP.
5. Criar scripts reproduziveis para treino e figuras.
6. Rodar smoke tests antes de execucoes longas.
7. Atualizar o artigo com resultados reais.

## Escopo de Codigo

Priorizar estes componentes:

- `src/qrl/agents/`: DQN, replay buffer, politica epsilon-greedy, target network.
- `src/qrl/models/`: `MLPQNetwork` e `VQCQNetwork`.
- `src/qrl/training/`: loop de treino e avaliacao.
- `src/qrl/utils/`: seeds, logging, contagem de parametros.
- `scripts/train.py`: executar um experimento a partir de config.
- `scripts/plot_results.py`: gerar figuras finais.
- `scripts/count_params.py`: gerar tabela de parametros.

## Regras de Implementacao

- Usar Gymnasium, PyTorch e PennyLane.
- Fixar seeds em Python, NumPy, PyTorch e ambiente.
- Salvar resultados em CSV com colunas: `run_id`, `agent`, `env`, `seed`, `episode`, `reward`, `steps`, `wall_time_sec`.
- Salvar configs junto dos resultados.
- Nao depender de notebook para resultados finais.
- Manter hiperparametros identicos entre MLP e VQC quando tecnicamente possivel.
- Registrar diferencas inevitaveis na metodologia.

## Validacoes Minimas

Antes de rodar seeds:

- forward pass do MLP retorna shape `[batch, n_actions]`;
- forward pass do VQC retorna shape `[batch, n_actions]`;
- replay buffer amostra batch correto;
- uma atualizacao DQN executa sem erro;
- um episodio completo de CartPole executa e gera log.

## Decisoes de Corte

Se VQC estiver lento:

- reduzir episodios;
- reduzir seeds para 3;
- usar L=1;
- manter CartPole apenas.

Se VQC nao aprender:

- manter resultado;
- escrever discussao honesta;
- nao ajustar hiperparametros ate comprometer comparacao.

Se faltar tempo:

1. Gerar uma figura principal correta.
2. Gerar tabela de parametros.
3. Escrever limitacoes fortes.
4. Submeter artigo pequeno e honesto.

## Cuidados com o Artigo

- Nao adicionar nomes, instituicao ou agradecimentos.
- Nao declarar vantagem quantica.
- Nao escrever resultados antes de existirem arquivos em `outputs/`.
- Conferir limite de 12 paginas.
- Conferir referencias bibliograficas antes da submissao.
