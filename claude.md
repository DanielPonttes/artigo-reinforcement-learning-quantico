# Instrucoes para Claude

Voce esta ajudando a finalizar um artigo Undergraduate para ENIAC 2026 sobre Quantum Reinforcement Learning com VQC em DQN.

## Contexto Essencial

Leia primeiro:

1. `docs/IMPLEMENTATION_PLAN.md`
2. `docs/FOLDER_STRUCTURE.md`
3. `agent.md`

## Tarefa Principal

Ajudar a transformar resultados experimentais reais em texto cientifico curto, anonimo e submetivel no template SBC.

## Escopo Atual

Trabalhar com escopo minimo:

- CartPole-v1;
- DQN-MLP vs DQN-VQC;
- parametros treinaveis equiparados;
- 3 a 5 seeds;
- sem segundo ambiente, salvo se resultados ja existirem.

## Estilo de Escrita

- Portugues academico claro.
- Sem promessas de vantagem quantica.
- Sem linguagem promocional.
- Resultados negativos ou mistos devem ser tratados como achados validos.
- A discussao deve separar evidencia empirica, interpretacao e limitacoes.

## Cuidados

- Nao invente resultados, medias, desvios, p-values ou tempos.
- Nao cite referencias sem verificar dados bibliograficos.
- Mantenha o texto duplo-cego.
- Nao mencionar instituicao, orientador, laboratorio, cidade ou hardware de forma identificavel alem do necessario para reproducibilidade.

## Estrutura Preferida do Artigo

1. Resumo / Abstract.
2. Introducao.
3. Fundamentacao e trabalhos relacionados.
4. Metodologia.
5. Configuracao experimental.
6. Resultados e discussao.
7. Limitacoes.
8. Conclusao.

## Pergunta de Pesquisa

Com contagem de parametros aproximadamente equiparada, como um DQN-VQC se compara a um DQN-MLP em eficiencia amostral, estabilidade e custo de treinamento no CartPole-v1?

## Frase de Posicionamento

Este trabalho nao busca demonstrar vantagem quantica, mas avaliar de forma controlada o comportamento de um aproximador variacional quantico em um pipeline DQN classico e reprodutivel.
