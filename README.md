# Risco de reidentificação em populações pequenas

Código e resultados do artigo **"Quando a anonimização não basta: um modelo causal-probabilístico do risco de reidentificação em pesquisas com populações pequenas"**, de Paulo Ricardo Barboza (PPG em Planejamento e Análise de Políticas Públicas, FCHS/Unesp). O artigo é o trabalho final da disciplina *Big Data, Análise de Dados e Modelos Probabilísticos* (Prof. Dr. Carlos Dias Maciel).

A simulação estima quantos respondentes de um questionário "anônimo" ficam identificáveis numa turma pequena, conforme o tamanho da turma e o desenho do questionário.

> **Todos os dados são sintéticos.** As distribuições usadas são hipóteses de modelagem, não estatísticas de nenhum programa real. Nenhum dado pessoal foi coletado, lido ou vinculado a pessoas.

## Modelo, em resumo

- **População:** gerada por uma rede bayesiana com sete quase-identificadores: gênero, raça/cor, vínculo, idade, formação, linha de pesquisa e cidade. A idade depende do vínculo, e a linha de pesquisa depende da formação.
- **Desenhos do questionário (D0 a D3):** tratados como intervenções do(·) sobre o instrumento de coleta, do mais detalhado ao mais restritivo.
- **Métricas:**
  - proporção de registros únicos (k = 1);
  - risco médio E[1/F];
  - proporção de registros com k < 5.
- **Validação:** expressões analíticas, conferidas contra o Monte Carlo (diferença máxima de 0,54 ponto percentual).

**Experimentos**

| Experimento | O que testa |
|---|---|
| E1 | Tamanho da turma (N de 10 a 5.000) |
| E2 | Acúmulo de quase-identificadores (N = 30) |
| E3 | Resposta incompleta (60% de adesão) |
| E4 | Sensibilidade a quatro cenários de parâmetros |

## Estrutura

| Arquivo ou pasta | Conteúdo |
|---|---|
| `sim_reidentificacao.py` | Modelo, métricas e experimentos E1 a E3 |
| `sensibilidade.py` | Análise de sensibilidade (E4) |
| `figuras_artigo.py` | Figuras do artigo, a partir dos CSVs |
| `resultados/` | CSVs de cada experimento e `ambiente.txt` (versões e tempos de execução) |
| `figuras/` | Figuras do artigo (PDF e PNG) |

## Como reproduzir

Requer Python 3.10 ou superior.

```bash
pip install -r requirements.txt
python sim_reidentificacao.py   # cerca de 4 min; gera resultados/
python sensibilidade.py         # cerca de 30 s
python figuras_artigo.py        # gera figuras/
```

A semente aleatória é fixa (2026), então os resultados se repetem.

## Principais resultados (N = 30, cenário-base)

| Desenho | Únicos | Risco médio |
|---|---|---|
| D0: tudo detalhado | 99,7% | 99,9% |
| D1: idade em faixas | 97,5% | 98,7% |
| D2: atributos agregados | 87,7% | 93,6% |
| D3: sem raça/cor e sem cidade | 56,2% | 75,3% |

Na análise de sensibilidade, o desenho D3 manteve entre 16,0% e 56,1% de respondentes únicos.

## Licença

Código sob licença MIT (ver `LICENSE`).
