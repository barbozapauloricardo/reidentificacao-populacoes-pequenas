# Risco de reidentificação em populações pequenas

Código e resultados do artigo **"Quando a anonimização não basta: um modelo causal-probabilístico do risco de reidentificação em pesquisas com populações pequenas"**, de Paulo Ricardo Barboza (PPG em Planejamento e Análise de Políticas Públicas, FCHS/Unesp). O artigo é o trabalho final da disciplina *Big Data, Análise de Dados e Modelos Probabilísticos* (Prof. Dr. Carlos Dias Maciel).

O estudo estima quantos respondentes de um questionário "anônimo" ficam identificáveis numa turma pequena e usa redes bayesianas para raciocinar sobre esse risco: inferência preditiva e diagnóstica, d-separação, intervenção do(·), explicação mais relevante (MRE) e aprendizado de estrutura.

> **Todos os dados são sintéticos.** As distribuições usadas são hipóteses de modelagem, não estatísticas de nenhum programa real. Nenhum dado pessoal foi coletado, lido ou vinculado a pessoas.

## Modelo, em resumo

- **Rede geradora:** sete quase-identificadores (gênero, raça/cor, vínculo, idade, formação, linha de pesquisa e cidade), com as arestas Vínculo → Idade e Formação → Linha. Gera as turmas sintéticas.
- **Desenhos do questionário (D0 a D3):** intervenções do(·) sobre o instrumento de coleta, do mais detalhado ao mais restritivo.
- **Rede de reidentificação:** Desenho (D), Tamanho da turma (N) e Composição (S) → Tamanho da classe de equivalência (F); F e Tipo de adversário (K) → Reidentificação (Re). A tabela de F é calculada analiticamente a partir da rede geradora para as 64 combinações de D, N e S.
- **Métricas:** proporção de registros únicos (k = 1), risco médio E[1/F] e proporção de registros com k < 5, conferidas contra expressões analíticas (diferença máxima de 0,54 ponto percentual).

**Experimentos**

| Experimento | O que testa | Script |
|---|---|---|
| E1 | Tamanho da turma (N de 10 a 5.000) | `sim_reidentificacao.py` |
| E2 | Acúmulo de quase-identificadores (N = 30) | `sim_reidentificacao.py` |
| E3 | Resposta incompleta (60% de adesão) | `sim_reidentificacao.py` |
| E4 | Sensibilidade a quatro cenários de parâmetros | `sensibilidade.py` |
| E5 | Inferência na rede de reidentificação: preditiva, diagnóstica, d-separação, explicação por exclusão, observar × intervir | `rede_reidentificacao.py` |
| E6 | Explicação mais relevante (MRE) dada uma reidentificação, com sensibilidade às hipóteses arbitradas | `rede_reidentificacao.py`, `sensibilidade_mre.py` |
| E7 | Aprendizado de estrutura (PC e Hill Climbing com BIC) por tamanho de amostra | `aprendizado_estrutura.py` |

## Estrutura

| Arquivo ou pasta | Conteúdo |
|---|---|
| `sim_reidentificacao.py` | Rede geradora, métricas, expressões analíticas e experimentos E1 a E3 |
| `sensibilidade.py` | Análise de sensibilidade (E4) |
| `rede_reidentificacao.py` | Rede bayesiana de reidentificação (E5 e E6); saídas `A1` a `A5` |
| `sensibilidade_mre.py` | Sensibilidade da MRE a P(K = interno) e à chance de vínculo do adversário externo; saída `A6` |
| `aprendizado_estrutura.py` | Aprendizado de estrutura (E7); saídas `B1` |
| `figuras_artigo.py` | Figuras do artigo, a partir dos CSVs |
| `resultados/` | CSVs de cada experimento e `ambiente.txt` (versões e tempos de execução) |
| `figuras/` | Figuras do artigo (PDF e PNG) |

## Como reproduzir

Requer Python 3.10 a 3.14. O pgmpy 1.1 ainda não é compatível com o Python 3.15; quem tiver o 3.15 instalado pode instalar também o 3.13 e rodar os comandos com `py -3.13` no lugar de `python`.

```bash
pip install -r requirements.txt
python sim_reidentificacao.py     # cerca de 4 min; E1 a E3
python sensibilidade.py           # cerca de 30 s; E4
python rede_reidentificacao.py    # cerca de 2 min; E5 e E6
python sensibilidade_mre.py       # cerca de 1 min; sensibilidade da MRE
python aprendizado_estrutura.py   # cerca de 20 s; E7
python figuras_artigo.py          # gera figuras/
```

A semente aleatória é fixa (2026), então os resultados se repetem.

## Principais resultados (N = 30, cenário-base)

| Desenho | Únicos | P(Re), adversário interno | P(Re), adversário externo |
|---|---|---|---|
| D0: tudo detalhado | 99,7% | 93,1% | 8,8% |
| D1: idade em faixas | 97,5% | 82,6% | 7,5% |
| D2: atributos agregados | 87,7% | 77,0% | 6,5% |
| D3: sem raça/cor e sem cidade | 56,2% | 60,8% | 4,0% |

- Na análise de sensibilidade, o desenho D3 manteve entre 16,0% e 56,1% de respondentes únicos.
- Dada uma reidentificação, a explicação mais relevante foi o adversário interno (fator de Bayes generalizado de 12,2), e não o desenho do questionário. Essa conclusão se mantém sempre que o adversário externo vincula até 10% dos registros únicos, qualquer que seja P(K = interno).
- Com 30 respondentes, nenhum algoritmo recuperou a estrutura da rede geradora; a busca com BIC só a recuperou de forma consistente a partir de 1.000.

## Licença

Código sob licença MIT (ver `LICENSE`).
