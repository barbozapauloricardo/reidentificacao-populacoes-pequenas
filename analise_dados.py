"""
Análise exploratória e descoberta de estrutura sobre os dados sintéticos.

Segue a ordem usual dos trabalhos de descoberta causal:
  1. análise exploratória dos dados (distribuições, associações, unicidade);
  2. testes de independência condicional (qui-quadrado de Pearson e G²);
  3. algoritmos de descoberta de estrutura (PC, baseado em testes, e Hill
     Climbing com BIC, baseado em pontuação), com e sem conhecimento prévio;
  4. comparação da topologia aprendida com a rede geradora.

Os dados analisados são os de dados_sinteticos/ (gerados por
gerar_dados_exemplo.py): uma turma de 30 respondentes e uma população de 5.000.
Todos são sintéticos.

Saídas (pasta resultados/):
  C1_marginais.csv            frequências de cada categoria (N = 30 e N = 5.000)
  C2_cramer.csv               V de Cramér para os 21 pares de variáveis
  C3_testes_independencia.csv testes qui-quadrado e G² para os 21 pares
  C4_testes_condicionais.csv  testes condicionais que distinguem as estruturas
  C5_conhecimento_previo.csv  Hill Climbing com e sem conhecimento prévio (20 repetições)
"""
import time
import warnings
import logging
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from pgmpy.estimators import PC, HillClimbSearch, BIC, ExpertKnowledge
from pgmpy.estimators.CITests import chi_square, g_sq
import sim_reidentificacao as sim
import aprendizado_estrutura as ae

warnings.filterwarnings("ignore")
logging.getLogger("pgmpy").setLevel(logging.ERROR)
OUT = Path("resultados")
DADOS = Path("dados_sinteticos")
VARS = ["genero", "raca", "idade", "formacao", "linha", "vinculo", "cidade"]
ALFA = 0.05

# Conhecimento prévio de um especialista, que não depende dos dados:
# nenhum atributo do questionário causa o gênero ou a raça/cor, e a linha de
# pesquisa é escolhida depois da formação, do vínculo e da cidade.
PROIBIDAS = [(o, d) for d in ("genero", "raca") for o in VARS if o != d]
PROIBIDAS += [("linha", d) for d in ("formacao", "vinculo", "cidade")]
CONHECIMENTO = ExpertKnowledge(forbidden_edges=PROIBIDAS)


def carregar():
    """Lê a turma de 30 e a população de 5.000, com a idade em faixas de 10 anos."""
    turma = pd.read_csv(DADOS / "turma_N30.csv")[VARS]
    pop = pd.read_csv(DADOS / "populacao_N5000.csv")[VARS]
    return ae.preparar(turma), ae.preparar(pop)


def marginais(df):
    return pd.concat({v: df[v].value_counts(normalize=True) for v in VARS}, names=["variavel", "categoria"])


def cramer(df, a, b):
    """V de Cramér (sem correção de viés) entre duas variáveis categóricas."""
    tab = pd.crosstab(df[a], df[b])
    if min(tab.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(tab, correction=False)[0]
    return float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))))


def matriz_cramer(df):
    m = pd.DataFrame(np.eye(len(VARS)), index=VARS, columns=VARS)
    for a, b in combinations(VARS, 2):
        m.loc[a, b] = m.loc[b, a] = cramer(df, a, b)
    return m


def frac_esperado_baixo(df, a, b):
    """Proporção de células da tabela de contingência com frequência esperada < 5."""
    tab = pd.crosstab(df[a], df[b])
    esp = chi2_contingency(tab, correction=False)[3]
    return float((esp < 5).mean())


def testes_pares(df):
    """Qui-quadrado e G² para cada par de variáveis (independência marginal)."""
    linhas = []
    for a, b in combinations(VARS, 2):
        _, p_chi, gl = chi_square(a, b, [], df, boolean=False)
        _, p_g, _ = g_sq(a, b, [], df, boolean=False)
        linhas.append({"X": a, "Y": b, "aresta_verdadeira": frozenset((a, b)) in ae.REAL,
                       "gl": int(gl), "p_qui2": p_chi, "p_G2": p_g,
                       "dependente_qui2": p_chi < ALFA, "dependente_G2": p_g < ALFA,
                       "celulas_esperado_menor_5": frac_esperado_baixo(df, a, b)})
    return pd.DataFrame(linhas)


# Independências condicionais que a rede geradora implica, e que um algoritmo
# baseado em testes precisa confirmar para não acrescentar arestas.
CONDICIONAIS = [("idade", "formacao", ["vinculo"]), ("linha", "vinculo", ["formacao"]),
                ("idade", "linha", ["vinculo", "formacao"]), ("genero", "idade", ["vinculo"])]


def testes_condicionais(df):
    linhas = []
    for a, b, z in CONDICIONAIS:
        _, p_chi, gl = chi_square(a, b, z, df, boolean=False)
        _, p_g, _ = g_sq(a, b, z, df, boolean=False)
        linhas.append({"teste": f"{a} ⊥ {b} | {', '.join(z)}", "gl": int(gl),
                       "p_qui2": p_chi, "p_G2": p_g})
    return pd.DataFrame(linhas)


def aprender(df, metodo, conhecimento=None, teste="chi_square"):
    if metodo == "PC":
        m = PC(df).estimate(ci_test=teste, significance_level=ALFA, return_type="pdag",
                            expert_knowledge=conhecimento, show_progress=False, n_jobs=1)
    else:
        m = HillClimbSearch(df).estimate(scoring_method=BIC(df), expert_knowledge=conhecimento,
                                         show_progress=False)
    return m


def arestas_legiveis(modelo):
    """Arestas orientadas (a → b) e não orientadas (a — b) de um DAG ou PDAG."""
    dirig = getattr(modelo, "directed_edges", None)
    if dirig is None:
        return sorted(f"{a} → {b}" for a, b in modelo.edges())
    nao = {frozenset(e) for e in modelo.undirected_edges}
    return sorted([f"{a} → {b}" for a, b in dirig] + [" — ".join(sorted(e)) for e in nao])


def conhecimento_previo(tamanhos=(30, 100, 300, 1000), reps=20, semente=2026):
    """Recuperação do esqueleto pelo Hill Climbing com e sem conhecimento prévio."""
    sim.rng = np.random.default_rng(semente)
    linhas = []
    for n in tamanhos:
        for r in range(reps):
            dados = ae.preparar(sim.gerar_turma(n))
            for rotulo, ck in (("sem conhecimento", None), ("com conhecimento", CONHECIMENTO)):
                m = aprender(dados, "HC", ck)
                linhas.append({"N": n, "rep": r, "versao": rotulo, **ae.avaliar(ae.esqueleto(m.edges()))})
    res = pd.DataFrame(linhas)
    return (res.groupby(["versao", "N"])
            .agg(recuperacao_exata=("exato", "mean"), arestas_reais_achadas=("verdadeiros_pos", "mean"),
                 falsos_pos=("falsos_pos", "mean"), SHD_medio=("SHD_esqueleto", "mean"))
            .reset_index())


if __name__ == "__main__":
    t0 = time.perf_counter()
    turma, pop = carregar()
    pd.concat({"N=30": marginais(turma), "N=5000": marginais(pop)}, axis=1).to_csv(OUT / "C1_marginais.csv")
    linhas = []
    for a, b in combinations(VARS, 2):
        linhas.append({"X": a, "Y": b, "aresta_verdadeira": frozenset((a, b)) in ae.REAL,
                       "V_N30": cramer(turma, a, b), "V_N5000": cramer(pop, a, b)})
    pd.DataFrame(linhas).to_csv(OUT / "C2_cramer.csv", index=False)
    tp = pd.concat({"N=30": testes_pares(turma), "N=5000": testes_pares(pop)}, names=["dados"]).reset_index(level=0)
    tp.to_csv(OUT / "C3_testes_independencia.csv", index=False)
    tc = pd.concat({"N=30": testes_condicionais(turma), "N=5000": testes_condicionais(pop)},
                   names=["dados"]).reset_index(level=0)
    tc.to_csv(OUT / "C4_testes_condicionais.csv", index=False)
    ck = conhecimento_previo()
    ck.to_csv(OUT / "C5_conhecimento_previo.csv", index=False)
    with open(OUT / "ambiente.txt", "a", encoding="utf-8") as fh:
        fh.write(f"Análise exploratória e testes (C1-C5): {time.perf_counter() - t0:.1f} s\n")
    pd.set_option("display.width", 200)
    print(tp.groupby("dados")[["dependente_qui2", "dependente_G2"]].sum())
    print(tc.round(4))
    print(ck.round(3))
    print(f"tempo: {time.perf_counter() - t0:.1f} s")
