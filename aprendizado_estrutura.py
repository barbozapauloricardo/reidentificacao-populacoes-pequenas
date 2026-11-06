"""
Parte B - Aprendizado de estrutura a partir dos questionários sintéticos.

Pergunta: com quantos respondentes é possível recuperar, a partir dos dados,
a estrutura da rede geradora (Vínculo - Idade e Formação - Linha; demais
variáveis independentes)?

Métodos: algoritmo PC (testes de independência qui-quadrado, alfa = 0,05) e
busca gulosa (Hill Climbing) com pontuação BIC, ambos do pgmpy.
Como a rede geradora não tem colisores, a orientação das arestas não é
identificável por dados observacionais (classe de equivalência de Markov);
por isso a comparação é feita sobre o esqueleto (arestas sem direção).
"""
import time
import warnings
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from pgmpy.estimators import PC, HillClimbSearch, BIC
import sim_reidentificacao as sim

warnings.filterwarnings("ignore")
logging.getLogger("pgmpy").setLevel(logging.ERROR)
OUT = Path("resultados")
REAL = {frozenset(("vinculo", "idade")), frozenset(("formacao", "linha"))}
TAMANHOS = [30, 100, 300, 1000, 5000]
REPS = 20


def preparar(df):
    d = df.copy()
    d["idade"] = sim.faixa10(d["idade"].to_numpy())   # variável discreta para os testes
    return d.astype(str)


def esqueleto(arestas):
    return {frozenset(e) for e in arestas}


def avaliar(est):
    vp = len(est & REAL)
    fp = len(est - REAL)
    fn = len(REAL - est)
    return {"verdadeiros_pos": vp, "falsos_pos": fp, "falsos_neg": fn,
            "SHD_esqueleto": fp + fn, "exato": int(fp + fn == 0)}


if __name__ == "__main__":
    t0 = time.perf_counter()
    linhas = []
    for n in TAMANHOS:
        for r in range(REPS):
            dados = preparar(sim.gerar_turma(n))
            pc = PC(dados).estimate(ci_test="chi_square", significance_level=0.05,
                                    show_progress=False)
            hc = HillClimbSearch(dados).estimate(scoring_method=BIC(dados), show_progress=False)
            for nome, modelo in (("PC", pc), ("HC-BIC", hc)):
                linhas.append({"N": n, "rep": r, "metodo": nome, **avaliar(esqueleto(modelo.edges()))})
    res = pd.DataFrame(linhas)
    res.to_csv(OUT / "B1_estrutura_reps.csv", index=False)
    resumo = (res.groupby(["metodo", "N"])
              .agg(recuperacao_exata=("exato", "mean"),
                   arestas_reais_achadas=("verdadeiros_pos", "mean"),
                   falsos_pos=("falsos_pos", "mean"),
                   SHD_medio=("SHD_esqueleto", "mean"))
              .reset_index())
    resumo.to_csv(OUT / "B1_estrutura_resumo.csv", index=False)
    with open(OUT / "ambiente.txt", "a", encoding="utf-8") as fh:
        fh.write(f"Aprendizado de estrutura (B1): {time.perf_counter() - t0:.1f} s\n")
    pd.set_option("display.width", 200)
    print(resumo.round(3))
    print(f"tempo: {time.perf_counter() - t0:.1f} s")
