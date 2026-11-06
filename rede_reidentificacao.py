"""
Parte A - Rede bayesiana de reidentificação (inferência, intervenção, d-separação e MRE).

Estrutura (DAG):  D -> F <- N,  S -> F,  F -> Re <- K
  D  desenho do questionário (D0..D3)
  N  tamanho da turma (15, 30, 100, 500)
  S  composição da turma (S1 heterogênea, S0 base, S2 homogênea, S3 muito homogênea)
  F  tamanho da classe de equivalência do registro (1, 2, 3, 4, 5+)
  K  adversário (interno: conhece a turma; externo: sem conhecimento auxiliar)
  Re reidentificação correta do registro (não, sim)

As CPTs de F são calculadas analiticamente a partir da rede geradora
(sim_reidentificacao.py / sensibilidade.py):
  P(F = k | D, N, S) = sum_c p_c * Binomial(k - 1; N - 1, p_c)
As demais CPTs são hipóteses explícitas (ver HIPOTESES abaixo).
Todos os dados são sintéticos.
"""
import time
import warnings
from itertools import product, combinations
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import binom
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, CausalInference

import sim_reidentificacao as sim
import sensibilidade as sens

OUT = Path("resultados")
OUT.mkdir(exist_ok=True)

D_ST = ["D0", "D1", "D2", "D3"]
N_ST = [15, 30, 100, 500]
S_ST = ["S1", "S0", "S2", "S3"]
S_NOME = {"S1": "S1 Heterogênea (pessimista)", "S0": "S0 Base",
          "S2": "S2 Homogênea", "S3": "S3 Muito homogênea (otimista)"}
F_ST = ["1", "2", "3", "4", "5+"]
K_ST = ["interno", "externo"]
RE_ST = ["nao", "sim"]

# ---------------- HIPÓTESES (explícitas) ----------------
HIPOTESES = {
    "P(D)": "uniforme (ignorância sobre o desenho escolhido)",
    "P(N)": "uniforme",
    "P(S)": "uniforme",
    "P(K=interno)": 0.5,
    "P(Re=sim | F, interno)": "1/F (5+ -> 1/5, limite superior)",
    "P(Re=sim | F, externo)": "0,10 se F = 1; 0 caso contrário (só registros únicos podem ser vinculados a fontes externas)",
}
P_K_INTERNO = 0.5
P_EXTERNO_UNICO = 0.10
RISCO_F = np.array([1, 1 / 2, 1 / 3, 1 / 4, 1 / 5])


def dist_F(pc, n):
    """Distribuição do tamanho da classe de um registro sorteado, em 5 estados."""
    k = np.arange(1, 5)
    probs = [np.sum(pc * binom.pmf(kk - 1, n - 1, pc)) for kk in k]
    probs.append(max(0.0, 1 - sum(probs)))
    return np.array(probs)


def cpt_F():
    cols = []
    for d, n, s in product(D_ST, N_ST, S_ST):
        sens.aplicar(S_NOME[s])
        pc = sim.distribuicao_conjunta_rapida(d)
        cols.append(dist_F(pc, n))
    sens.aplicar("S0 Base")
    return np.array(cols).T   # linhas: estados de F; colunas: (D, N, S)


def montar_rede(confundida=False):
    arestas = [("D", "F"), ("N", "F"), ("S", "F"), ("F", "Re"), ("K", "Re")]
    if confundida:
        arestas.append(("N", "D"))
    bn = DiscreteBayesianNetwork(arestas)
    sn = {"D": D_ST, "N": [str(x) for x in N_ST], "S": S_ST, "F": F_ST, "K": K_ST, "Re": RE_ST}
    if confundida:
        # cenário ilustrativo: turmas pequenas tendem a usar questionários mais enxutos
        pd_n = np.array([[0.10, 0.20, 0.35, 0.40],   # D0
                         [0.20, 0.25, 0.30, 0.30],   # D1
                         [0.30, 0.30, 0.20, 0.20],   # D2
                         [0.40, 0.25, 0.15, 0.10]])  # D3   colunas: N = 15, 30, 100, 500
        cpd_d = TabularCPD("D", 4, pd_n, evidence=["N"], evidence_card=[4],
                           state_names={"D": D_ST, "N": sn["N"]})
    else:
        cpd_d = TabularCPD("D", 4, [[0.25]] * 4, state_names={"D": D_ST})
    cpd_n = TabularCPD("N", 4, [[0.25]] * 4, state_names={"N": sn["N"]})
    cpd_s = TabularCPD("S", 4, [[0.25]] * 4, state_names={"S": S_ST})
    cpd_k = TabularCPD("K", 2, [[P_K_INTERNO], [1 - P_K_INTERNO]], state_names={"K": K_ST})
    cpd_f = TabularCPD("F", 5, cpt_F(), evidence=["D", "N", "S"], evidence_card=[4, 4, 4],
                       state_names={"F": F_ST, "D": D_ST, "N": sn["N"], "S": S_ST})
    sim_re = []
    for f_i, k in product(range(5), K_ST):
        p = RISCO_F[f_i] if k == "interno" else (P_EXTERNO_UNICO if f_i == 0 else 0.0)
        sim_re.append(p)
    cpd_re = TabularCPD("Re", 2, [[1 - p for p in sim_re], sim_re], evidence=["F", "K"],
                        evidence_card=[5, 2], state_names={"Re": RE_ST, "F": F_ST, "K": K_ST})
    bn.add_cpds(cpd_d, cpd_n, cpd_s, cpd_k, cpd_f, cpd_re)
    assert bn.check_model()
    return bn


def p(q, estado):
    return float(q.get_value(**{q.variables[0]: estado}))


# ---------------- MRE (Yuan, Lim e Lu, 2011), enumeração exaustiva ----------------
def conjunta(bn, alvos, evidencia_fixa):
    """Distribuição conjunta de alvos + Re, condicionada a evidencia_fixa."""
    ve = VariableElimination(bn)
    fat = ve.query(variables=alvos + ["Re"], evidence=evidencia_fixa, joint=True)
    linhas = []
    for idx in product(*[range(c) for c in fat.cardinality]):
        estado = {v: fat.state_names[v][i] for v, i in zip(fat.variables, idx)}
        linhas.append((estado, float(fat.values[idx])))
    return linhas


def prob(linhas, ev):
    return sum(pv for st, pv in linhas if all(st[k] == v for k, v in ev.items()))


def mre(linhas, alvos, evidencia):
    pe = prob(linhas, evidencia)
    estados = {a: sorted({st[a] for st, _ in linhas}) for a in alvos}
    cand = []
    for t in range(1, len(alvos) + 1):
        for nomes in combinations(alvos, t):
            for vals in product(*[estados[n] for n in nomes]):
                x = dict(zip(nomes, vals))
                px = prob(linhas, x)
                if not 0 < px < 1:
                    continue
                pxe = prob(linhas, {**x, **evidencia})
                post = pxe / pe
                alt = (pe - pxe) / (1 - px)
                gbf = (pxe / px) / alt if alt > 0 else math.inf
                cand.append({"explicacao": x, "priori": px, "posteriori": post,
                             "r": post / px, "GBF": gbf})
    cand.sort(key=lambda r: (-r["GBF"], len(r["explicacao"])))

    def domina(a, b):
        A, B = set(a["explicacao"].items()), set(b["explicacao"].items())
        return (A < B and a["GBF"] >= b["GBF"] - 1e-12) or (A > B and a["GBF"] > b["GBF"] + 1e-12)
    minimas = [b for b in cand if not any(domina(a, b) for a in cand if a is not b)]
    return cand, minimas


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    t0 = time.perf_counter()
    bn = montar_rede()
    ve = VariableElimination(bn)
    res = {}

    # A1. Inferência preditiva: P(Re=sim | D, N, K=interno), com média sobre S (composição incerta)
    linhas = []
    for n in N_ST:
        for d in D_ST:
            q = ve.query(["Re"], evidence={"D": d, "N": str(n), "K": "interno"})
            q1 = ve.query(["F"], evidence={"D": d, "N": str(n)})
            linhas.append({"N": n, "D": d, "P_Re_sim": p(q, "sim"), "P_F_1": p(q1, "1")})
    a1 = pd.DataFrame(linhas)
    a1.to_csv(OUT / "A1_preditiva.csv", index=False)

    # A2. Inferência diagnóstica com N = 30 conhecido
    q_d = ve.query(["D"], evidence={"Re": "sim", "N": "30"})
    q_s = ve.query(["S"], evidence={"Re": "sim", "N": "30"})
    q_k = ve.query(["K"], evidence={"Re": "sim", "N": "30"})
    prior_re = p(ve.query(["Re"], evidence={"N": "30"}), "sim")
    a2 = pd.DataFrame(
        [{"variavel": "D", "estado": s, "posteriori": p(q_d, s), "priori": 0.25} for s in D_ST] +
        [{"variavel": "S", "estado": s, "posteriori": p(q_s, s), "priori": 0.25} for s in S_ST] +
        [{"variavel": "K", "estado": s, "posteriori": p(q_k, s),
          "priori": P_K_INTERNO if s == "interno" else 1 - P_K_INTERNO} for s in K_ST])
    a2.to_csv(OUT / "A2_diagnostica.csv", index=False)

    # A3. d-separação, Markov blanket e explicação por exclusão (explaining away)
    dsep = {
        "D ⊥ K": not bn.is_dconnected("D", "K"),
        "D ⊥ K | Re": not bn.is_dconnected("D", "K", observed=["Re"]),
        "D ⊥ Re | F": not bn.is_dconnected("D", "Re", observed=["F"]),
        "N ⊥ S": not bn.is_dconnected("N", "S"),
        "N ⊥ S | F": not bn.is_dconnected("N", "S", observed=["F"]),
    }
    mb_f = sorted(bn.get_markov_blanket("F"))
    ea = {f"P(K=interno | Re=sim, N=30, D={d})":
          p(ve.query(["K"], evidence={"Re": "sim", "N": "30", "D": d}), "interno") for d in D_ST}
    pd.DataFrame([{"afirmacao": k, "verdadeira": v} for k, v in dsep.items()]).to_csv(
        OUT / "A3_dseparacao.csv", index=False)
    pd.DataFrame([{"consulta": k, "valor": v} for k, v in ea.items()]).to_csv(
        OUT / "A3_explaining_away.csv", index=False)

    # A4. Observar x intervir, num cenário ilustrativo com confundimento N -> D
    bnc = montar_rede(confundida=True)
    vec, ci = VariableElimination(bnc), CausalInference(bnc)
    a4 = []
    for d in D_ST:
        obs = p(vec.query(["Re"], evidence={"D": d, "K": "interno"}), "sim")
        # fórmula de ajuste pela porta dos fundos: sum_n P(Re | D=d, N=n, K) P(N=n)
        do = sum(0.25 * p(vec.query(["Re"], evidence={"D": d, "N": str(n), "K": "interno"}), "sim")
                 for n in N_ST)
        a4.append({"D": d, "P_Re_obs": obs, "P_Re_do": do})
    a4 = pd.DataFrame(a4)
    a4.to_csv(OUT / "A4_observar_intervir.csv", index=False)
    ajuste = sorted(ci.get_all_backdoor_adjustment_sets("D", "Re"), key=len)

    # A5. MRE: dado Re = sim numa turma de 30, que configuração melhor explica?
    alvos = ["D", "N", "S", "K"]
    lin = conjunta(bn, alvos, {})
    cand, minimas = mre(lin, alvos, {"Re": "sim"})
    a5 = pd.DataFrame([{**{"explicacao": ", ".join(f"{k}={v}" for k, v in r["explicacao"].items())},
                        **{k: r[k] for k in ("priori", "posteriori", "r", "GBF")}} for r in minimas])
    a5.to_csv(OUT / "A5_mre_minimas.csv", index=False)
    top = pd.DataFrame([{**{"explicacao": ", ".join(f"{k}={v}" for k, v in r["explicacao"].items())},
                         **{k: r[k] for k in ("priori", "posteriori", "r", "GBF")}} for r in cand[:10]])
    top.to_csv(OUT / "A5_mre_top10.csv", index=False)
    # MAP sobre os alvos, para comparação
    lin_e = [(st, pv) for st, pv in lin if st["Re"] == "sim"]
    mapa = max(lin_e, key=lambda t: t[1])

    # MRE com o adversário interno conhecido: alvos D e S
    lin2 = conjunta(bn, ["D", "N", "S"], {"K": "interno"})
    cand2, min2 = mre(lin2, ["D", "N", "S"], {"Re": "sim"})
    a5b = pd.DataFrame([{**{"explicacao": ", ".join(f"{k}={v}" for k, v in r["explicacao"].items())},
                         **{k: r[k] for k in ("priori", "posteriori", "r", "GBF")}} for r in min2])
    a5b.to_csv(OUT / "A5_mre_interno_minimas.csv", index=False)

    pd.set_option("display.width", 200)
    print("A1\n", a1.pivot(index="D", columns="N", values="P_Re_sim").round(3))
    print("P(F=1|D,N)\n", a1.pivot(index="D", columns="N", values="P_F_1").round(3))
    print("prior P(Re=sim|N=30) =", round(prior_re, 4))
    print("A2\n", a2.round(3))
    print("A3", dsep, "MB(F) =", mb_f)
    print(pd.Series(ea).round(3))
    print("A4\n", a4.round(4), "\nconjuntos de ajuste:", ajuste)
    print("A5 top\n", top.round(3))
    print("A5 mínimas\n", a5.round(3))
    print("MAP:", mapa[0], round(mapa[1] / prob(lin, {'Re': 'sim'}), 3))
    print("A5b mínimas (K=interno)\n", a5b.round(3))
    t2 = pd.DataFrame([{"explicacao": ", ".join(f"{k}={v}" for k, v in r["explicacao"].items()),
                        **{k: r[k] for k in ("priori", "posteriori", "r", "GBF")}} for r in cand2[:8]])
    t2.to_csv(OUT / "A5_mre_interno_top.csv", index=False)
    print("A5b top\n", t2.round(3))
    with open(OUT / "ambiente.txt", "a", encoding="utf-8") as fh:
        fh.write(f"Rede de reidentificação (A1-A5): {time.perf_counter() - t0:.1f} s\n")
