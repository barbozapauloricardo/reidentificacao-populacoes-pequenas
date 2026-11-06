"""
Sensibilidade da MRE (E6) às duas hipóteses arbitradas da rede de reidentificação:
P(K = interno) e P(Re = sim | F = 1, externo).
Para cada combinação, refaz a busca da explicação mais relevante dado Re = sim.
"""
import warnings, logging, time
from itertools import product
import pandas as pd
import rede_reidentificacao as rr

warnings.filterwarnings("ignore")
logging.getLogger("pgmpy").setLevel(logging.ERROR)
PK = [0.10, 0.25, 0.50, 0.75, 0.90]
PEXT = [0.05, 0.10, 0.25, 0.50, 1.00]

if __name__ == "__main__":
    t0 = time.perf_counter()
    cpt = rr.cpt_F()                 # calculada uma única vez
    rr.cpt_F = lambda: cpt
    linhas = []
    for pk, pe in product(PK, PEXT):
        rr.P_K_INTERNO, rr.P_EXTERNO_UNICO = pk, pe
        bn = rr.montar_rede()
        lin = rr.conjunta(bn, ["D", "N", "S", "K"], {})
        cand, minimas = rr.mre(lin, ["D", "N", "S", "K"], {"Re": "sim"})
        k_int = next(r for r in cand if r["explicacao"] == {"K": "interno"})
        melhor = cand[0]
        linhas.append({
            "P_K_interno": pk, "P_externo_unico": pe,
            "melhor_explicacao": ", ".join(f"{k}={v}" for k, v in melhor["explicacao"].items()),
            "GBF_melhor": melhor["GBF"],
            "GBF_K_interno": k_int["GBF"],
            "posteriori_K_interno": k_int["posteriori"],
            "minimas": " | ".join(", ".join(f"{k}={v}" for k, v in r["explicacao"].items()) for r in minimas),
        })
    res = pd.DataFrame(linhas)
    res.to_csv("resultados/A6_sensibilidade_mre.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(res.round(3).to_string())
    with open("resultados/ambiente.txt", "a", encoding="utf-8") as fh:
        fh.write(f"Sensibilidade da MRE (A6): {time.perf_counter() - t0:.1f} s\n")
    print(f"tempo: {time.perf_counter() - t0:.1f} s")
