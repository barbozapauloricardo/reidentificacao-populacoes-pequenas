"""
Gera as figuras do artigo no tamanho de uma coluna (layout em duas colunas),
a partir dos CSVs produzidos por sim_reidentificacao.py e sensibilidade.py.
Saída: pasta figuras/ (PDF vetorial e PNG).
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("resultados")
OUT = Path("figuras")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
                     "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})
LARG, ALT = 3.35, 2.5   # polegadas (largura de uma coluna)
CORES = {"D0": "#1f4e79", "D1": "#d9822b", "D2": "#2e8b57", "D3": "#b22222"}
MARC = {"D0": "o", "D1": "s", "D2": "^", "D3": "d"}


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(OUT / f"{nome}.pdf")
    fig.savefig(OUT / f"{nome}.png", dpi=300)
    plt.close(fig)


# Figura 2: únicos x tamanho da turma
e1 = pd.read_csv(RES / "E1_tamanho_turma.csv")
fig, ax = plt.subplots(figsize=(LARG, ALT))
for d in ["D0", "D1", "D2", "D3"]:
    s = e1[e1.desenho == d]
    ax.fill_between(s.N, 100 * s.prop_unicos_p2_5, 100 * s.prop_unicos_p97_5,
                    color=CORES[d], alpha=0.12, lw=0)
    ax.plot(s.N, 100 * s.prop_unicos_MC, marker=MARC[d], ms=3, lw=1.1,
            color=CORES[d], label=d)
ax.set_xscale("log")
ax.set_xlabel("Tamanho da turma (N, escala log)")
ax.set_ylabel("Respondentes únicos (%)")
ax.set_ylim(-2, 102)
ax.grid(alpha=0.3, lw=0.5)
ax.legend(frameon=False, loc="lower left")
salvar(fig, "fig2_unicos_tamanho")

# Figura 3: acumulação de quase-identificadores
e2 = pd.read_csv(RES / "E2_numero_qi.csv")
rot = ["gênero", "+raça/cor", "+idade", "+formação", "+linha", "+vínculo", "+cidade"]
fig, ax = plt.subplots(figsize=(LARG, ALT))
ax.plot(e2.num_QI, 100 * e2.prop_unicos, "o-", ms=3, lw=1.1, color="#1f4e79", label="únicos (k = 1)")
ax.plot(e2.num_QI, 100 * e2.prop_k_menor_5, "s--", ms=3, lw=1.1, color="#d9822b", label="k < 5")
ax.set_xticks(e2.num_QI, rot, rotation=35, ha="right")
ax.set_ylabel("Respondentes (%)")
ax.set_ylim(-2, 104)
ax.grid(alpha=0.3, lw=0.5)
ax.legend(frameon=False, loc="lower right")
salvar(fig, "fig3_numero_qi")

# Figura 4: sensibilidade (N = 30)
e4 = pd.read_csv(RES / "E4_sensibilidade.csv")
piv = e4[e4.N == 30].pivot(index="desenho", columns="cenario", values="prop_unicos") * 100
cen = ["S1 Heterogênea (pessimista)", "S0 Base", "S2 Homogênea", "S3 Muito homogênea (otimista)"]
rot_c = ["S1 heterogênea", "S0 base", "S2 homogênea", "S3 muito homogênea"]
cores_c = ["#7f7f7f", "#1f4e79", "#5b9bd5", "#bdd7ee"]
fig, ax = plt.subplots(figsize=(LARG, ALT))
w = 0.2
for i, (c, r, cor) in enumerate(zip(cen, rot_c, cores_c)):
    ax.bar([k + (i - 1.5) * w for k in range(4)], piv[c].values, w, label=r,
           color=cor, edgecolor="black", lw=0.3)
ax.set_xticks(range(4), piv.index)
ax.set_xlabel("Desenho do questionário")
ax.set_ylabel("Respondentes únicos (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3, lw=0.5)
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
salvar(fig, "fig4_sensibilidade")
print("ok")
