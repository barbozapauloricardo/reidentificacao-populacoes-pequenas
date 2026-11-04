"""
Análise de sensibilidade da simulação de risco de reidentificação.

Pergunta: a conclusão (turmas pequenas permanecem altamente únicas mesmo após
generalização) se mantém quando os parâmetros hipotéticos mudam, inclusive em
cenários deliberadamente FAVORÁVEIS ao anonimato (turmas muito homogêneas)?

Usa a solução analítica (validada contra o Monte Carlo em sim_reidentificacao.py)
e confere um cenário por Monte Carlo.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sim_reidentificacao as sim

OUT = sim.OUT
CURSOS = list(sim.FORMACAO)
CIDADES = list(sim.CIDADE)


def cidade(sede):
    resto = (1 - sede) / 9
    return {c: (sede if i == 0 else resto) for i, c in enumerate(CIDADES)}


def formacao(direito):
    resto = (1 - direito) / 7
    return {c: (direito if c == "Direito" else resto) for c in CURSOS}


def linha(conc):
    r = (1 - conc) / 2
    return {c: [conc, r, r] for c in CURSOS}


BASE = dict(GENERO=dict(sim.GENERO), RACA=dict(sim.RACA), VINCULO=dict(sim.VINCULO),
            FORMACAO=dict(sim.FORMACAO), CIDADE=dict(sim.CIDADE),
            P_LINHA={k: list(v) for k, v in sim.P_LINHA.items()},
            PARAM_IDADE=dict(sim.PARAM_IDADE))

CENARIOS = {
    "S0 Base": {},
    "S1 Heterogênea (pessimista)": dict(
        GENERO={"F": 0.48, "M": 0.48, "Outro/NI": 0.04},
        RACA={"Branca": 0.40, "Parda": 0.30, "Preta": 0.20, "Amarela": 0.05, "Indigena": 0.05},
        VINCULO={"Bolsista": 0.5, "Trabalha": 0.5},
        FORMACAO={c: 1 / 8 for c in CURSOS}, CIDADE={c: 0.1 for c in CIDADES},
        P_LINHA={c: [1 / 3] * 3 for c in CURSOS},
        PARAM_IDADE={"Bolsista": (28, 6), "Trabalha": (38, 10)}),
    "S2 Homogênea": dict(
        GENERO={"F": 0.70, "M": 0.29, "Outro/NI": 0.01},
        RACA={"Branca": 0.80, "Parda": 0.12, "Preta": 0.06, "Amarela": 0.01, "Indigena": 0.01},
        VINCULO={"Bolsista": 0.15, "Trabalha": 0.85},
        FORMACAO=formacao(0.70), CIDADE=cidade(0.80), P_LINHA=linha(0.70),
        PARAM_IDADE={"Bolsista": (26, 2), "Trabalha": (35, 5)}),
    "S3 Muito homogênea (otimista)": dict(
        GENERO={"F": 0.85, "M": 0.14, "Outro/NI": 0.01},
        RACA={"Branca": 0.90, "Parda": 0.06, "Preta": 0.02, "Amarela": 0.01, "Indigena": 0.01},
        VINCULO={"Bolsista": 0.05, "Trabalha": 0.95},
        FORMACAO=formacao(0.90), CIDADE=cidade(0.90), P_LINHA=linha(0.85),
        PARAM_IDADE={"Bolsista": (25, 1.5), "Trabalha": (32, 3)}),
}


def aplicar(cen):
    params = {**BASE, **CENARIOS[cen]}
    for k, v in params.items():
        if k != "PARAM_IDADE":
            setattr(sim, k, v)
    sim.PARAM_IDADE = params["PARAM_IDADE"]
    sim.P_IDADE = {v: sim.p_idade(v) for v in sim.VINCULO}


def entropia_bits(pc):
    pc = pc[pc > 0]
    return float(-(pc * np.log2(pc)).sum())


if __name__ == "__main__":
    t0 = time.perf_counter()
    linhas = []
    for cen in CENARIOS:
        aplicar(cen)
        for des in sim.DESENHOS:
            pc = sim.distribuicao_conjunta_rapida(des)
            for n in [15, 30, 50, 100]:
                a = sim.analitico(pc, n)
                linhas.append({"cenario": cen, "desenho": des, "N": n,
                               "entropia_bits": entropia_bits(pc),
                               "combinacoes_efetivas": 2 ** entropia_bits(pc),
                               "prop_unicos": a["prop_unicos"],
                               "risco_medio": a["risco_medio"]})
    res = pd.DataFrame(linhas)
    res.to_csv(OUT / "E4_sensibilidade.csv", index=False)

    # Conferência Monte Carlo no cenário mais otimista (S3, D3, N = 30)
    aplicar("S3 Muito homogênea (otimista)")
    mc = np.mean([sim.metricas(sim.aplicar_desenho(sim.gerar_turma(30), "D3"))["prop_unicos"]
                  for _ in range(sim.N_REP)])
    an = res.query("cenario.str.startswith('S3') and desenho == 'D3' and N == 30").prop_unicos.iloc[0]
    aplicar("S0 Base")

    # Figura: N = 30, proporção de únicos por cenário e desenho
    piv = res[res.N == 30].pivot(index="cenario", columns="desenho", values="prop_unicos") * 100
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(piv.columns))
    w = 0.2
    for i, (cen, row) in enumerate(piv.iterrows()):
        b = ax.bar(x + (i - 1.5) * w, row.values, w, label=cen)
        ax.bar_label(b, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, [f"{d}\n{sim.DESENHOS[d]}" for d in piv.columns], fontsize=8)
    ax.set_ylabel("Respondentes únicos (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Sensibilidade aos parâmetros hipotéticos (N = 30)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_sensibilidade.png", dpi=200)
    plt.close(fig)

    with open(OUT / "ambiente.txt", "a") as fh:
        fh.write(f"Sensibilidade (E4): {time.perf_counter() - t0:.1f} s | "
                 f"conferência MC S3/D3/N=30: MC={mc:.4f} x analítico={an:.4f}\n")
    pd.set_option("display.width", 200)
    print(res[res.N == 30].pivot(index="cenario", columns="desenho",
                                 values="prop_unicos").round(3))
    print(res[res.N.isin([15, 50, 100])].pivot_table(index=["cenario", "N"], columns="desenho",
                                                      values="prop_unicos").round(3))
    print(res[res.N == 30].pivot(index="cenario", columns="desenho",
                                 values="combinacoes_efetivas").round(1))
    print(f"MC={mc:.4f} analitico={an:.4f}")
