"""
Simulação do risco de reidentificação em questionários "anônimos"
aplicados a turmas pequenas de pós-graduação.

Trabalho final - Big Data, Análise de Dados e Modelos Probabilísticos (Prof. Carlos Maciel)
Autor: Paulo Ricardo Barboza

IMPORTANTE: todos os dados são SINTÉTICOS. As distribuições abaixo são HIPÓTESES
de modelagem (não são estatísticas reais de nenhum programa). Nenhum dado pessoal
é lido, gerado a partir de pessoas reais ou vinculado a pessoas.

Estrutura:
  1. Rede bayesiana geradora (amostragem ancestral) da população sintética.
  2. Quatro desenhos de questionário (D0-D3), tratados como intervenções do(.)
     sobre o instrumento de coleta.
  3. Métricas de risco: proporção de registros únicos (k = 1), risco médio do
     "promotor" E[1/F] e proporção de registros com k < 5.
  4. Solução analítica (enumeração da distribuição conjunta) para validar o
     Monte Carlo.
  5. Experimentos E1 (tamanho da turma), E2 (número de quase-identificadores),
     E3 (resposta incompleta).
"""

import time
import itertools
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 2026
N_REP = 2000                     # repetições Monte Carlo por cenário
TAMANHOS = [10, 15, 20, 30, 50, 100, 500, 5000]
OUT = Path("resultados")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. Rede bayesiana geradora (parâmetros hipotéticos)
#    Raízes: Genero, Raca, Vinculo, Formacao, Cidade
#    Arestas: Vinculo -> Idade ; Formacao -> Linha
#    P(G,R,V,I,F,L,C) = P(G)P(R)P(V)P(I|V)P(F)P(L|F)P(C)
# ---------------------------------------------------------------------------
GENERO = {"F": 0.55, "M": 0.43, "Outro/NI": 0.02}
RACA = {"Branca": 0.60, "Parda": 0.25, "Preta": 0.12, "Amarela": 0.02, "Indigena": 0.01}
VINCULO = {"Bolsista": 0.30, "Trabalha": 0.70}
FORMACAO = {"Direito": 0.30, "Servico Social": 0.15, "Administracao": 0.15,
            "Ciencias Sociais": 0.10, "Historia": 0.10, "Economia": 0.05,
            "Comunicacao": 0.05, "Outra": 0.10}
CIDADE = {"Cidade sede": 0.50, **{f"Cidade {i}": w for i, w in
          zip(range(1, 10), [0.12, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02])}}
LINHAS = ["Linha 1", "Linha 2", "Linha 3"]
# P(Linha | Formacao): hipotético
P_LINHA = {"Direito": [0.50, 0.30, 0.20], "Servico Social": [0.20, 0.60, 0.20],
           "Administracao": [0.30, 0.20, 0.50], "Ciencias Sociais": [0.40, 0.40, 0.20],
           "Historia": [0.40, 0.40, 0.20], "Economia": [0.20, 0.20, 0.60],
           "Comunicacao": [0.35, 0.35, 0.30], "Outra": [1/3, 1/3, 1/3]}
# P(Idade | Vinculo): normal discretizada em 22..65 anos (hipotético)
IDADES = np.arange(22, 66)
PARAM_IDADE = {"Bolsista": (27, 4), "Trabalha": (36, 8)}


def p_idade(vinculo):
    mu, sd = PARAM_IDADE[vinculo]
    w = np.exp(-0.5 * ((IDADES - mu) / sd) ** 2)
    return w / w.sum()


P_IDADE = {v: p_idade(v) for v in VINCULO}


def amostra_categorica(dist, n):
    chaves = list(dist)
    return rng.choice(chaves, size=n, p=list(dist.values()))


def gerar_turma(n):
    """Amostragem ancestral da rede bayesiana: pais antes dos filhos."""
    g = amostra_categorica(GENERO, n)
    r = amostra_categorica(RACA, n)
    v = amostra_categorica(VINCULO, n)
    f = amostra_categorica(FORMACAO, n)
    c = amostra_categorica(CIDADE, n)
    i = np.empty(n, dtype=int)
    for vv in VINCULO:
        m = v == vv
        i[m] = rng.choice(IDADES, size=m.sum(), p=P_IDADE[vv])
    l = np.empty(n, dtype=object)
    for ff in FORMACAO:
        m = f == ff
        l[m] = rng.choice(LINHAS, size=m.sum(), p=P_LINHA[ff])
    return pd.DataFrame({"genero": g, "raca": r, "idade": i, "formacao": f,
                         "linha": l, "vinculo": v, "cidade": c})


# ---------------------------------------------------------------------------
# 2. Desenhos do questionário = intervenções do(.) sobre o instrumento
# ---------------------------------------------------------------------------
def faixa10(idade):
    return np.where(idade < 30, "<30", np.where(idade < 40, "30-39",
                    np.where(idade < 50, "40-49", "50+")))


def formacao_agregada(f):
    return np.where(f == "Direito", "Direito", "Outra CHS")


def cidade_binaria(c):
    return np.where(c == "Cidade sede", "Sede", "Outra")


def aplicar_desenho(df, desenho):
    d = df.copy()
    if desenho == "D0":            # tudo detalhado (7 quase-identificadores)
        cols = ["genero", "raca", "idade", "formacao", "linha", "vinculo", "cidade"]
    elif desenho == "D1":          # do(idade -> faixas de 10 anos)
        d["idade"] = faixa10(d["idade"].to_numpy())
        cols = ["genero", "raca", "idade", "formacao", "linha", "vinculo", "cidade"]
    elif desenho == "D2":          # D1 + do(generalizar formação e cidade)
        d["idade"] = faixa10(d["idade"].to_numpy())
        d["formacao"] = formacao_agregada(d["formacao"].to_numpy())
        d["cidade"] = cidade_binaria(d["cidade"].to_numpy())
        cols = ["genero", "raca", "idade", "formacao", "linha", "vinculo", "cidade"]
    elif desenho == "D3":          # D2 + do(suprimir raça e cidade)
        d["idade"] = faixa10(d["idade"].to_numpy())
        d["formacao"] = formacao_agregada(d["formacao"].to_numpy())
        cols = ["genero", "idade", "formacao", "linha", "vinculo"]
    else:
        raise ValueError(desenho)
    return d[cols]


DESENHOS = {"D0": "Detalhado (7 QI, idade exata)",
            "D1": "Idade em faixas",
            "D2": "Faixas + formação e cidade agregadas",
            "D3": "D2 sem raça e sem cidade"}


# ---------------------------------------------------------------------------
# 3. Métricas de risco
#    F_j = tamanho da classe de equivalência do registro j (registros com a
#    mesma combinação de quase-identificadores). Adversário com conhecimento
#    da composição da turma (colega, coordenação): risco do registro = 1/F_j.
# ---------------------------------------------------------------------------
def metricas(qi):
    tam = qi.groupby(list(qi.columns), sort=False)[qi.columns[0]].transform("size").to_numpy()
    return {"prop_unicos": np.mean(tam == 1),
            "risco_medio": np.mean(1.0 / tam),
            "prop_k_menor_5": np.mean(tam < 5)}


# ---------------------------------------------------------------------------
# 4. Solução analítica
#    Seja p_c a probabilidade da combinação c (distribuição conjunta fatorada).
#    E[proporção de únicos]  = sum_c p_c (1 - p_c)^(N-1)
#    E[risco médio 1/F]      = (1/N) sum_c [1 - (1 - p_c)^N]
#                            = E[nº de combinações distintas observadas] / N
# ---------------------------------------------------------------------------
def distribuicao_conjunta_rapida(desenho):
    """Enumera P(combinação) no espaço do desenho, somando a conjunta original."""
    grade = list(itertools.product(GENERO, RACA, VINCULO, FORMACAO, CIDADE,
                                   range(len(IDADES)), range(len(LINHAS))))
    g, r, v, f, c, ii, ll = map(np.array, zip(*grade))
    p = (np.vectorize(GENERO.get)(g) * np.vectorize(RACA.get)(r) *
         np.vectorize(VINCULO.get)(v) * np.vectorize(FORMACAO.get)(f) *
         np.vectorize(CIDADE.get)(c))
    p_i = np.where(v == "Bolsista", P_IDADE["Bolsista"][ii], P_IDADE["Trabalha"][ii])
    p_l = np.array([P_LINHA[ff][k] for ff, k in zip(f, ll)])
    df = pd.DataFrame({"genero": g, "raca": r, "idade": IDADES[ii], "formacao": f,
                       "linha": np.array(LINHAS)[ll], "vinculo": v, "cidade": c})
    qi = aplicar_desenho(df, desenho)
    qi = qi.assign(_p=p * p_i * p_l)
    return qi.groupby(list(qi.columns[:-1]))["_p"].sum().to_numpy()


def analitico(pc, n):
    return {"prop_unicos": np.sum(pc * (1 - pc) ** (n - 1)),
            "risco_medio": np.sum(1 - (1 - pc) ** n) / n}


# ---------------------------------------------------------------------------
# 5. Experimentos
# ---------------------------------------------------------------------------
def e1_tamanho_turma():
    linhas = []
    for des in DESENHOS:
        pc = distribuicao_conjunta_rapida(des)
        for n in TAMANHOS:
            reps = N_REP if n <= 500 else 200
            m = pd.DataFrame([metricas(aplicar_desenho(gerar_turma(n), des))
                              for _ in range(reps)])
            a = analitico(pc, n)
            linhas.append({"desenho": des, "N": n, "repeticoes": reps,
                           "combinacoes_possiveis": int((pc > 0).sum()),
                           "prop_unicos_MC": m.prop_unicos.mean(),
                           "prop_unicos_p2_5": m.prop_unicos.quantile(0.025),
                           "prop_unicos_p97_5": m.prop_unicos.quantile(0.975),
                           "prop_unicos_analitico": a["prop_unicos"],
                           "risco_medio_MC": m.risco_medio.mean(),
                           "risco_medio_analitico": a["risco_medio"],
                           "prop_k_menor_5_MC": m.prop_k_menor_5.mean()})
    return pd.DataFrame(linhas)


ROTULO = {"genero": "gênero", "raca": "raça/cor", "idade": "idade", "formacao": "formação",
          "linha": "linha", "vinculo": "vínculo", "cidade": "cidade"}
ORDEM_QI = ["genero", "raca", "idade", "formacao", "linha", "vinculo", "cidade"]


def e2_numero_qi(n=30):
    linhas = []
    for q in range(1, len(ORDEM_QI) + 1):
        cols = ORDEM_QI[:q]
        m = pd.DataFrame([metricas(gerar_turma(n)[cols]) for _ in range(N_REP)])
        linhas.append({"N": n, "num_QI": q, "QI_incluidos": "+".join(cols),
                       "prop_unicos": m.prop_unicos.mean(),
                       "risco_medio": m.risco_medio.mean(),
                       "prop_k_menor_5": m.prop_k_menor_5.mean()})
    return pd.DataFrame(linhas)


def e3_resposta_incompleta(n=30, taxa=0.6, desenho="D1"):
    """Nem todos respondem. Entre os respondentes que são únicos NA AMOSTRA,
    quantos são únicos também NA TURMA (onde o adversário os procura)?"""
    reg = []
    for _ in range(N_REP):
        qi = aplicar_desenho(gerar_turma(n), desenho)
        tam_pop = qi.groupby(list(qi.columns))[qi.columns[0]].transform("size").to_numpy()
        resp = rng.random(n) < taxa
        if resp.sum() == 0:
            continue
        amostra = qi[resp]
        tam_am = amostra.groupby(list(amostra.columns))[amostra.columns[0]].transform("size").to_numpy()
        unicos_am = tam_am == 1
        reg.append({"respondentes": resp.sum(),
                    "prop_unicos_na_amostra": unicos_am.mean(),
                    "prop_unicos_na_turma": (tam_pop[resp] == 1).mean(),
                    "precisao_unico_amostra": ((tam_pop[resp] == 1) & unicos_am).sum() / max(unicos_am.sum(), 1),
                    "risco_medio_respondentes": np.mean(1 / tam_pop[resp])})
    r = pd.DataFrame(reg).mean().to_frame().T
    r.insert(0, "desenho", desenho)
    r.insert(0, "taxa_resposta", taxa)
    r.insert(0, "N", n)
    return r


def figuras(e1, e2):
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(8, 6))
    estilos = {"D0": "o-", "D1": "s-", "D2": "^-", "D3": "d-"}
    for des, rot in DESENHOS.items():
        s = e1[e1.desenho == des]
        ax.plot(s.N, 100 * s.prop_unicos_MC, estilos[des], label=f"{des}: {rot}")
        ax.fill_between(s.N, 100 * s.prop_unicos_p2_5, 100 * s.prop_unicos_p97_5, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("Tamanho da turma (N, escala log)")
    ax.set_ylabel("Respondentes únicos (%)")
    ax.set_title("Proporção de registros únicos por desenho do questionário")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.text(0.99, 0.98, "Faixas: 95% das turmas simuladas", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="0.35")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_unicos_por_tamanho.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(e2.num_QI, 100 * e2.prop_unicos, "o-", label="Únicos (k = 1)")
    ax.plot(e2.num_QI, 100 * e2.prop_k_menor_5, "s--", label="k < 5")
    ax.set_xticks(e2.num_QI, [f"+{ROTULO[c]}" if i else ROTULO[c] for i, c in enumerate(ORDEM_QI)],
                  rotation=30)
    ax.set_xlabel("Quase-identificadores acumulados")
    ax.set_ylabel("Respondentes (%)")
    ax.set_title(f"Efeito de acumular quase-identificadores (N = {int(e2.N.iloc[0])})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_unicos_por_num_qi.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.perf_counter()
    e1 = e1_tamanho_turma()
    t1 = time.perf_counter()
    e2 = e2_numero_qi()
    e3 = pd.concat([e3_resposta_incompleta(desenho=d) for d in DESENHOS], ignore_index=True)
    t2 = time.perf_counter()
    e1.to_csv(OUT / "E1_tamanho_turma.csv", index=False)
    e2.to_csv(OUT / "E2_numero_qi.csv", index=False)
    e3.to_csv(OUT / "E3_resposta_incompleta.csv", index=False)
    figuras(e1, e2)
    with open(OUT / "ambiente.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Python {platform.python_version()} | numpy {np.__version__} | "
                 f"pandas {pd.__version__} | matplotlib {matplotlib.__version__}\n"
                 f"SO: {platform.system()} {platform.release()} | CPU: {platform.processor() or platform.machine()}\n"
                 f"Semente: {SEED} | Repetições: {N_REP}\n"
                 f"Tempo E1: {t1 - t0:.1f} s | Tempo E2+E3: {t2 - t1:.1f} s | Total: {t2 - t0:.1f} s\n")
    print(open(OUT / "ambiente.txt", encoding="utf-8").read())
