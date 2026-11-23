"""
Comparação com dados reais: unicidade dos respondentes da pesquisa de 2022
do Stack Overflow com desenvolvedores (Stack Overflow Developer Survey 2022).

Fonte: Stack Overflow, Developer Survey 2022, respostas anonimizadas
(results.csv), https://survey.stackoverflow.co/ , licença Open Database
License (ODbL) 1.0. O arquivo não é redistribuído neste repositório: o script
o baixa na primeira execução, ou lê uma cópia local em dados_reais/.

Ideia: cada país funciona como uma "população" de tamanho conhecido. Os
países com poucos respondentes fazem o papel das turmas pequenas do artigo.
Para cada país e cada desenho de questionário (do mais detalhado ao mais
restritivo), mede-se:
  - a proporção de respondentes únicos na própria amostra do país;
  - a proporção esperada se as respostas fossem independentes entre si
    (produto das distribuições marginais do país), como na maior parte da
    rede geradora do artigo.

Cuidados: só se calculam proporções agregadas. Nenhum registro individual
é examinado, exibido ou vinculado a outra fonte. As perguntas sobre
orientação sexual, identidade trans, saúde mental e deficiência não são
usadas. Respostas em branco contam como uma categoria (a ausência de
resposta também é observável).

Saídas (pasta resultados/):
  R1_unicidade_por_pais.csv   uma linha por país e desenho (sem o nome do país)
  R2_unicidade_por_faixa.csv  médias por faixa de tamanho do país
"""
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("resultados")
LOCAL = Path("dados_reais") / "so2022_results.csv"
URL = "https://github.com/StackExchange/Survey/raw/refs/heads/main/packages/archive/2022/results.csv"
FAIXAS = [15, 31, 101, 501, 5001, 10**6]
ROTULOS = ["15–30", "31–100", "101–500", "501–5.000", "> 5.000"]


def carregar():
    if not LOCAL.exists():
        LOCAL.parent.mkdir(exist_ok=True)
        print("Baixando os dados do Stack Overflow (cerca de 100 MB)...")
        urllib.request.urlretrieve(URL, LOCAL)
    cols = ["Country", "Age", "Gender", "Ethnicity", "EdLevel", "Employment", "DevType", "YearsCodePro"]
    d = pd.read_csv(LOCAL, usecols=cols, dtype=str, low_memory=False)
    d = d[d.Country.notna()].copy()
    return d.fillna("(sem resposta)")


def primeiro(s):
    return s.str.split(";").str[0]


def idade3(a):
    m = {"Under 18 years old": "até 24", "18-24 years old": "até 24", "25-34 years old": "25–44",
         "35-44 years old": "25–44", "45-54 years old": "45 ou mais", "55-64 years old": "45 ou mais",
         "65 years or older": "45 ou mais"}
    return a.map(m).fillna("(sem resposta)")


def genero3(g):
    return np.where(g == "Man", "Homem", np.where(g == "Woman", "Mulher", "Outro/NR"))


def escolaridade3(e):
    pos = e.str.contains("Master|doctoral|Professional degree", regex=True)
    sup = e.str.contains("Bachelor|Associate", regex=True)
    return np.where(pos, "pós", np.where(sup, "superior", "outro/NR"))


def experiencia(y):
    v = pd.to_numeric(y.replace({"Less than 1 year": "0", "More than 50 years": "51"}), errors="coerce")
    return pd.cut(v, [-1, 4, 9, 19, 100], labels=["0–4", "5–9", "10–19", "20+"]).astype(object).fillna("(sem resposta)")


def desenhos(d):
    """Quatro desenhos de questionário, análogos aos D0–D3 do artigo."""
    base = {
        "R0": pd.DataFrame({"idade": d.Age, "genero": d.Gender, "etnia": d.Ethnicity, "escolaridade": d.EdLevel,
                            "vinculo": d.Employment, "funcao": primeiro(d.DevType), "experiencia": d.YearsCodePro}),
        "R1": pd.DataFrame({"idade": d.Age, "genero": d.Gender, "etnia": primeiro(d.Ethnicity),
                            "escolaridade": d.EdLevel, "vinculo": primeiro(d.Employment),
                            "funcao": primeiro(d.DevType), "experiencia": experiencia(d.YearsCodePro)}),
        "R2": pd.DataFrame({"idade": idade3(d.Age), "genero": genero3(d.Gender), "etnia": primeiro(d.Ethnicity),
                            "escolaridade": escolaridade3(d.EdLevel), "vinculo": primeiro(d.Employment),
                            "experiencia": experiencia(d.YearsCodePro)}),
        "R3": pd.DataFrame({"idade": idade3(d.Age), "genero": genero3(d.Gender),
                            "escolaridade": escolaridade3(d.EdLevel), "vinculo": primeiro(d.Employment)}),
    }
    return {k: v.astype(object).fillna("(sem resposta)").astype(str) for k, v in base.items()}


DESCRICAO = {"R0": "detalhado: 7 perguntas, respostas completas",
             "R1": "respostas múltiplas reduzidas à primeira; experiência em faixas",
             "R2": "agregado: idade, gênero e escolaridade em 3 grupos; sem função",
             "R3": "restrito: só idade, gênero, escolaridade e vínculo, agregados"}


def unicidade(q):
    chave = q.agg("|".join, axis=1)
    return float((chave.map(chave.value_counts()) == 1).mean())


def unicidade_independente(q):
    """Proporção esperada de únicos se as perguntas fossem independentes (marginais do próprio grupo)."""
    n = len(q)
    logp = np.zeros(n)
    for c in q.columns:
        logp += np.log(q[c].map(q[c].value_counts(normalize=True)).to_numpy())
    return float(np.mean((1 - np.exp(logp)) ** (n - 1)))


if __name__ == "__main__":
    t0 = time.perf_counter()
    d = carregar()
    tam = d.Country.value_counts()
    paises = tam[tam >= FAIXAS[0]].index
    dz = desenhos(d)
    linhas = []
    for i, p in enumerate(paises):
        m = (d.Country == p).to_numpy()
        for k, q in dz.items():
            qp = q[m]
            linhas.append({"pais_id": i + 1, "N": int(m.sum()), "desenho": k,
                           "unicos_observado": unicidade(qp), "unicos_independencia": unicidade_independente(qp)})
    r1 = pd.DataFrame(linhas)
    r1["faixa"] = pd.cut(r1.N, FAIXAS, labels=ROTULOS, right=False)
    r1.to_csv(OUT / "R1_unicidade_por_pais.csv", index=False)
    r2 = (r1.groupby(["desenho", "faixa"], observed=True)
          .agg(paises=("pais_id", "nunique"), unicos_observado=("unicos_observado", "mean"),
               unicos_independencia=("unicos_independencia", "mean"))
          .reset_index())
    r2.to_csv(OUT / "R2_unicidade_por_faixa.csv", index=False)
    with open(OUT / "ambiente.txt", "a", encoding="utf-8") as fh:
        fh.write(f"Dados reais (R1-R2): {time.perf_counter() - t0:.1f} s\n")
    pd.set_option("display.width", 200)
    print(f"{len(d)} respondentes com país informado; {len(paises)} países com pelo menos {FAIXAS[0]}")
    print(r2.pivot(index="faixa", columns="desenho", values="unicos_observado").round(3))
    print(r2.pivot(index="faixa", columns="desenho", values="unicos_independencia").round(3))
    print(f"tempo: {time.perf_counter() - t0:.1f} s")
