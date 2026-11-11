"""
Gera exemplos dos dados sintéticos usados no artigo, para inspeção.

Os experimentos não leem estes arquivos: cada script gera seus próprios dados
a partir da rede bayesiana geradora (sim_reidentificacao.py). Estes arquivos
apenas mostram como são os registros simulados.

Saída (pasta dados_sinteticos/):
  turma_N30.csv        uma turma de 30 respondentes, com o tamanho da classe de
                       equivalência de cada registro sob os desenhos D0 a D3
  populacao_N5000.csv  5.000 respondentes do cenário-base
Todos os dados são sintéticos. Semente fixa (2026).
"""
from pathlib import Path
import numpy as np
import sim_reidentificacao as sim

OUT = Path("dados_sinteticos")
OUT.mkdir(exist_ok=True)


def tamanho_classe(df_desenho):
    chave = df_desenho.astype(str).agg("|".join, axis=1)
    return chave.map(chave.value_counts()).to_numpy()


if __name__ == "__main__":
    sim.rng = np.random.default_rng(2026)
    turma = sim.gerar_turma(30)
    turma.insert(0, "id", range(1, 31))
    for d in ["D0", "D1", "D2", "D3"]:
        turma[f"F_{d}"] = tamanho_classe(sim.aplicar_desenho(turma.drop(columns="id"), d))
    turma.to_csv(OUT / "turma_N30.csv", index=False, encoding="utf-8")

    pop = sim.gerar_turma(5000)
    pop.insert(0, "id", range(1, 5001))
    pop.to_csv(OUT / "populacao_N5000.csv", index=False, encoding="utf-8")

    print(turma.to_string(index=False))
    print()
    for d in ["D0", "D1", "D2", "D3"]:
        print(f"{d}: {np.mean(turma[f'F_{d}'] == 1):.1%} de respondentes únicos nesta turma")
