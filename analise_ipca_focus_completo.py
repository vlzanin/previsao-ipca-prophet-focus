# -*- coding: utf-8 -*-
"""
Previsão do IPCA com PROPHET + VARIÁVEIS EXÓGENAS (Dólar e Selic)
Versão otimizada para GOOGLE COLAB.

Principais ganhos de eficiência:
  • Downloads do Banco Central em paralelo (com retry)
  • Grid Search e Backtest paralelizados em todos os núcleos (joblib)
  • Grid Search sem cálculo de intervalos (uncertainty_samples=0) e em poucas janelas
  • Previsão apenas das datas futuras de cada janela (sem recalcular o histórico)
  • Backtest em janelas móveis do horizonte realmente usado (métricas mais confiáveis)
"""

# %%
# ==========================================
# 0. INSTALAÇÃO DAS DEPENDÊNCIAS
# ==========================================
# Instala automaticamente apenas o que estiver faltando.
import importlib.util
import subprocess
import sys

_pacotes = {"prophet": "prophet", "joblib": "joblib"}
_faltantes = [pacote for modulo, pacote in _pacotes.items()
              if importlib.util.find_spec(modulo) is None]
if _faltantes:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *_faltantes])
print("✓ Dependências verificadas!")

# %%
# ==========================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ==========================================
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from joblib import Parallel, delayed
from prophet import Prophet

warnings.filterwarnings("ignore")
for _nome in ("cmdstanpy", "prophet"):          # silencia o log verboso do Stan
    logging.getLogger(_nome).setLevel(logging.CRITICAL)

T_INICIO = time.perf_counter()
print("=" * 60)
print("Bibliotecas importadas com sucesso!")
print("=" * 60)

# %%
# ==========================================
# 2. CONFIGURAÇÕES INICIAIS
# ==========================================
data_inicial_str = "2015-01-01"  # @param {type:"date"}

meses_para_prever = 4   # @param {type:"slider", min:1, max:48, step:1}
meses_para_teste = 32   # @param {type:"slider", min:1, max:60, step:1}
passo_backtest = 2      # @param {type:"slider", min:1, max:12, step:1}

EXECUTAR_GRID_SEARCH = True   # @param {type:"boolean"}
COMPARAR_MODOS = True         # @param {type:"boolean"}
MODO_FINAL = "auto"           # @param ["auto", "univariada", "future_naive", "future_real"]
TRATAR_OUTLIERS = True        # @param {type:"boolean"}

N_ORIGENS_GRID = 4            # nº de janelas usadas no Grid Search (mais = mais lento e mais robusto)
N_JOBS = -1                   # -1 = todos os núcleos
AMOSTRAS_BACKTEST = 300       # amostras p/ intervalo de incerteza no backtest
AMOSTRAS_FINAL = 1000         # amostras p/ intervalo de incerteza na previsão final

# Busca automática das expectativas Focus para Câmbio e Selic.
# A API do Focus informa expectativas anuais; neste código, a mediana anual
# mais recente é repetida nos meses daquele ano. Isso é uma aproximação mensal,
# não uma trajetória mensal publicada pelo Banco Central.
BUSCAR_FOCUS_AUTOMATICAMENTE = True

# Deixe None para consultar automaticamente o Focus.
# Opcionalmente, informe um DataFrame com colunas ['ds', 'dolar', 'selic'].
PROJECOES_COVARIAVEIS = None

assert meses_para_teste >= meses_para_prever, "meses_para_teste deve ser >= meses_para_prever"

print(f"\n📊 CONFIGURAÇÕES:")
print(f"   Data Inicial: {data_inicial_str}")
print(f"   Meses para Prever: {meses_para_prever}")
print(f"   Janela de Teste: {meses_para_teste} meses (passo {passo_backtest})")

# %%
# ==========================================
# 2.1 FUNÇÕES DE CONSULTA AO RELATÓRIO FOCUS
# ==========================================
URL_FOCUS = (
    "https://olinda.bcb.gov.br/olinda/servico/Expectativas/"
    "versao/v1/odata"
)


def buscar_expectativa_anual_focus(indicador, datas_futuras, dias_busca=120):
    """Consulta a mediana anual mais recente do Focus e expande por mês.

    A consulta usa o recurso oficial ExpectativasMercadoAnuais. Como a API
    divulga expectativa por ano de referência, cada valor anual é repetido
    para os meses futuros pertencentes àquele ano.
    """
    entidade = "ExpectativasMercadoAnuais"
    data_corte = (
        pd.Timestamp.today().normalize() - pd.Timedelta(days=dias_busca)
    ).strftime("%Y-%m-%d")
    filtro = f"Indicador eq '{indicador}' and Data ge '{data_corte}'"
    filtro_codificado = quote(filtro, safe="'")
    url = (
        f"{URL_FOCUS}/{entidade}"
        f"?$format=json&$filter={filtro_codificado}&$top=10000"
    )

    resposta = requests.get(url, timeout=30)
    resposta.raise_for_status()
    registros = resposta.json().get("value", [])
    if not registros:
        raise RuntimeError(
            f"Nenhum dado Focus recente encontrado para '{indicador}'. "
            "Verifique a conexão, o nome do indicador ou aumente dias_busca."
        )

    df = pd.DataFrame(registros)
    obrigatorias = {"Data", "DataReferencia", "Mediana"}
    if not obrigatorias.issubset(df.columns):
        raise RuntimeError(
            f"Resposta inesperada da API Focus para '{indicador}'. "
            f"Colunas recebidas: {list(df.columns)}"
        )

    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df["DataReferencia"] = pd.to_numeric(
        df["DataReferencia"], errors="coerce"
    )
    df["Mediana"] = pd.to_numeric(df["Mediana"], errors="coerce")
    df = df.dropna(subset=["Data", "DataReferencia", "Mediana"])
    if df.empty:
        raise RuntimeError(f"Não há medianas válidas no Focus para '{indicador}'.")

    data_mais_recente = df["Data"].max()
    df = df.loc[df["Data"] == data_mais_recente]
    por_ano = df.groupby("DataReferencia")["Mediana"].median().to_dict()

    # Exibe as observações anuais efetivamente retornadas pela API e usadas
    # na construção das covariáveis mensais.
    dados_focus_exibicao = (
        df.groupby("DataReferencia", as_index=False)["Mediana"]
          .median()
          .sort_values("DataReferencia")
    )
    dados_focus_exibicao["Indicador"] = indicador
    dados_focus_exibicao["Data da divulgação"] = data_mais_recente.strftime("%Y-%m-%d")
    dados_focus_exibicao = dados_focus_exibicao.rename(
        columns={"DataReferencia": "Ano de referência", "Mediana": "Mediana Focus"}
    )
    print(f"\nDados anuais lidos da API Focus — {indicador}:")
    print(
        dados_focus_exibicao[
            ["Indicador", "Data da divulgação", "Ano de referência", "Mediana Focus"]
        ].to_string(index=False)
    )

    meses = pd.DatetimeIndex(datas_futuras)
    resultado = pd.Series(
        [por_ano.get(int(ano), np.nan) for ano in meses.year],
        index=meses,
        name=indicador,
        dtype=float,
    )
    if resultado.isna().any():
        anos_sem_expectativa = sorted(
            set(meses[resultado.isna()].year.astype(int).tolist())
        )
        raise RuntimeError(
            f"Focus não trouxe expectativa anual recente para '{indicador}' "
            f"nos anos {anos_sem_expectativa}. O código não preencherá esses "
            "meses com valores inventados."
        )

    print(
        f"✓ Focus — {indicador}: mediana da divulgação de "
        f"{data_mais_recente:%Y-%m-%d}"
    )
    return resultado


def montar_projecoes_focus(datas_futuras):
    """Retorna DataFrame com ds, dolar e selic no formato esperado pelo modelo."""
    dolar = buscar_expectativa_anual_focus("Câmbio", datas_futuras)
    selic = buscar_expectativa_anual_focus("Selic", datas_futuras)
    return pd.DataFrame({
        "ds": pd.DatetimeIndex(datas_futuras),
        "dolar": dolar.to_numpy(),
        "selic": selic.to_numpy(),
    })


# ==========================================
# 3. EXTRAÇÃO DE DADOS (API BANCO CENTRAL) - EM PARALELO
# ==========================================
print("\n" + "=" * 60)
print("ETAPA 1: CARREGAMENTO DOS DADOS (IPCA, DÓLAR, SELIC)")
print("=" * 60)

SERIES_BCB = {"y": 433, "dolar": 3698, "selic": 4390}   # IPCA, Dólar médio mensal, Selic mensal
COLS_COV = ["dolar", "selic"]


def buscar_dados_bcb(codigo_sgs, nome, data_inicial):
    url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_sgs}/dados"
           f"?formato=json&dataInicial={data_inicial:%d/%m/%Y}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for tentativa in range(3):                      # até 3 tentativas
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            break
        time.sleep(2 * (tentativa + 1))
    else:
        raise RuntimeError(f"Erro na requisição ({codigo_sgs}). "
                           f"Código {resp.status_code}: {resp.text[:200]}")
    df = pd.DataFrame(resp.json())
    idx = pd.to_datetime(df["data"], format="%d/%m/%Y").dt.to_period("M").dt.to_timestamp()
    return pd.Series(pd.to_numeric(df["valor"]).to_numpy(), index=idx, name=nome)


data_inicial = pd.to_datetime(data_inicial_str)
with ThreadPoolExecutor(max_workers=3) as ex:
    futuros = {nome: ex.submit(buscar_dados_bcb, cod, nome, data_inicial)
               for nome, cod in SERIES_BCB.items()}
    series = {nome: f.result() for nome, f in futuros.items()}

# Grade mensal contínua: histórico do IPCA + horizonte de previsão
y_obs = series["y"]
grade_y = pd.date_range(y_obs.index.min(), y_obs.index.max(), freq="MS")
n = len(grade_y)
H = meses_para_prever
grade = pd.date_range(grade_y[0], periods=n + H, freq="MS")
datas = grade.to_numpy()

# Obtém projeções Focus para as datas futuras, se não houver projeções manuais.
datas_futuras = pd.DatetimeIndex(grade[n:])
if PROJECOES_COVARIAVEIS is None and BUSCAR_FOCUS_AUTOMATICAMENTE:
    PROJECOES_COVARIAVEIS = montar_projecoes_focus(datas_futuras)

if PROJECOES_COVARIAVEIS is not None:
    PROJECOES_COVARIAVEIS = PROJECOES_COVARIAVEIS.copy()
    colunas_necessarias = {"ds", "dolar", "selic"}
    if not colunas_necessarias.issubset(PROJECOES_COVARIAVEIS.columns):
        raise ValueError(
            "PROJECOES_COVARIAVEIS precisa conter as colunas: "
            "'ds', 'dolar' e 'selic'."
        )
    PROJECOES_COVARIAVEIS["ds"] = pd.to_datetime(
        PROJECOES_COVARIAVEIS["ds"]
    ).dt.to_period("M").dt.to_timestamp()
    print("\nProjeções de covariáveis futuras:")
    print(PROJECOES_COVARIAVEIS.to_string(index=False))

y_raw = y_obs.reindex(grade_y).interpolate(limit_direction="both").to_numpy(np.float64)

# Covariáveis alinhadas à grade. Histórico: forward/back fill (como no original).
# Futuro: valores REAIS se já publicados (Dólar/Selic costumam sair antes do IPCA).
cov_grade = pd.concat([series[c] for c in COLS_COV], axis=1).reindex(grade)
cov_hist = cov_grade.iloc[:n].ffill().bfill()
cov_fut = cov_grade.iloc[n:].copy()
if PROJECOES_COVARIAVEIS is not None:
    proj = PROJECOES_COVARIAVEIS.set_index("ds")[COLS_COV]
    proj.index = pd.to_datetime(proj.index).to_period("M").to_timestamp()
    # Valores já PUBLICADOS têm prioridade; as projeções só preenchem meses ainda sem dado real
    cov_fut = cov_fut.combine_first(proj).reindex(cov_fut.index)[COLS_COV]
COV = np.concatenate([cov_hist.to_numpy(), cov_fut.to_numpy()]).T.astype(np.float64)   # (2, n+H)

if PROJECOES_COVARIAVEIS is not None and not np.isfinite(COV[:, n:]).all():
    raise ValueError(
        "Há valores ausentes de dólar/Selic no horizonte futuro depois de "
        "combinar dados observados e Focus. Verifique a cobertura das projeções."
    )

print(f"\n✓ Dados unificados: {n} registros")
print(f"✓ Período: {grade_y[0]:%Y-%m-%d} até {grade_y[-1]:%Y-%m-%d}")
print(f"✓ Covariáveis já conhecidas após o último IPCA: "
      f"{int(np.isfinite(COV[:, n:]).all(axis=0).sum())} mês(es)")
print("\nÚltimos 5 registros:")
print(pd.DataFrame({"ds": grade_y, "y": y_raw, **dict(zip(COLS_COV, cov_hist.to_numpy().T))}).tail())

# %%
# ==========================================
# 4. TRATAMENTO DE OUTLIERS (APENAS NO IPCA)
# ==========================================
print("\n" + "=" * 60)
print("ETAPA 2: TRATAMENTO DE OUTLIERS (TARGET)")
print("=" * 60)

n_treino = n - meses_para_teste
assert n_treino >= 24, "Poucos dados de treino: reduza meses_para_teste ou antecipe a data inicial."


def limites_iqr(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def marcar_outliers(x, limites):
    """Outliers viram NaN no treino do Prophet (ele ignora esses pontos)."""
    lo, hi = limites
    mask = (x < lo) | (x > hi)
    y = x.copy()
    if TRATAR_OUTLIERS:
        y[mask] = np.nan
    return y, int(mask.sum())


# Backtest: limites calculados SÓ com o período de treino (sem olhar o teste)
y_bt, n_out_bt = marcar_outliers(y_raw, limites_iqr(y_raw[:n_treino]))
# Modelo final: usa toda a série
y_final, n_out = marcar_outliers(y_raw, limites_iqr(y_raw))
y_bt_interp = pd.Series(y_bt).interpolate(limit_direction="both").to_numpy()   # só p/ baselines

print(f"🔍 Outliers no IPCA: {n_out} (série completa) | {n_out_bt} (regra definida só no treino)")
if TRATAR_OUTLIERS:
    print("✓ Outliers marcados como NaN no treino (as métricas usam sempre o valor real)")

# %%
# ==========================================
# 5. FUNÇÕES: MODELO, PREVISÃO POR JANELA E MÉTRICAS
# ==========================================
# Modos de uso das covariáveis (Dólar e Selic):
#   univariada   : só o IPCA (sem regressores)
#   future_naive : regressores com o futuro = último valor conhecido (igual ao seu código na previsão final)
#   future_real  : regressores com o futuro REAL (no backtest é "oráculo", só referência;
#                  na previsão final usa o que já foi publicado + Focus, se informado)
MODOS = ["univariada", "future_naive", "future_real"]


def ffill_tempo(a):
    return pd.DataFrame(a.T).ffill().to_numpy().T


def covariaveis_futuras(modo, COV, o, H):
    """Valores de Dólar/Selic para os H meses após a origem o. Retorna (2, H)."""
    if modo == "future_naive":
        return np.repeat(COV[:, o - 1:o], H, axis=1)
    return ffill_tempo(COV[:, o - 1:o + H])[:, 1:]          # future_real


def criar_modelo(params, modo, amostras):
    m = Prophet(
        daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True,
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
        interval_width=0.9, uncertainty_samples=amostras,    # 0 = sem intervalos (bem mais rápido)
    )
    if modo != "univariada":
        for c in COLS_COV:
            m.add_regressor(c, standardize=True)
    m.add_seasonality(name="trimestral", period=91.25, fourier_order=3)
    return m


def prever_janela(params, modo, o, H, Y, COV, datas, amostras):
    """Treina com os dados até a origem o e prevê SÓ os H meses seguintes.
    Roda em processo separado (joblib). Retorna (yhat, lower, upper) ou None em caso de erro."""
    for nome in ("cmdstanpy", "prophet"):
        logging.getLogger(nome).setLevel(logging.CRITICAL)
    try:
        treino = pd.DataFrame({"ds": datas[:o], "y": Y[:o]})
        futuro = pd.DataFrame({"ds": datas[o:o + H]})
        if modo != "univariada":
            fut = covariaveis_futuras(modo, COV, o, H)
            for j, c in enumerate(COLS_COV):
                treino[c] = COV[j, :o]
                futuro[c] = fut[j]
        m = criar_modelo(params, modo, amostras)
        m.fit(treino)
        p = m.predict(futuro)
        if amostras > 0:
            return p["yhat"].to_numpy(), p["yhat_lower"].to_numpy(), p["yhat_upper"].to_numpy()
        return p["yhat"].to_numpy(), None, None
    except Exception as e:
        return None


def metricas(real, pont, lo, hi, escala):
    erro = real - pont
    mae = np.nanmean(np.abs(erro))
    out = {"MAE": mae,
           "RMSE": np.sqrt(np.nanmean(erro ** 2)),
           "WAPE": np.nansum(np.abs(erro)) / np.sum(np.abs(real)),
           "MASE": mae / escala,
           "Cobertura IC90": np.nan, "Largura média IC": np.nan}
    if lo is not None:
        out["Cobertura IC90"] = np.nanmean((real >= lo) & (real <= hi))
        out["Largura média IC"] = np.nanmean(hi - lo)
    return out

# %%
# ==========================================
# 6. GRID SEARCH PARALELO PARA OTIMIZAÇÃO
# ==========================================
PARAMS_PADRAO = {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 1.0,
                 "seasonality_mode": "additive"}

# Modo usado na otimização: sempre um modo realizável em produção
modo_grid = MODO_FINAL if MODO_FINAL not in ("auto", "future_real") else "future_naive"

if EXECUTAR_GRID_SEARCH:
    print("\n" + "=" * 60)
    print("ETAPA 3: OTIMIZAÇÃO DE HIPERPARÂMETROS (GRID SEARCH PARALELO)")
    print("=" * 60)

    param_grid = {
        "changepoint_prior_scale": [0.01, 0.05, 0.1],
        "seasonality_prior_scale": [0.1, 1.0, 10.0],
        "seasonality_mode": ["additive", "multiplicative"],
    }
    all_params = [dict(zip(param_grid, v)) for v in product(*param_grid.values())]
    origens_grid = np.unique(np.linspace(n_treino, n - H, N_ORIGENS_GRID).round().astype(int))
    tarefas = [(i, o) for i in range(len(all_params)) for o in origens_grid]
    print(f"🔄 {len(all_params)} combinações × {len(origens_grid)} janelas = {len(tarefas)} ajustes "
          f"(modo '{modo_grid}')...")

    t = time.perf_counter()
    saidas = Parallel(n_jobs=N_JOBS)(
        delayed(prever_janela)(all_params[i], modo_grid, o, H, y_bt, COV, datas, 0) for i, o in tarefas)

    erros = {i: [] for i in range(len(all_params))}
    for (i, o), s in zip(tarefas, saidas):
        if s is not None:
            erros[i].append(np.mean(np.abs(y_raw[o:o + H] - s[0])))
    mae_grid = np.array([np.mean(erros[i]) if len(erros[i]) == len(origens_grid) else np.nan
                         for i in range(len(all_params))])
    if np.isnan(mae_grid).all():
        raise RuntimeError("Todos os ajustes do Grid Search falharam. Verifique a instalação do Prophet.")

    df_grid = pd.DataFrame(all_params).assign(MAE=mae_grid).sort_values("MAE")
    melhores_params = all_params[int(np.nanargmin(mae_grid))]
    print(f"✓ Concluído em {time.perf_counter() - t:.1f}s | falhas: {int(np.isnan(mae_grid).sum())}")
    print(f"✓ Melhores hiperparâmetros (MAE: {np.nanmin(mae_grid):.4f}): {melhores_params}")
    print("\nTop 5 combinações:")
    print(df_grid.head(5).round(4).to_string(index=False))
else:
    melhores_params = PARAMS_PADRAO
    print(f"\nGrid Search desativado. Parâmetros padrão: {melhores_params}")

# %%
# ==========================================
# 7. BACKTESTING (JANELAS MÓVEIS, EM PARALELO)
# ==========================================
print("\n" + "=" * 60)
print("ETAPA 4: AVALIAÇÃO DE DESEMPENHO (BACKTESTING)")
print("=" * 60)

origens = np.arange(n_treino, n - H + 1, passo_backtest)   # cada origem = um "corte" de treino
alvos = np.stack([y_raw[o:o + H] for o in origens])        # (W, H) valores reais
escala = np.mean(np.abs(np.diff(y_raw[:n_treino])))        # escala do MASE (naive no treino)
print(f"🔄 {len(origens)} janelas de {H} mês(es) | treino inicial: {n_treino} meses")

modos_bt = MODOS if COMPARAR_MODOS else [MODO_FINAL if MODO_FINAL != "auto" else "future_naive"]

resultados = {}
for modo in modos_bt:
    t = time.perf_counter()
    saidas = Parallel(n_jobs=N_JOBS)(
        delayed(prever_janela)(melhores_params, modo, o, H, y_bt, COV, datas, AMOSTRAS_BACKTEST)
        for o in origens)
    falhas = sum(s is None for s in saidas)
    vazio = np.full(H, np.nan)
    pont, lo, hi = (np.stack([(s if s is not None else (vazio, vazio, vazio))[k] for s in saidas])
                    for k in range(3))
    resultados[modo] = {"pont": pont, "lo": lo, "hi": hi, **metricas(alvos, pont, lo, hi, escala)}
    print(f"   {modo:<13} MAE={resultados[modo]['MAE']:.4f}  "
          f"({time.perf_counter() - t:.1f}s" + (f", {falhas} falhas" if falhas else "") + ")")

# Baselines (sem treino), mesmas janelas
naive = np.stack([np.full(H, y_bt_interp[o - 1]) for o in origens])
sazonal = np.stack([y_bt_interp[o - 12 + (np.arange(H) % 12)] for o in origens])

linhas = {}
for m, r in resultados.items():
    rotulo = f"Prophet · {m}" + (" (oráculo)" if m == "future_real" else "")
    linhas[rotulo] = {k: v for k, v in r.items() if k not in ("pont", "lo", "hi")}
linhas["Baseline · naive"] = metricas(alvos, naive, None, None, escala)
linhas["Baseline · sazonal (12m)"] = metricas(alvos, sazonal, None, None, escala)
tabela = pd.DataFrame(linhas).T.sort_values("MAE")

print("\n📊 COMPARATIVO (menor MAE/MASE = melhor; cobertura ideal ≈ 90%):")
print(tabela.round(4).to_string())
print("\nℹ️ 'oráculo' usa Dólar/Selic futuros reais: serve de teto de referência, não é viável em produção.")

# Escolha do modo final (nunca escolhe o oráculo automaticamente)
if MODO_FINAL != "auto":
    modo_final = MODO_FINAL
else:
    candidatos = [m for m in resultados if m != "future_real"] or list(resultados)
    modo_final = min(candidatos, key=lambda m: resultados[m]["MAE"])
print(f"\n✓ Modo utilizado na previsão final: {modo_final}")

r = resultados[modo_final]
print(f"📊 MÉTRICAS ({modo_final}): MAE: {r['MAE']:.4f} | RMSE: {r['RMSE']:.4f} | "
      f"WAPE: {r['WAPE'] * 100:.1f}% | MASE: {r['MASE']:.2f}")

# %%
# ==========================================
# 8. MODELO FINAL E PREVISÃO FUTURA
# ==========================================
print("\n" + "=" * 60)
print("ETAPA 5: MODELO FINAL E PREVISÃO FUTURA")
print("=" * 60)

treino = pd.DataFrame({"ds": datas[:n], "y": y_final})
futuro = pd.DataFrame({"ds": datas[n:n + H]})
if modo_final != "univariada":
    fut = covariaveis_futuras(modo_final, COV, n, H)
    for j, c in enumerate(COLS_COV):
        treino[c] = COV[j, :n]
        futuro[c] = fut[j]

# -------------------------------------------------------------
# ⚠️ ATENÇÃO: TRATAMENTO DAS VARIÁVEIS EXÓGENAS NO FUTURO ⚠️
# Em 'future_naive' o Dólar/Selic futuros repetem o último valor conhecido;
# em 'future_real' usa-se o que já foi publicado (+ Focus, se informado em
# PROJECOES_COVARIAVEIS). O ideal é sempre usar as projeções do Focus (Bacen).
# -------------------------------------------------------------
modelo_final = criar_modelo(melhores_params, modo_final, AMOSTRAS_FINAL)
modelo_final.fit(treino)

# Previsão em histórico + futuro (necessário para o gráfico de componentes)
previsao_futura = modelo_final.predict(pd.concat([treino.drop(columns="y"), futuro], ignore_index=True))
prev_fut = previsao_futura.tail(H)
print("✓ Previsão realizada com sucesso!")

# %%
# ==========================================
# 9. RESULTADOS E GRÁFICOS
# ==========================================
print("\n" + "=" * 60)
print(f"RESULTADOS: PREVISÃO IPCA PARA {H} MESES")
print("=" * 60)

dados_futuros = prev_fut[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
dados_futuros.columns = ["Data", "IPCA Previsto (%)", "Lim Inferior", "Lim Superior"]
dados_futuros["Data"] = dados_futuros["Data"].dt.strftime("%Y-%m-%d")
print("\n" + dados_futuros.round(4).to_string(index=False))

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(grade_y, y_raw, label="IPCA Real", color="#2E86AB", marker="o", markersize=3, linewidth=2)
ax.plot(grade[origens], r["pont"][:, 0], label="Backtest (1 mês à frente)",
        color="#F77F00", linestyle="--", linewidth=2)
if H > 1:
    ax.plot(grade[origens + H - 1], r["pont"][:, -1], label=f"Backtest ({H} meses à frente)",
            color="#9B5DE5", linestyle=":", linewidth=2)
ax.plot(prev_fut["ds"], prev_fut["yhat"], label=f"Previsão Futura ({H}m)",
        color="#06A77D", linewidth=2.5, marker="o")
ax.fill_between(prev_fut["ds"], prev_fut["yhat_lower"], prev_fut["yhat_upper"],
                color="#06A77D", alpha=0.3, label="Intervalo de 90%")
ax.axvline(x=grade_y[-1], color="red", linestyle=":", linewidth=2, label="Início da Previsão")
ax.set_title(f"Previsão IPCA com Prophet (modo: {modo_final})", fontsize=18, fontweight="bold")
ax.set_ylabel("IPCA (%)", fontsize=12, fontweight="bold")
ax.legend(fontsize=11, loc="upper left")
plt.tight_layout()
plt.show()

# Gráfico de Componentes (inclui o peso do Dólar e da Selic)
modelo_final.plot_components(previsao_futura, figsize=(16, 12))
plt.show()

# Erro por horizonte: Prophet vs baselines
fig, ax = plt.subplots(figsize=(10, 5))
horizontes = np.arange(1, H + 1)
for rotulo, p, estilo in [(f"Prophet · {modo_final}", r["pont"], "-o"),
                          ("Naive", naive, "--s"), ("Sazonal (12m)", sazonal, ":^")]:
    ax.plot(horizontes, np.nanmean(np.abs(alvos - p), axis=0), estilo, label=rotulo, linewidth=2)
ax.set_title("MAE por horizonte de previsão", fontsize=14, fontweight="bold")
ax.set_xlabel("Meses à frente")
ax.set_ylabel("MAE (p.p. do IPCA)")
ax.set_xticks(horizontes)
ax.legend()
plt.tight_layout()
plt.show()

# %%
# ==========================================
# 10. DIAGNÓSTICO DE CONFIANÇA DA PREVISÃO
# ==========================================
print("\n" + "=" * 60)
print("ETAPA 6: DIAGNÓSTICO DE CONFIANÇA DO MODELO")
print("=" * 60)

# O MAPE é instável para o IPCA (valores próximos de zero ou negativos), então a
# regra de bolso compara o erro do Prophet com o dos baselines simples.
mae_prophet = r["MAE"]
mae_base = min(linhas["Baseline · naive"]["MAE"], linhas["Baseline · sazonal (12m)"]["MAE"])
razao = mae_prophet / mae_base
if razao < 0.8:
    qualidade = "EXCELENTE 🌟 (supera claramente os baselines)"
elif razao < 1.0:
    qualidade = "BOA ✅ (supera os baselines por margem pequena)"
elif razao < 1.1:
    qualidade = "RAZOÁVEL ⚠️ (equivalente aos baselines; use com margem de segurança)"
else:
    qualidade = "BAIXA ❌ (pior que um baseline simples; revise dados/covariáveis)"

print(f"\n💡 AVALIAÇÃO DE DESEMPENHO NO BACKTEST ({len(origens)} janelas de {H} meses):")
print(f"   Classificação do Modelo: {qualidade}")
print(f"   • Erro médio de {mae_prophet:.4f} p.p. do IPCA (MAE), {razao:.2f}x o do melhor baseline.")
print(f"   • RMSE (penaliza erros grandes): {r['RMSE']:.4f} | MASE: {r['MASE']:.2f} (<1 = melhor que o naive).")
print(f"   • O intervalo de 90% conteve {r['Cobertura IC90'] * 100:.0f}% dos valores reais (ideal ≈ 90%).")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
real_flat, prev_flat = alvos.ravel(), r["pont"].ravel()
ok = np.isfinite(prev_flat)
real_flat, prev_flat = real_flat[ok], prev_flat[ok]

ax1.scatter(real_flat, prev_flat, color="#06A77D", alpha=0.6, s=40, edgecolor="black")
lo_, hi_ = min(real_flat.min(), prev_flat.min()), max(real_flat.max(), prev_flat.max())
ax1.plot([lo_, hi_], [lo_, hi_], "r--", lw=2.5, label="Linha de Previsão Perfeita")
ax1.set_title("Acurácia: Valores Reais vs. Previstos", fontsize=14, fontweight="bold")
ax1.set_xlabel("IPCA Real (Observado) %", fontsize=12)
ax1.set_ylabel("IPCA Previsto pelo Modelo %", fontsize=12)
ax1.legend(loc="upper left")

sns.histplot(real_flat - prev_flat, kde=True, ax=ax2, color="#E63946", bins=15)
ax2.axvline(x=0, color="black", linestyle="--", lw=2.5, label="Erro Zero (Acerto Exato)")
ax2.set_title("Distribuição dos Erros do Modelo (Resíduos)", fontsize=14, fontweight="bold")
ax2.set_xlabel("Erro (Real - Previsto)", fontsize=12)
ax2.set_ylabel("Frequência de Ocorrência", fontsize=12)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print(f"FIM DO PROCESSAMENTO. Tempo total: {time.perf_counter() - T_INICIO:.1f}s")
print("=" * 60)
