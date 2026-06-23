
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# =========================================================
# =========================================================

COLUNAS_EVENTOS_OBRIGATORIAS = [
    "periodo",
    "tempo",
    "posse_id",
    "equipe",
    "tipo",
    "origem",
    "destino",
    "zona_origem",
    "zona_destino",
    "resultado",
    "classificacao",
    "cenario",
    "sob_pressao",
    "gerou_finalizacao",
    "gerou_gol",
]

RESULTADOS_PASSE_CERTO = {"certo", "finalizacao", "gol"}
RESULTADOS_NEGATIVOS = {"errado", "perda"}
CLASSIFICACOES_OFENSIVAS = {"progressivo", "ruptura", "pivo", "finalizacao"}


@dataclass
class Config:
    eventos_csv: Path
    saida_dir: Path
    tamanho_janela_min: int = 5
    equipe_principal: str = "nosso"
    gerar_graficos: bool = True
    encoding: str = "utf-8-sig"


# =========================================================
# =========================================================

def ler_csv(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    tentativas = [
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "latin1"},
    ]

    melhor_df = None
    melhor_cols = 0
    ultimo_erro = None

    for kwargs in tentativas:
        try:
            df = pd.read_csv(caminho, **kwargs)
            if len(df.columns) > melhor_cols:
                melhor_df = df
                melhor_cols = len(df.columns)
        except Exception as exc:  # pragma: no cover - fallback defensivo
            ultimo_erro = exc

    if melhor_df is None:
        raise ValueError(f"Não foi possível ler o CSV. Último erro: {ultimo_erro}")

    return melhor_df


def validar_colunas(df: pd.DataFrame, colunas_obrigatorias: Iterable[str], nome_base: str) -> None:
    faltantes = [col for col in colunas_obrigatorias if col not in df.columns]
    if faltantes:
        raise ValueError(f"A base '{nome_base}' está sem as colunas obrigatórias: {faltantes}")


def normalizar_texto(valor) -> str:
    if pd.isna(valor):
        return ""
    return str(valor).strip()


def normalizar_minusculo(valor) -> str:
    return normalizar_texto(valor).lower()


def normalizar_booleano(valor) -> bool:
    if isinstance(valor, bool):
        return valor
    if pd.isna(valor):
        return False
    if isinstance(valor, (int, float)):
        return bool(valor)
    return str(valor).strip().lower() in {"true", "1", "sim", "s", "yes", "y", "verdadeiro"}


def tempo_para_segundos(tempo: str) -> int:
    if pd.isna(tempo):
        return 0

    texto = str(tempo).strip()
    if not texto:
        return 0

    if ":" not in texto:
        return int(float(texto) * 60)

    partes = texto.split(":")
    if len(partes) != 2:
        raise ValueError(f"Tempo inválido: {tempo}. Use MM:SS, por exemplo 03:14.")

    minutos = int(partes[0])
    segundos = int(partes[1])
    if segundos < 0 or segundos >= 60:
        raise ValueError(f"Tempo inválido: {tempo}. Segundos devem estar entre 00 e 59.")
    return minutos * 60 + segundos


def janela_temporal(minuto_jogo: float, tamanho_janela_min: int) -> str:
    inicio = int(minuto_jogo // tamanho_janela_min) * tamanho_janela_min
    fim = inicio + tamanho_janela_min
    return f"{inicio:02d}-{fim:02d} min"


def parse_zona(zona: str) -> Tuple[str, str]:
    z = normalizar_texto(zona).upper()
    if not z:
        return "desconhecido", "desconhecido"
    if z == "AREA":
        return "area", "centro"
    partes = z.split("_")
    setor = partes[0].lower() if partes else "desconhecido"
    faixa = partes[1].lower() if len(partes) > 1 else "centro"

    mapa_setor = {
        "DEF": "defesa",
        "MEIO": "meio",
        "ATAQ": "ataque",
        "ATAQUE": "ataque",
    }
    mapa_faixa = {
        "ESQ": "esquerda",
        "DIR": "direita",
        "CENTRO": "centro",
    }
    return mapa_setor.get(setor.upper(), setor), mapa_faixa.get(faixa.upper(), faixa)


def zona_para_numero_profundidade(zona: str) -> int:
    setor, _ = parse_zona(zona)
    return {
        "defesa": 1,
        "meio": 2,
        "ataque": 3,
        "area": 4,
    }.get(setor, 0)


def lado_da_jogada(zona_destino: str, zona_origem: str = "") -> str:
    _, faixa_destino = parse_zona(zona_destino)
    _, faixa_origem = parse_zona(zona_origem)
    if faixa_destino != "desconhecido":
        return faixa_destino
    return faixa_origem


# =========================================================
# =========================================================

def valor_acao_evento(row: pd.Series) -> float:
    tipo = normalizar_minusculo(row.get("tipo", ""))
    resultado = normalizar_minusculo(row.get("resultado", ""))
    classificacao = normalizar_minusculo(row.get("classificacao", ""))
    gerou_finalizacao = normalizar_booleano(row.get("gerou_finalizacao", False))
    gerou_gol = normalizar_booleano(row.get("gerou_gol", False))
    sob_pressao = normalizar_booleano(row.get("sob_pressao", False))
    profundidade_origem = zona_para_numero_profundidade(row.get("zona_origem", ""))
    profundidade_destino = zona_para_numero_profundidade(row.get("zona_destino", ""))

    valor = 0.0

    if tipo == "passe":
        if resultado in RESULTADOS_PASSE_CERTO:
            valor += 1.0
        elif resultado in RESULTADOS_NEGATIVOS:
            valor -= 2.0

        delta_profundidade = profundidade_destino - profundidade_origem
        if delta_profundidade > 0 and resultado in RESULTADOS_PASSE_CERTO:
            valor += 0.5 * delta_profundidade
        elif delta_profundidade < 0 and resultado in RESULTADOS_PASSE_CERTO:
            valor += 0.2

    if tipo == "recuperacao":
        valor += 2.0
    elif tipo == "perda":
        valor -= 3.0
    elif tipo == "finalizacao":
        valor += 4.0

    if classificacao == "seguranca":
        valor += 0.5
    elif classificacao == "progressivo":
        valor += 2.0
    elif classificacao == "ruptura":
        valor += 3.5
    elif classificacao == "pivo":
        valor += 2.5
    elif classificacao == "finalizacao":
        valor += 3.0
    elif classificacao == "perda_perigosa":
        valor -= 4.0
    elif classificacao == "recuperacao":
        valor += 2.0

    if sob_pressao and resultado in RESULTADOS_PASSE_CERTO:
        valor += 0.5
    if sob_pressao and resultado in RESULTADOS_NEGATIVOS:
        valor -= 0.5

    if gerou_finalizacao:
        valor += 2.0

    if gerou_gol or resultado == "gol":
        valor += 6.0

    return round(float(valor), 2)


def classificar_pressao(row: pd.Series) -> str:
    if not normalizar_booleano(row.get("sob_pressao", False)):
        return "sem_pressao"

    setor_origem, _ = parse_zona(row.get("zona_origem", ""))
    if setor_origem == "defesa":
        return "pressao_alta_na_saida"
    if setor_origem == "meio":
        return "pressao_media"
    if setor_origem in {"ataque", "area"}:
        return "pressao_baixa_ou_bloco_baixo"
    return "pressao_nao_classificada"


def classificar_fase_jogo(row: pd.Series) -> str:
    tipo = normalizar_minusculo(row.get("tipo", ""))
    classificacao = normalizar_minusculo(row.get("classificacao", ""))
    resultado = normalizar_minusculo(row.get("resultado", ""))
    prof_ori = zona_para_numero_profundidade(row.get("zona_origem", ""))
    prof_des = zona_para_numero_profundidade(row.get("zona_destino", ""))

    if tipo == "recuperacao":
        return "transicao_ofensiva"
    if tipo == "perda" or classificacao == "perda_perigosa" or resultado == "perda":
        return "transicao_defensiva"
    if tipo == "finalizacao" or classificacao == "finalizacao":
        return "finalizacao"
    if classificacao in {"ruptura", "pivo"}:
        return "criacao"
    if prof_des > prof_ori:
        return "progressao"
    return "construcao"


def resultado_posse(grupo: pd.DataFrame) -> str:
    if bool(grupo["gerou_gol"].any()) or bool((grupo["resultado"] == "gol").any()):
        return "gol"
    if bool((grupo["classificacao"] == "perda_perigosa").any()):
        return "perda_perigosa"
    if bool((grupo["tipo"] == "finalizacao").any()) or bool(grupo["gerou_finalizacao"].any()):
        return "finalizacao"
    if bool((grupo["tipo"] == "perda").any()) or bool((grupo["resultado"] == "perda").any()):
        return "perda"
    return "sem_desfecho"


def padronizar_eventos(eventos: pd.DataFrame, config: Config) -> pd.DataFrame:
    validar_colunas(eventos, COLUNAS_EVENTOS_OBRIGATORIAS, "eventos")
    df = eventos.copy()

    for col in ["equipe", "tipo", "resultado", "classificacao", "cenario"]:
        df[col] = df[col].apply(normalizar_minusculo)

    for col in ["origem", "destino", "zona_origem", "zona_destino", "tempo"]:
        df[col] = df[col].apply(normalizar_texto)

    df["sob_pressao"] = df["sob_pressao"].apply(normalizar_booleano)
    df["gerou_finalizacao"] = df["gerou_finalizacao"].apply(normalizar_booleano)
    df["gerou_gol"] = df["gerou_gol"].apply(normalizar_booleano)
    df["periodo"] = pd.to_numeric(df["periodo"], errors="raise").astype(int)
    df["posse_id"] = pd.to_numeric(df["posse_id"], errors="raise").astype(int)

    df["segundos_periodo"] = df["tempo"].apply(tempo_para_segundos)
    df["segundos_jogo"] = (df["periodo"] - 1) * 20 * 60 + df["segundos_periodo"]
    df["minuto_jogo"] = (df["segundos_jogo"] / 60).round(2)
    df["janela"] = df["minuto_jogo"].apply(lambda m: janela_temporal(m, config.tamanho_janela_min))
    df["chave_posse"] = "P" + df["periodo"].astype(str) + "_" + df["posse_id"].astype(str)

    zonas_origem = df["zona_origem"].apply(parse_zona)
    zonas_destino = df["zona_destino"].apply(parse_zona)
    df["setor_origem"] = [z[0] for z in zonas_origem]
    df["faixa_origem"] = [z[1] for z in zonas_origem]
    df["setor_destino"] = [z[0] for z in zonas_destino]
    df["faixa_destino"] = [z[1] for z in zonas_destino]
    df["lado_ataque"] = df.apply(lambda r: lado_da_jogada(r["zona_destino"], r["zona_origem"]), axis=1)
    df["delta_profundidade"] = df["zona_destino"].apply(zona_para_numero_profundidade) - df["zona_origem"].apply(zona_para_numero_profundidade)

    df["tipo_pressao"] = df.apply(classificar_pressao, axis=1)
    df["fase_jogo"] = df.apply(classificar_fase_jogo, axis=1)
    df["valor_acao"] = df.apply(valor_acao_evento, axis=1)

    df = df.sort_values(["segundos_jogo", "periodo", "posse_id", "tempo"]).reset_index(drop=True)

    evento_gol_nosso = ((df["equipe"] == config.equipe_principal) & ((df["gerou_gol"]) | (df["resultado"] == "gol"))).astype(int)
    evento_gol_adv = ((df["equipe"] != config.equipe_principal) & ((df["gerou_gol"]) | (df["resultado"] == "gol"))).astype(int)

    df["gols_nosso_antes"] = evento_gol_nosso.cumsum().shift(fill_value=0).astype(int)
    df["gols_adv_antes"] = evento_gol_adv.cumsum().shift(fill_value=0).astype(int)
    df["gols_nosso_apos"] = evento_gol_nosso.cumsum().astype(int)
    df["gols_adv_apos"] = evento_gol_adv.cumsum().astype(int)
    df["placar_antes"] = df["gols_nosso_antes"].astype(str) + "x" + df["gols_adv_antes"].astype(str)
    df["placar_apos"] = df["gols_nosso_apos"].astype(str) + "x" + df["gols_adv_apos"].astype(str)
    saldo = df["gols_nosso_antes"] - df["gols_adv_antes"]
    df["estado_placar"] = np.select(
        [saldo > 0, saldo < 0],
        ["vencendo", "perdendo"],
        default="empatando",
    )

    df["ordem_evento_posse"] = df.groupby("chave_posse").cumcount() + 1
    df["numero_passe_na_posse"] = df[df["tipo"] == "passe"].groupby("chave_posse").cumcount() + 1
    df["numero_passe_na_posse"] = df["numero_passe_na_posse"].fillna(0).astype(int)

    posses = df.groupby("chave_posse").agg(
        inicio_posse_seg=("segundos_jogo", "min"),
        fim_posse_seg=("segundos_jogo", "max"),
        eventos_posse=("tipo", "count"),
        passes_tentados_posse=("tipo", lambda s: int((s == "passe").sum())),
        passes_certos_posse=("resultado", lambda s: int(s.isin(RESULTADOS_PASSE_CERTO).sum())),
        valor_total_posse=("valor_acao", "sum"),
        equipe_posse=("equipe", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
    ).reset_index()

    posses["duracao_posse_seg"] = (posses["fim_posse_seg"] - posses["inicio_posse_seg"]).astype(int)
    posses["taxa_acerto_posse"] = np.where(
        posses["passes_tentados_posse"] > 0,
        posses["passes_certos_posse"] / posses["passes_tentados_posse"],
        0,
    ).round(4)

    resultados_posse = pd.DataFrame([
        {"chave_posse": chave, "resultado_posse": resultado_posse(grupo)}
        for chave, grupo in df.groupby("chave_posse")
    ])
    posses = posses.merge(resultados_posse, on="chave_posse", how="left")
    posses["posse_com_finalizacao"] = posses["resultado_posse"].isin(["finalizacao", "gol"])
    posses["posse_com_gol"] = posses["resultado_posse"] == "gol"
    posses["posse_com_perda"] = posses["resultado_posse"].isin(["perda", "perda_perigosa"])

    df = df.merge(posses, on="chave_posse", how="left")

    return df


# =========================================================
# =========================================================

def filtrar_eventos(
    eventos: pd.DataFrame,
    equipe: Optional[str] = None,
    tipo: Optional[str] = None,
    janela: Optional[str] = None,
    cenario: Optional[str] = None,
    classificacao: Optional[str] = None,
    sob_pressao: Optional[bool] = None,
    estado_placar: Optional[str] = None,
    fase_jogo: Optional[str] = None,
) -> pd.DataFrame:
    df = eventos.copy()
    if equipe is not None:
        df = df[df["equipe"] == equipe]
    if tipo is not None:
        df = df[df["tipo"] == tipo]
    if janela is not None:
        df = df[df["janela"] == janela]
    if cenario is not None:
        df = df[df["cenario"] == cenario]
    if classificacao is not None:
        df = df[df["classificacao"] == classificacao]
    if sob_pressao is not None:
        df = df[df["sob_pressao"] == sob_pressao]
    if estado_placar is not None:
        df = df[df["estado_placar"] == estado_placar]
    if fase_jogo is not None:
        df = df[df["fase_jogo"] == fase_jogo]
    return df.copy()


def passes_validos_para_rede(eventos: pd.DataFrame, apenas_certos: bool = True) -> pd.DataFrame:
    passes = eventos[eventos["tipo"] == "passe"].copy()
    passes = passes[(passes["origem"].fillna("") != "") & (passes["destino"].fillna("") != "")]
    if apenas_certos:
        passes = passes[passes["resultado"].isin(RESULTADOS_PASSE_CERTO)]
    return passes


def distancia_aresta(quantidade: float, valor_total: float) -> float:
    forca = quantidade + max(0.0, valor_total) / 3.0
    return 1.0 / max(0.01, forca)


def construir_grafo_passes(
    eventos: pd.DataFrame,
    equipe: str = "nosso",
    apenas_certos: bool = True,
    **filtros,
) -> nx.DiGraph:
    df = filtrar_eventos(eventos, equipe=equipe, tipo="passe", **filtros)
    df = passes_validos_para_rede(df, apenas_certos=apenas_certos)

    G = nx.DiGraph()
    if df.empty:
        return G

    agrupado = (
        df.groupby(["origem", "destino"])
        .agg(
            quantidade_passes=("tipo", "count"),
            valor_total=("valor_acao", "sum"),
            valor_medio=("valor_acao", "mean"),
            passes_progressivos=("classificacao", lambda s: int((s == "progressivo").sum())),
            passes_ruptura=("classificacao", lambda s: int((s == "ruptura").sum())),
            passes_pivo=("classificacao", lambda s: int((s == "pivo").sum())),
            passes_sob_pressao=("sob_pressao", "sum"),
            finalizacoes_geradas=("gerou_finalizacao", "sum"),
            gols_gerados=("gerou_gol", "sum"),
            perda_perigosa=("classificacao", lambda s: int((s == "perda_perigosa").sum())),
        )
        .reset_index()
    )

    for _, row in agrupado.iterrows():
        quantidade = int(row["quantidade_passes"])
        valor_total = float(row["valor_total"])
        G.add_edge(
            row["origem"],
            row["destino"],
            weight=quantidade,
            quantidade_passes=quantidade,
            valor_total=round(valor_total, 2),
            valor_medio=round(float(row["valor_medio"]), 2),
            passes_progressivos=int(row["passes_progressivos"]),
            passes_ruptura=int(row["passes_ruptura"]),
            passes_pivo=int(row["passes_pivo"]),
            passes_sob_pressao=int(row["passes_sob_pressao"]),
            finalizacoes_geradas=int(row["finalizacoes_geradas"]),
            gols_gerados=int(row["gols_gerados"]),
            perda_perigosa=int(row["perda_perigosa"]),
            distancia=distancia_aresta(quantidade, valor_total),
        )

    return G


def construir_grafo_zonas(eventos: pd.DataFrame, equipe: str = "nosso", apenas_certos: bool = False, **filtros) -> nx.DiGraph:
    df = filtrar_eventos(eventos, equipe=equipe, tipo="passe", **filtros)
    df = passes_validos_para_rede(df, apenas_certos=apenas_certos)
    G = nx.DiGraph()
    if df.empty:
        return G

    agrupado = (
        df.groupby(["zona_origem", "zona_destino"])
        .agg(
            tentativas=("tipo", "count"),
            certos=("resultado", lambda s: int(s.isin(RESULTADOS_PASSE_CERTO).sum())),
            valor_total=("valor_acao", "sum"),
            perdas_perigosas=("classificacao", lambda s: int((s == "perda_perigosa").sum())),
            finalizacoes_geradas=("gerou_finalizacao", "sum"),
            gols_gerados=("gerou_gol", "sum"),
        )
        .reset_index()
    )

    for _, row in agrupado.iterrows():
        tentativas = int(row["tentativas"])
        valor_total = float(row["valor_total"])
        G.add_edge(
            row["zona_origem"],
            row["zona_destino"],
            weight=tentativas,
            tentativas=tentativas,
            certos=int(row["certos"]),
            taxa_acerto=round(int(row["certos"]) / tentativas, 4) if tentativas else 0,
            valor_total=round(valor_total, 2),
            perdas_perigosas=int(row["perdas_perigosas"]),
            finalizacoes_geradas=int(row["finalizacoes_geradas"]),
            gols_gerados=int(row["gols_gerados"]),
            distancia=distancia_aresta(tentativas, valor_total),
        )
    return G


# =========================================================
# =========================================================

def safe_pagerank(G: nx.DiGraph) -> Dict[str, float]:
    if len(G.nodes) == 0:
        return {}
    try:
        return nx.pagerank(G, weight="weight")
    except Exception:
        return {n: 0.0 for n in G.nodes}


def safe_betweenness(G: nx.DiGraph) -> Dict[str, float]:
    if len(G.nodes) == 0:
        return {}
    try:
        return nx.betweenness_centrality(G, weight="distancia", normalized=True)
    except Exception:
        return {n: 0.0 for n in G.nodes}


def safe_closeness(G: nx.DiGraph) -> Dict[str, float]:
    if len(G.nodes) == 0:
        return {}
    try:
        return nx.closeness_centrality(G, distance="distancia")
    except Exception:
        return {n: 0.0 for n in G.nodes}


def total_peso(G: nx.Graph) -> float:
    return float(sum(d.get("weight", 0) for _, _, d in G.edges(data=True)))


def total_valor_rede(G: nx.Graph) -> float:
    return float(sum(d.get("valor_total", 0) for _, _, d in G.edges(data=True)))


def centralizacao_grau(G: nx.DiGraph) -> float:
    n = len(G.nodes)
    if n <= 2:
        return 0.0
    graus = dict(G.degree(weight="weight"))
    if not graus:
        return 0.0
    max_grau = max(graus.values())
    soma_diferencas = sum(max_grau - g for g in graus.values())
    denominador = max(1.0, (n - 1) * max(1.0, total_peso(G)))
    return round(float(soma_diferencas / denominador), 4)


def eficiencia_global_segura(G: nx.DiGraph) -> float:
    if len(G.nodes) <= 1:
        return 0.0
    try:
        return round(float(nx.global_efficiency(G.to_undirected())), 4)
    except Exception:
        return 0.0


def metricas_coletivas(G: nx.DiGraph, nome_rede: str = "rede") -> pd.DataFrame:
    if len(G.nodes) == 0:
        return pd.DataFrame([{
            "rede": nome_rede,
            "jogadores_ou_nos": 0,
            "conexoes": 0,
            "total_passes_ou_transicoes": 0,
            "valor_total": 0,
            "densidade": 0,
            "reciprocidade": 0,
            "clustering_medio": 0,
            "centralizacao_grau": 0,
            "componentes_fracas": 0,
            "eficiencia_global": 0,
        }])

    und = G.to_undirected()
    try:
        clustering = nx.average_clustering(und, weight="weight") if len(und.nodes) > 1 else 0
    except Exception:
        clustering = 0

    try:
        reciprocidade = nx.reciprocity(G) if len(G.nodes) > 1 else 0
        if reciprocidade is None:
            reciprocidade = 0
    except Exception:
        reciprocidade = 0

    return pd.DataFrame([{
        "rede": nome_rede,
        "jogadores_ou_nos": len(G.nodes),
        "conexoes": len(G.edges),
        "total_passes_ou_transicoes": round(total_peso(G), 2),
        "valor_total": round(total_valor_rede(G), 2),
        "densidade": round(float(nx.density(G)), 4),
        "reciprocidade": round(float(reciprocidade), 4),
        "clustering_medio": round(float(clustering), 4),
        "centralizacao_grau": centralizacao_grau(G),
        "componentes_fracas": nx.number_weakly_connected_components(G) if isinstance(G, nx.DiGraph) else nx.number_connected_components(G),
        "eficiencia_global": eficiencia_global_segura(G),
    }])


def metricas_jogadores(G: nx.DiGraph) -> pd.DataFrame:
    if len(G.nodes) == 0:
        return pd.DataFrame(columns=[
            "jogador", "passes_realizados", "passes_recebidos", "participacao_total",
            "valor_criado", "valor_recebido", "valor_total_participacao",
            "centralidade_intermediacao", "closeness", "pagerank",
            "dependencia_ofensiva", "indice_importancia"
        ])

    bet = safe_betweenness(G)
    close_bruto = safe_closeness(G)
    max_close = max(close_bruto.values()) if close_bruto else 0
    close = {n: (v / max_close if max_close > 0 else 0.0) for n, v in close_bruto.items()}
    pr = safe_pagerank(G)

    total_participacao_rede = sum(G.out_degree(n, weight="weight") + G.in_degree(n, weight="weight") for n in G.nodes)

    linhas = []
    for jogador in G.nodes:
        passes_realizados = G.out_degree(jogador, weight="weight")
        passes_recebidos = G.in_degree(jogador, weight="weight")
        participacao_total = passes_realizados + passes_recebidos

        valor_criado = sum(G[jogador][dest].get("valor_total", 0) for dest in G.successors(jogador))
        valor_recebido = sum(G[origem][jogador].get("valor_total", 0) for origem in G.predecessors(jogador))
        valor_total_participacao = valor_criado + valor_recebido
        dependencia = participacao_total / total_participacao_rede if total_participacao_rede > 0 else 0

        indice_importancia = (
            0.35 * participacao_total
            + 0.25 * valor_total_participacao
            + 0.20 * bet.get(jogador, 0) * 100
            + 0.10 * close.get(jogador, 0) * 100
            + 0.10 * pr.get(jogador, 0) * 100
        )

        linhas.append({
            "jogador": jogador,
            "passes_realizados": round(float(passes_realizados), 2),
            "passes_recebidos": round(float(passes_recebidos), 2),
            "participacao_total": round(float(participacao_total), 2),
            "valor_criado": round(float(valor_criado), 2),
            "valor_recebido": round(float(valor_recebido), 2),
            "valor_total_participacao": round(float(valor_total_participacao), 2),
            "centralidade_intermediacao": round(float(bet.get(jogador, 0)), 4),
            "closeness": round(float(close.get(jogador, 0)), 4),
            "pagerank": round(float(pr.get(jogador, 0)), 4),
            "dependencia_ofensiva": round(float(dependencia), 4),
            "indice_importancia": round(float(indice_importancia), 2),
        })

    return pd.DataFrame(linhas).sort_values("indice_importancia", ascending=False).reset_index(drop=True)


def detectar_comunidades(G: nx.DiGraph) -> pd.DataFrame:
    if len(G.nodes) == 0:
        return pd.DataFrame(columns=["comunidade", "jogador", "tamanho_comunidade"])

    und = G.to_undirected()
    try:
        comunidades = list(nx.algorithms.community.greedy_modularity_communities(und, weight="weight"))
    except Exception:
        comunidades = [set(G.nodes)]

    linhas = []
    for idx, comunidade in enumerate(comunidades, start=1):
        for jogador in sorted(comunidade):
            linhas.append({
                "comunidade": idx,
                "jogador": jogador,
                "tamanho_comunidade": len(comunidade),
            })
    return pd.DataFrame(linhas)


# =========================================================
# =========================================================

def resumo_basico_eventos(eventos: pd.DataFrame, equipe: str) -> Dict[str, float]:
    df = eventos[eventos["equipe"] == equipe]
    passes = df[df["tipo"] == "passe"]
    passes_certos = passes[passes["resultado"].isin(RESULTADOS_PASSE_CERTO)]
    return {
        "eventos": int(len(df)),
        "passes_tentados": int(len(passes)),
        "passes_certos": int(len(passes_certos)),
        "taxa_acerto_passes": round(len(passes_certos) / len(passes), 4) if len(passes) else 0,
        "passes_progressivos": int((passes["classificacao"] == "progressivo").sum()),
        "passes_ruptura": int((passes["classificacao"] == "ruptura").sum()),
        "passes_pivo": int((passes["classificacao"] == "pivo").sum()),
        "perdas_perigosas": int((df["classificacao"] == "perda_perigosa").sum()),
        "finalizacoes": int((df["tipo"] == "finalizacao").sum()),
        "finalizacoes_geradas": int(df["gerou_finalizacao"].sum()),
        "gols_gerados": int(df["gerou_gol"].sum()),
        "valor_total_eventos": round(float(df["valor_acao"].sum()), 2),
    }


def resumo_temporal(eventos: pd.DataFrame, equipe: str = "nosso") -> pd.DataFrame:
    linhas = []
    for janela in sorted(eventos["janela"].dropna().unique(), key=lambda x: int(x.split("-")[0])):
        eventos_janela = eventos[eventos["janela"] == janela]
        G = construir_grafo_passes(eventos, equipe=equipe, janela=janela)
        mc = metricas_coletivas(G, nome_rede=f"jogadores_{janela}").iloc[0].to_dict()
        ranking = metricas_jogadores(G)
        jogador_chave = ranking.iloc[0]["jogador"] if not ranking.empty else "-"
        dependencia = ranking.iloc[0]["dependencia_ofensiva"] if not ranking.empty else 0

        linha = {
            "janela": janela,
            **resumo_basico_eventos(eventos_janela, equipe),
            "jogador_chave": jogador_chave,
            "dependencia_jogador_chave": dependencia,
            **mc,
        }
        linhas.append(linha)
    return pd.DataFrame(linhas)


def comparar_por_coluna(eventos: pd.DataFrame, coluna: str, equipe: str = "nosso") -> pd.DataFrame:
    linhas = []
    if coluna not in eventos.columns:
        return pd.DataFrame()

    for valor in sorted(eventos[coluna].dropna().unique().tolist(), key=lambda x: str(x)):
        eventos_grupo = eventos[eventos[coluna] == valor]
        if eventos_grupo.empty:
            continue
        filtros = {coluna: valor} if coluna in {"cenario", "estado_placar", "fase_jogo"} else {}

        if coluna == "sob_pressao":
            G = construir_grafo_passes(eventos, equipe=equipe, sob_pressao=bool(valor))
        elif filtros:
            G = construir_grafo_passes(eventos, equipe=equipe, **filtros)
        else:
            # Para colunas derivadas que não entram diretamente em filtrar_eventos.
            G = construir_grafo_passes(eventos_grupo, equipe=equipe)

        ranking = metricas_jogadores(G)
        mc = metricas_coletivas(G, nome_rede=f"{coluna}_{valor}").iloc[0].to_dict()
        linha = {
            "criterio": coluna,
            "valor_criterio": str(valor),
            **resumo_basico_eventos(eventos_grupo, equipe),
            "jogador_chave": ranking.iloc[0]["jogador"] if not ranking.empty else "-",
            "dependencia_jogador_chave": ranking.iloc[0]["dependencia_ofensiva"] if not ranking.empty else 0,
            **mc,
        }
        linhas.append(linha)
    return pd.DataFrame(linhas)


def similaridade_arestas(G1: nx.DiGraph, G2: nx.DiGraph) -> float:
    e1 = set(G1.edges())
    e2 = set(G2.edges())
    if not e1 and not e2:
        return 1.0
    if not e1 or not e2:
        return 0.0
    return round(len(e1 & e2) / len(e1 | e2), 4)


def detectar_mudancas_temporais(eventos: pd.DataFrame, equipe: str = "nosso") -> pd.DataFrame:
    resumo = resumo_temporal(eventos, equipe=equipe)
    if resumo.empty or len(resumo) <= 1:
        return pd.DataFrame(columns=["transicao", "similaridade_arestas", "alerta"])

    linhas = []
    janelas = resumo["janela"].tolist()
    for i in range(1, len(janelas)):
        janela_anterior = janelas[i - 1]
        janela_atual = janelas[i]
        anterior = resumo.iloc[i - 1]
        atual = resumo.iloc[i]
        G_ant = construir_grafo_passes(eventos, equipe=equipe, janela=janela_anterior)
        G_atu = construir_grafo_passes(eventos, equipe=equipe, janela=janela_atual)

        alertas = []
        delta_densidade = atual["densidade"] - anterior["densidade"]
        delta_valor = atual["valor_total"] - anterior["valor_total"]
        delta_centralizacao = atual["centralizacao_grau"] - anterior["centralizacao_grau"]

        if delta_densidade < -0.12:
            alertas.append("queda relevante de densidade")
        if delta_centralizacao > 0.10:
            alertas.append("rede ficou mais centralizada")
        if atual["perdas_perigosas"] > anterior["perdas_perigosas"]:
            alertas.append("aumento de perdas perigosas")
        if atual["dependencia_jogador_chave"] > 0.35:
            alertas.append("dependência alta em um jogador")
        if atual["jogador_chave"] != anterior["jogador_chave"]:
            alertas.append("mudança do jogador central")

        linhas.append({
            "transicao": f"{janela_anterior} -> {janela_atual}",
            "similaridade_arestas": similaridade_arestas(G_ant, G_atu),
            "delta_densidade": round(float(delta_densidade), 4),
            "delta_valor_total_rede": round(float(delta_valor), 2),
            "delta_centralizacao_grau": round(float(delta_centralizacao), 4),
            "jogador_chave_anterior": anterior["jogador_chave"],
            "jogador_chave_atual": atual["jogador_chave"],
            "alerta": "; ".join(alertas) if alertas else "sem alerta relevante",
        })
    return pd.DataFrame(linhas)


# =========================================================
# =========================================================

def cadeia_jogadores_da_posse(grupo: pd.DataFrame) -> List[str]:
    grupo = grupo.sort_values("segundos_jogo")
    jogadores: List[str] = []
    for _, row in grupo.iterrows():
        if row["tipo"] != "passe":
            continue
        origem = normalizar_texto(row.get("origem", ""))
        destino = normalizar_texto(row.get("destino", ""))
        if origem and not jogadores:
            jogadores.append(origem)
        elif origem and jogadores and jogadores[-1] != origem:
            jogadores.append(origem)
        if destino:
            jogadores.append(destino)

    compacta = []
    for j in jogadores:
        if j and (not compacta or compacta[-1] != j):
            compacta.append(j)
    return compacta


def extrair_ngramas(seq: Sequence[str], n: int) -> Iterable[Tuple[str, ...]]:
    if len(seq) < n:
        return []
    return zip(*(islice(seq, i, None) for i in range(n)))


def ranking_caminhos_posse(eventos: pd.DataFrame, equipe: str = "nosso", tamanho_caminho: int = 3) -> pd.DataFrame:
    df = eventos[eventos["equipe"] == equipe].copy()
    linhas = []
    for chave_posse, grupo in df.groupby("chave_posse"):
        seq = cadeia_jogadores_da_posse(grupo)
        if len(seq) < tamanho_caminho:
            continue
        resultado = grupo["resultado_posse"].iloc[0]
        valor_posse = float(grupo["valor_total_posse"].iloc[0])
        for ngrama in extrair_ngramas(seq, tamanho_caminho):
            linhas.append({
                "caminho": " -> ".join(ngrama),
                "chave_posse": chave_posse,
                "resultado_posse": resultado,
                "valor_posse": valor_posse,
                "gerou_finalizacao": resultado in {"finalizacao", "gol"},
                "gerou_gol": resultado == "gol",
                "gerou_perda": resultado in {"perda", "perda_perigosa"},
            })
    if not linhas:
        return pd.DataFrame(columns=["caminho", "quantidade", "valor_total", "finalizacoes", "gols", "perdas"])

    base = pd.DataFrame(linhas)
    resumo = (
        base.groupby("caminho")
        .agg(
            quantidade=("caminho", "count"),
            valor_total=("valor_posse", "sum"),
            finalizacoes=("gerou_finalizacao", "sum"),
            gols=("gerou_gol", "sum"),
            perdas=("gerou_perda", "sum"),
        )
        .reset_index()
    )
    resumo["valor_medio"] = (resumo["valor_total"] / resumo["quantidade"]).round(2)
    resumo["valor_total"] = resumo["valor_total"].round(2)
    return resumo.sort_values(["finalizacoes", "gols", "valor_total", "quantidade"], ascending=False).reset_index(drop=True)


def ranking_duplas(eventos: pd.DataFrame, equipe: str = "nosso") -> pd.DataFrame:
    passes = passes_validos_para_rede(filtrar_eventos(eventos, equipe=equipe, tipo="passe"), apenas_certos=True)
    if passes.empty:
        return pd.DataFrame()
    df = (
        passes.groupby(["origem", "destino"])
        .agg(
            quantidade=("tipo", "count"),
            valor_total=("valor_acao", "sum"),
            valor_medio=("valor_acao", "mean"),
            finalizacoes_geradas=("gerou_finalizacao", "sum"),
            gols_gerados=("gerou_gol", "sum"),
            sob_pressao=("sob_pressao", "sum"),
        )
        .reset_index()
    )
    df["valor_total"] = df["valor_total"].round(2)
    df["valor_medio"] = df["valor_medio"].round(2)
    return df.sort_values(["valor_total", "finalizacoes_geradas", "quantidade"], ascending=False).reset_index(drop=True)


def ranking_zonas(eventos: pd.DataFrame, equipe: str = "nosso") -> pd.DataFrame:
    passes = filtrar_eventos(eventos, equipe=equipe, tipo="passe")
    if passes.empty:
        return pd.DataFrame()
    df = (
        passes.groupby(["zona_origem", "zona_destino"])
        .agg(
            tentativas=("tipo", "count"),
            certos=("resultado", lambda s: int(s.isin(RESULTADOS_PASSE_CERTO).sum())),
            valor_total=("valor_acao", "sum"),
            finalizacoes_geradas=("gerou_finalizacao", "sum"),
            gols_gerados=("gerou_gol", "sum"),
            perdas_perigosas=("classificacao", lambda s: int((s == "perda_perigosa").sum())),
        )
        .reset_index()
    )
    df["taxa_acerto"] = np.where(df["tentativas"] > 0, df["certos"] / df["tentativas"], 0).round(4)
    df["valor_medio"] = (df["valor_total"] / df["tentativas"]).round(2)
    df["valor_total"] = df["valor_total"].round(2)
    return df.sort_values(["valor_total", "finalizacoes_geradas", "tentativas"], ascending=False).reset_index(drop=True)


# =========================================================
# =========================================================

def inferir_pivos(eventos: pd.DataFrame, equipe: str = "nosso") -> List[str]:
    df = eventos[(eventos["equipe"] == equipe) & (eventos["classificacao"] == "pivo")]
    candidatos = []
    candidatos.extend(df["destino"].dropna().astype(str).tolist())
    candidatos.extend(df["origem"].dropna().astype(str).tolist())
    candidatos = [c for c in candidatos if c.strip()]
    if not candidatos:
        return []
    contagem = pd.Series(candidatos).value_counts()
    return contagem.head(3).index.tolist()


def acesso_aos_pivos(G: nx.DiGraph, pivos: Sequence[str]) -> float:
    total = 0.0
    for pivo in pivos:
        if pivo in G.nodes:
            total += G.in_degree(pivo, weight="weight")
            total += G.out_degree(pivo, weight="weight")
    return float(total)


def analisar_robustez_jogadores(G: nx.DiGraph, pivos: Sequence[str]) -> pd.DataFrame:
    if len(G.nodes) == 0:
        return pd.DataFrame()

    densidade_base = nx.density(G)
    valor_base = total_valor_rede(G)
    passes_base = total_peso(G)
    eficiencia_base = eficiencia_global_segura(G)
    acesso_pivo_base = acesso_aos_pivos(G, pivos)

    linhas = []
    for jogador in G.nodes:
        H = G.copy()
        H.remove_node(jogador)
        valor_novo = total_valor_rede(H)
        passes_novo = total_peso(H)
        densidade_nova = nx.density(H) if len(H.nodes) > 1 else 0
        eficiencia_nova = eficiencia_global_segura(H)
        acesso_pivo_novo = acesso_aos_pivos(H, pivos)

        linhas.append({
            "jogador_removido": jogador,
            "interpretacao": "simula neutralização/marcação forte desse vértice",
            "queda_passes_%": round((passes_base - passes_novo) / passes_base * 100, 2) if passes_base else 0,
            "queda_valor_%": round((valor_base - valor_novo) / valor_base * 100, 2) if valor_base else 0,
            "queda_densidade": round(float(densidade_base - densidade_nova), 4),
            "queda_eficiencia_global": round(float(eficiencia_base - eficiencia_nova), 4),
            "queda_acesso_pivo_%": round((acesso_pivo_base - acesso_pivo_novo) / acesso_pivo_base * 100, 2) if acesso_pivo_base else 0,
            "componentes_fracas_restantes": nx.number_weakly_connected_components(H) if len(H.nodes) else 0,
        })
    return pd.DataFrame(linhas).sort_values(["queda_valor_%", "queda_passes_%"], ascending=False).reset_index(drop=True)


def analisar_robustez_arestas(G: nx.DiGraph, top_n: int = 15) -> pd.DataFrame:
    if len(G.edges) == 0:
        return pd.DataFrame()

    valor_base = total_valor_rede(G)
    passes_base = total_peso(G)
    eficiencia_base = eficiencia_global_segura(G)

    arestas_ordenadas = sorted(G.edges(data=True), key=lambda e: (e[2].get("valor_total", 0), e[2].get("weight", 0)), reverse=True)
    linhas = []
    for origem, destino, dados in arestas_ordenadas[:top_n]:
        H = G.copy()
        H.remove_edge(origem, destino)
        valor_novo = total_valor_rede(H)
        passes_novo = total_peso(H)
        eficiencia_nova = eficiencia_global_segura(H)
        linhas.append({
            "aresta_removida": f"{origem} -> {destino}",
            "interpretacao": "simula adversário fechando essa linha de passe",
            "passes_da_aresta": dados.get("weight", 0),
            "valor_da_aresta": dados.get("valor_total", 0),
            "queda_passes_%": round((passes_base - passes_novo) / passes_base * 100, 2) if passes_base else 0,
            "queda_valor_%": round((valor_base - valor_novo) / valor_base * 100, 2) if valor_base else 0,
            "queda_eficiencia_global": round(float(eficiencia_base - eficiencia_nova), 4),
        })
    return pd.DataFrame(linhas).sort_values(["queda_valor_%", "queda_passes_%"], ascending=False).reset_index(drop=True)


# =========================================================
# =========================================================

FONTE_FLUXO = "SAIDA_DE_BOLA"
SUMIDOURO_FLUXO = "FINALIZACAO"


def capacidade_fluxo(tentativas: float, certos: float, valor_total: float, finalizacoes: float = 0, gols: float = 0) -> float:
    valor_positivo = max(0.0, float(valor_total))
    cap = float(certos) + 0.35 * float(tentativas) + valor_positivo / 3.0 + 2.0 * float(finalizacoes) + 5.0 * float(gols)
    return round(max(0.0, cap), 2)


def construir_rede_fluxo_zonas(eventos: pd.DataFrame, equipe: str = "nosso") -> nx.DiGraph:
    df = eventos[eventos["equipe"] == equipe].copy()
    G = nx.DiGraph()
    G.add_node(FONTE_FLUXO, tipo="fonte")
    G.add_node(SUMIDOURO_FLUXO, tipo="sumidouro")

    passes = df[(df["tipo"] == "passe") & (df["origem"].fillna("") != "") & (df["destino"].fillna("") != "")].copy()
    if passes.empty:
        return G

    transicoes = (
        passes.groupby(["zona_origem", "zona_destino"])
        .agg(
            tentativas=("tipo", "count"),
            certos=("resultado", lambda s: int(s.isin(RESULTADOS_PASSE_CERTO).sum())),
            valor_total=("valor_acao", "sum"),
            finalizacoes_geradas=("gerou_finalizacao", "sum"),
            gols_gerados=("gerou_gol", "sum"),
            perdas_perigosas=("classificacao", lambda s: int((s == "perda_perigosa").sum())),
        )
        .reset_index()
    )

    for _, row in transicoes.iterrows():
        origem = row["zona_origem"]
        destino = row["zona_destino"]
        cap = capacidade_fluxo(
            tentativas=row["tentativas"],
            certos=row["certos"],
            valor_total=row["valor_total"],
            finalizacoes=row["finalizacoes_geradas"],
            gols=row["gols_gerados"],
        )
        if cap <= 0:
            continue
        G.add_edge(
            origem,
            destino,
            capacity=cap,
            weight=cap,
            tentativas=int(row["tentativas"]),
            certos=int(row["certos"]),
            valor_total=round(float(row["valor_total"]), 2),
            finalizacoes_geradas=int(row["finalizacoes_geradas"]),
            gols_gerados=int(row["gols_gerados"]),
            perdas_perigosas=int(row["perdas_perigosas"]),
            tipo_aresta="transicao_zona",
        )

    primeiras_acoes = df.sort_values("segundos_jogo").groupby("chave_posse").head(1)
    inicio_posses = (
        primeiras_acoes.groupby("zona_origem")
        .agg(
            posses_iniciadas=("chave_posse", "count"),
            valor_total=("valor_acao", "sum"),
        )
        .reset_index()
    )
    capacidade_total_interna = max(1.0, sum(float(d.get("capacity", 0)) for _, _, d in G.edges(data=True)))
    capacidade_artificial_alta = round(capacidade_total_interna * 10, 2)
    for _, row in inicio_posses.iterrows():
        zona = row["zona_origem"]
        if not zona:
            continue
        G.add_edge(
            FONTE_FLUXO,
            zona,
            capacity=capacidade_artificial_alta,
            weight=capacidade_artificial_alta,
            posses_iniciadas=int(row["posses_iniciadas"]),
            tipo_aresta="entrada_fonte_alta",
        )
    zonas_candidatas = set()
    for zona in list(passes["zona_origem"].dropna().astype(str)) + list(passes["zona_destino"].dropna().astype(str)):
        setor, _ = parse_zona(zona)
        if setor in {"ataque", "area"}:
            zonas_candidatas.add(zona)

    finalizacoes_por_zona = (
        df[df["tipo"].isin(["finalizacao", "passe"])]
        .groupby("zona_origem")
        .agg(
            eventos=("tipo", "count"),
            finalizacoes=("tipo", lambda s: int((s == "finalizacao").sum())),
            finalizacoes_geradas=("gerou_finalizacao", "sum"),
            gols=("gerou_gol", "sum"),
            valor_total=("valor_acao", "sum"),
        )
        .reset_index()
    )
    mapa_final = {row["zona_origem"]: row for _, row in finalizacoes_por_zona.iterrows()}

    for zona in zonas_candidatas:
        row = mapa_final.get(zona)
        if row is not None:
            cap = capacidade_fluxo(
                tentativas=row["eventos"],
                certos=row["eventos"],
                valor_total=row["valor_total"],
                finalizacoes=row["finalizacoes"] + row["finalizacoes_geradas"],
                gols=row["gols"],
            )
        else:
            cap = 1.0
        cap = round(max(1.0, capacidade_artificial_alta), 2)
        G.add_edge(zona, SUMIDOURO_FLUXO, capacity=cap, weight=cap, tipo_aresta="saida_sumidouro_alta")

    return G


def analisar_fluxo_maximo_corte_minimo(G_fluxo: nx.DiGraph) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    if FONTE_FLUXO not in G_fluxo or SUMIDOURO_FLUXO not in G_fluxo:
        vazio = pd.DataFrame([{"fluxo_maximo": 0, "valor_corte_minimo": 0, "interpretacao": "rede sem fonte ou sumidouro"}])
        return vazio, pd.DataFrame(), {}

    try:
        fluxo_max, fluxo_dict = nx.maximum_flow(G_fluxo, FONTE_FLUXO, SUMIDOURO_FLUXO, capacity="capacity")
        valor_corte, particao = nx.minimum_cut(G_fluxo, FONTE_FLUXO, SUMIDOURO_FLUXO, capacity="capacity")
    except nx.NetworkXError as exc:
        vazio = pd.DataFrame([{"fluxo_maximo": 0, "valor_corte_minimo": 0, "interpretacao": f"erro ao calcular fluxo/corte: {exc}"}])
        return vazio, pd.DataFrame(), {}

    alcancaveis, nao_alcancaveis = particao
    arestas_corte = []
    for u in alcancaveis:
        for v in G_fluxo.successors(u):
            if v in nao_alcancaveis:
                dados = G_fluxo[u][v]
                arestas_corte.append({
                    "origem": u,
                    "destino": v,
                    "capacidade": round(float(dados.get("capacity", 0)), 2),
                    "fluxo_usado": round(float(fluxo_dict.get(u, {}).get(v, 0)), 2),
                    "tipo_aresta": dados.get("tipo_aresta", ""),
                    "interpretacao": "gargalo: se essa conexão for bloqueada, reduz a capacidade defesa->finalização",
                })

    resumo = pd.DataFrame([{
        "fonte": FONTE_FLUXO,
        "sumidouro": SUMIDOURO_FLUXO,
        "fluxo_maximo": round(float(fluxo_max), 2),
        "valor_corte_minimo": round(float(valor_corte), 2),
        "qtd_arestas_corte_minimo": len(arestas_corte),
        "zonas_lado_fonte": ", ".join(sorted(str(n) for n in alcancaveis if n not in {FONTE_FLUXO, SUMIDOURO_FLUXO})),
        "zonas_lado_sumidouro": ", ".join(sorted(str(n) for n in nao_alcancaveis if n not in {FONTE_FLUXO, SUMIDOURO_FLUXO})),
        "interpretacao": "fluxo máximo mede a capacidade observada de progressão até finalização; corte mínimo aponta os gargalos da rede",
    }])

    return resumo, pd.DataFrame(arestas_corte).sort_values("capacidade", ascending=False).reset_index(drop=True), fluxo_dict


def tabela_fluxo_arestas(G_fluxo: nx.DiGraph, fluxo_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    linhas = []
    for u, v, dados in G_fluxo.edges(data=True):
        capacidade = float(dados.get("capacity", 0))
        fluxo = float(fluxo_dict.get(u, {}).get(v, 0)) if fluxo_dict else 0.0
        linhas.append({
            "origem": u,
            "destino": v,
            "capacidade": round(capacidade, 2),
            "fluxo_usado": round(fluxo, 2),
            "saturacao_%": round((fluxo / capacidade) * 100, 2) if capacidade else 0,
            "tipo_aresta": dados.get("tipo_aresta", ""),
        })
    return pd.DataFrame(linhas).sort_values(["saturacao_%", "capacidade"], ascending=False).reset_index(drop=True)


def salvar_grafo_fluxo(G_fluxo: nx.DiGraph, fluxo_dict: Dict[str, Dict[str, float]], arestas_corte: pd.DataFrame, caminho: Path) -> None:
    if len(G_fluxo.nodes) == 0:
        return

    pos_base = {
        FONTE_FLUXO: (0, 0),
        "DEF_ESQ": (1.8, 1.6),
        "DEF_CENTRO": (1.8, 0),
        "DEF_DIR": (1.8, -1.6),
        "MEIO_ESQ": (4.0, 1.6),
        "MEIO_CENTRO": (4.0, 0),
        "MEIO_DIR": (4.0, -1.6),
        "ATAQ_ESQ": (6.2, 1.6),
        "ATAQ_CENTRO": (6.2, 0),
        "ATAQ_DIR": (6.2, -1.6),
        "AREA": (8.0, 0),
        SUMIDOURO_FLUXO: (9.6, 0),
    }
    pos = {}
    faltantes = []
    for n in G_fluxo.nodes:
        if n in pos_base:
            pos[n] = pos_base[n]
        else:
            faltantes.append(n)
    for i, n in enumerate(faltantes):
        pos[n] = (4.5, 2.4 - i * 0.5)

    corte_set = set()
    if arestas_corte is not None and not arestas_corte.empty:
        corte_set = set(zip(arestas_corte["origem"], arestas_corte["destino"]))

    fig, ax = plt.subplots(figsize=(14, 7))
    edge_widths = []
    edge_colors = []
    for u, v in G_fluxo.edges:
        cap = G_fluxo[u][v].get("capacity", 1)
        edge_widths.append(max(1.0, min(6.0, cap / 3)))
        edge_colors.append("crimson" if (u, v) in corte_set else "gray")

    node_sizes = [2200 if n in {FONTE_FLUXO, SUMIDOURO_FLUXO} else 1600 for n in G_fluxo.nodes]
    nx.draw_networkx_nodes(G_fluxo, pos, node_size=node_sizes, edgecolors="black", linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(G_fluxo, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G_fluxo, pos, arrows=True, arrowstyle="-|>", arrowsize=16, width=edge_widths, edge_color=edge_colors, alpha=0.75, ax=ax, connectionstyle="arc3,rad=0.04")
    labels = {}
    for u, v, d in G_fluxo.edges(data=True):
        f = fluxo_dict.get(u, {}).get(v, 0) if fluxo_dict else 0
        labels[(u, v)] = f"{round(float(f),1)}/{round(float(d.get('capacity',0)),1)}"
    nx.draw_networkx_edge_labels(G_fluxo, pos, edge_labels=labels, font_size=7, ax=ax)
    ax.set_title("Fluxo máximo e corte mínimo: saída de bola -> finalização", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(caminho, dpi=180)
    plt.close(fig)


# =========================================================
# =========================================================

def posicoes_rede_futsal(G: nx.DiGraph, eventos: pd.DataFrame, equipe: str = "nosso") -> Dict[str, Tuple[float, float]]:
    mapa_x = {"defesa": 2.0, "meio": 5.0, "ataque": 8.0, "area": 9.0, "desconhecido": 5.0}
    mapa_y = {"esquerda": 1.8, "centro": 0.0, "direita": -1.8, "desconhecido": 0.0}
    pos = {}
    eventos_eq = eventos[eventos["equipe"] == equipe]

    for jogador in G.nodes:
        linhas = eventos_eq[(eventos_eq["origem"] == jogador) | (eventos_eq["destino"] == jogador)]
        pontos = []
        for _, row in linhas.iterrows():
            if row["origem"] == jogador:
                setor, faixa = parse_zona(row["zona_origem"])
            else:
                setor, faixa = parse_zona(row["zona_destino"])
            pontos.append((mapa_x.get(setor, 5.0), mapa_y.get(faixa, 0.0)))
        if pontos:
            pos[jogador] = (float(np.mean([p[0] for p in pontos])), float(np.mean([p[1] for p in pontos])))
        else:
            pos[jogador] = (5.0, 0.0)

    usados = {}
    for i, jogador in enumerate(list(pos)):
        p = (round(pos[jogador][0], 1), round(pos[jogador][1], 1))
        usados[p] = usados.get(p, 0) + 1
        if usados[p] > 1:
            pos[jogador] = (pos[jogador][0], pos[jogador][1] + 0.35 * usados[p])
    return pos


def desenhar_quadra(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(-4.2, 4.2)
    ax.add_patch(plt.Rectangle((0.5, -3.5), 9, 7, fill=False, linewidth=2))
    ax.plot([5, 5], [-3.5, 3.5], linewidth=1)
    ax.add_patch(plt.Circle((5, 0), 0.7, fill=False, linewidth=1))
    ax.add_patch(plt.Rectangle((0.5, -1.5), 1.2, 3, fill=False, linewidth=1))
    ax.add_patch(plt.Rectangle((8.3, -1.5), 1.2, 3, fill=False, linewidth=1))
    for x in [3.5, 6.5]:
        ax.plot([x, x], [-3.5, 3.5], linestyle="--", linewidth=0.8, alpha=0.5)
    for y in [-1.2, 1.2]:
        ax.plot([0.5, 9.5], [y, y], linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(2.0, 3.75, "DEF", ha="center", fontsize=9)
    ax.text(5.0, 3.75, "MEIO", ha="center", fontsize=9)
    ax.text(8.0, 3.75, "ATAQUE", ha="center", fontsize=9)


def salvar_grafo_passes(G: nx.DiGraph, eventos: pd.DataFrame, caminho: Path, titulo: str, equipe: str = "nosso") -> None:
    if len(G.nodes) == 0:
        return
    pos = posicoes_rede_futsal(G, eventos, equipe=equipe)
    metricas = metricas_jogadores(G)
    tamanho = {row["jogador"]: 900 + row["participacao_total"] * 120 for _, row in metricas.iterrows()}

    fig, ax = plt.subplots(figsize=(13, 8))
    desenhar_quadra(ax)
    
    # larguras = [max(1, G[u][v].get("weight", 1) * 1.2) for u, v in G.edges]
    # nx.draw_networkx_edges(G, pos, ax=ax, width=larguras, alpha=0.65, arrows=True, arrowstyle="-|>", arrowsize=18, connectionstyle="arc3,rad=0.08")
    for u, v, d in G.edges(data=True):
        largura = max(1, d.get("weight", 1) * 1.2)

        # Se existe aresta nos dois sentidos, usa o mesmo rad.
        # Como a direção é invertida, o NetworkX desenha uma curva para cada lado.
        if G.has_edge(v, u):
            rad = 0.22
        else:
            rad = 0.08

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            ax=ax,
            width=largura,
            alpha=0.65,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=18,
            min_target_margin=18
        )
    
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[tamanho.get(n, 1200) for n in G.nodes], edgecolors="black", linewidths=1.5)
    labels = {}
    for n in G.nodes:
        linha = metricas[metricas["jogador"] == n]
        dep = linha.iloc[0]["dependencia_ofensiva"] if not linha.empty else 0
        labels[n] = f"{n}\nDep:{dep:.2f}"
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
    
    # labels_arestas = {(u, v): f"{d.get('weight', 0)} | V:{d.get('valor_total', 0)}" for u, v, d in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_arestas, ax=ax, font_size=7)
    for u, v, d in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        dx = x2 - x1
        dy = y2 - y1
        dist = (dx**2 + dy**2) ** 0.5

        if dist == 0:
            dist = 1

        px = -dy / dist
        py = dx / dist

        if G.has_edge(v, u):
            deslocamento = 0.34
        else:
            deslocamento = 0.18

        x_label = (x1 + x2) / 2 + px * deslocamento
        y_label = (y1 + y2) / 2 + py * deslocamento

        texto = f"{u}→{v}\n{d.get('weight', 0)} | V:{d.get('valor_total', 0):.1f}"

        ax.text(
            x_label,
            y_label,
            texto,
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.85
            )
        )
    
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(caminho, dpi=180)
    plt.close(fig)


def salvar_evolucao_temporal(resumo: pd.DataFrame, caminho: Path) -> None:
    if resumo.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    x = resumo["janela"]
    ax.plot(x, resumo["densidade"], marker="o", label="Densidade")
    ax.plot(x, resumo["taxa_acerto_passes"], marker="o", label="Taxa de acerto")
    ax.plot(x, resumo["dependencia_jogador_chave"], marker="o", label="Dependência do jogador-chave")
    ax.set_title("Evolução temporal da rede")
    ax.set_xlabel("Janela")
    ax.set_ylabel("Valor")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(caminho, dpi=180)
    plt.close(fig)


def salvar_heatmap_zonas(ranking: pd.DataFrame, caminho: Path) -> None:
    if ranking.empty:
        return
    matriz = ranking.pivot_table(index="zona_origem", columns="zona_destino", values="tentativas", fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(matriz.values, aspect="auto")
    ax.set_xticks(range(len(matriz.columns)))
    ax.set_xticklabels(matriz.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matriz.index)))
    ax.set_yticklabels(matriz.index)
    ax.set_title("Matriz de transições por zona")
    fig.colorbar(im, ax=ax, label="Quantidade de passes")
    for i in range(len(matriz.index)):
        for j in range(len(matriz.columns)):
            ax.text(j, i, int(matriz.values[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(caminho, dpi=180)
    plt.close(fig)


# =========================================================
# =========================================================

def interpretar_metricas_gerais(metricas_coletivas_df: pd.DataFrame, ranking: pd.DataFrame) -> List[str]:
    textos = []
    if metricas_coletivas_df.empty:
        return ["Não houve dados suficientes para construir a rede geral."]
    mc = metricas_coletivas_df.iloc[0]
    textos.append(
        f"A rede geral possui {int(mc['jogadores_ou_nos'])} nós e {int(mc['conexoes'])} conexões direcionadas, "
        f"com densidade {mc['densidade']} e reciprocidade {mc['reciprocidade']}."
    )
    textos.append(
        f"A centralização de grau foi {mc['centralizacao_grau']}. Quanto maior esse valor, mais a circulação depende de poucos jogadores."
    )
    if not ranking.empty:
        top = ranking.iloc[0]
        textos.append(
            f"O jogador mais influente pela composição de grau, valor, PageRank, closeness e intermediação foi {top['jogador']}, "
            f"com dependência ofensiva {top['dependencia_ofensiva']} e índice de importância {top['indice_importancia']}."
        )
    return textos


def gerar_relatorio_markdown(
    eventos: pd.DataFrame,
    G_geral: nx.DiGraph,
    metricas_geral: pd.DataFrame,
    ranking: pd.DataFrame,
    resumo_temp: pd.DataFrame,
    mudancas: pd.DataFrame,
    comparativo_cenario: pd.DataFrame,
    comparativo_pressao: pd.DataFrame,
    robustez_jogadores: pd.DataFrame,
    robustez_arestas: pd.DataFrame,
    caminhos: pd.DataFrame,
    comunidades: pd.DataFrame,
    fluxo_resumo: Optional[pd.DataFrame],
    corte_minimo: Optional[pd.DataFrame],
    config: Config,
) -> str:
    linhas = []
    linhas.append("# Relatório interpretativo — Análise de Futsal usando Grafos")
    linhas.append("")
    linhas.append("## 1. Pergunta de pesquisa")
    linhas.append(
        "Como a estrutura da rede de passes muda conforme o contexto da partida, como pressão adversária, cenário tático e janelas temporais?"
    )
    linhas.append("")
    linhas.append("## 2. Modelagem em grafos")
    linhas.append("- Jogadores = vértices.")
    linhas.append("- Passes = arestas direcionadas.")
    linhas.append("- Quantidade de passes = peso da aresta.")
    linhas.append("- Valor da ação = atributo da aresta usado para diferenciar passe simples, passe progressivo, ruptura, pivô, finalização e perda perigosa.")
    linhas.append("- Zonas da quadra = segundo grafo, usado para estudar progressão espacial.")
    linhas.append("- Remoção de jogador = simulação de neutralização de vértice.")
    linhas.append("- Remoção de aresta = simulação de bloqueio de linha de passe.")
    linhas.append("")
    linhas.append("## 3. Dados processados")
    linhas.append(f"Foram processados {len(eventos)} eventos.")
    linhas.append(f"Equipes encontradas: {', '.join(sorted(eventos['equipe'].dropna().unique()))}.")
    linhas.append(f"Cenários encontrados: {', '.join(sorted(eventos['cenario'].dropna().unique()))}.")
    linhas.append("")
    linhas.append("## 4. Leitura da rede geral")
    linhas.extend([f"- {t}" for t in interpretar_metricas_gerais(metricas_geral, ranking)])
    linhas.append("")

    if not comparativo_pressao.empty:
        linhas.append("## 5. Pressão adversária")
        for _, row in comparativo_pressao.iterrows():
            linhas.append(
                f"- Sob_pressao = {row['valor_criterio']}: taxa de acerto {row['taxa_acerto_passes']}, "
                f"densidade {row['densidade']}, centralização {row['centralizacao_grau']}, perdas perigosas {row['perdas_perigosas']}."
            )
        linhas.append("")

    if not comparativo_cenario.empty:
        linhas.append("## 6. Comparação por cenário")
        for _, row in comparativo_cenario.iterrows():
            linhas.append(
                f"- {row['valor_criterio']}: {row['passes_tentados']} passes tentados, densidade {row['densidade']}, "
                f"valor total da rede {row['valor_total']}, jogador-chave {row['jogador_chave']}."
            )
        linhas.append("")

    if not mudancas.empty:
        linhas.append("## 7. Mudanças temporais")
        for _, row in mudancas.iterrows():
            linhas.append(
                f"- {row['transicao']}: similaridade de arestas {row['similaridade_arestas']}, "
                f"delta densidade {row['delta_densidade']}, alerta: {row['alerta']}."
            )
        linhas.append("")

    if not robustez_jogadores.empty:
        linhas.append("## 8. Robustez por remoção de vértices")
        linhas.append("A remoção de um vértice simula a neutralização tática de um jogador por marcação forte.")
        for _, row in robustez_jogadores.head(5).iterrows():
            linhas.append(
                f"- Remover {row['jogador_removido']} gera queda de valor de {row['queda_valor_%']}% "
                f"e queda de passes de {row['queda_passes_%']}%."
            )
        linhas.append("")

    if not robustez_arestas.empty:
        linhas.append("## 9. Robustez por remoção de arestas")
        linhas.append("A remoção de uma aresta simula o adversário fechando uma linha de passe.")
        for _, row in robustez_arestas.head(5).iterrows():
            linhas.append(
                f"- Bloquear {row['aresta_removida']} reduz o valor da rede em {row['queda_valor_%']}%."
            )
        linhas.append("")

    if not caminhos.empty:
        linhas.append("## 10. Caminhos e sequências ofensivas")
        for _, row in caminhos.head(5).iterrows():
            linhas.append(
                f"- {row['caminho']}: ocorreu {int(row['quantidade'])} vez(es), gerou {int(row['finalizacoes'])} finalização(ões), "
                f"{int(row['gols'])} gol(s), valor médio {row['valor_medio']}."
            )
        linhas.append("")

    if fluxo_resumo is not None and not fluxo_resumo.empty:
        linhas.append("## 11. Fluxo máximo e corte mínimo")
        fr = fluxo_resumo.iloc[0]
        linhas.append(
            f"O fluxo máximo da saída de bola até a finalização foi {fr['fluxo_maximo']}. "
            f"Pelo teorema fluxo máximo-corte mínimo, o valor do corte mínimo também foi {fr['valor_corte_minimo']}."
        )
        linhas.append(
            "Na interpretação tática, o fluxo máximo representa a capacidade observada de progressão ofensiva; "
            "o corte mínimo indica os gargalos que, se bloqueados, reduzem essa capacidade."
        )
        if corte_minimo is not None and not corte_minimo.empty:
            for _, row in corte_minimo.head(5).iterrows():
                linhas.append(
                    f"- Gargalo {row['origem']} -> {row['destino']}: capacidade {row['capacidade']}, "
                    f"fluxo usado {row['fluxo_usado']}."
                )
        linhas.append("")

    if not comunidades.empty:
        linhas.append("## 12. Comunidades")
        for comunidade, grupo in comunidades.groupby("comunidade"):
            jogadores = ", ".join(grupo["jogador"].tolist())
            linhas.append(f"- Comunidade {comunidade}: {jogadores}.")
        linhas.append("")

    linhas.append("## 13. Conclusão")
    linhas.append(
        "O projeto deixa de apenas desenhar uma rede de passes e passa a usar conceitos centrais da Teoria dos Grafos: "
        "vértices, arestas direcionadas, pesos, centralidades, densidade, reciprocidade, clustering, comunidades, "
        "caminhos, grafos temporais e robustez por remoção de vértices/arestas."
    )
    return "\n".join(linhas)


# =========================================================
# =========================================================

def exportar_csv(df: pd.DataFrame, caminho: Path, encoding: str = "utf-8-sig") -> None:
    df.to_csv(caminho, index=False, encoding=encoding)


def executar_analise(config: Config) -> Dict[str, Path]:
    config.saida_dir.mkdir(parents=True, exist_ok=True)

    eventos_brutos = ler_csv(config.eventos_csv)
    eventos = padronizar_eventos(eventos_brutos, config)

    G_geral = construir_grafo_passes(eventos, equipe=config.equipe_principal)
    G_zonas = construir_grafo_zonas(eventos, equipe=config.equipe_principal)
    G_fluxo = construir_rede_fluxo_zonas(eventos, equipe=config.equipe_principal)
    fluxo_resumo, corte_minimo, fluxo_dict = analisar_fluxo_maximo_corte_minimo(G_fluxo)
    fluxo_arestas = tabela_fluxo_arestas(G_fluxo, fluxo_dict)

    metricas_geral = metricas_coletivas(G_geral, nome_rede="rede_geral_jogadores")
    metricas_zonas = metricas_coletivas(G_zonas, nome_rede="rede_geral_zonas")
    ranking = metricas_jogadores(G_geral)
    comunidades = detectar_comunidades(G_geral)
    resumo_temp = resumo_temporal(eventos, equipe=config.equipe_principal)
    mudancas = detectar_mudancas_temporais(eventos, equipe=config.equipe_principal)
    comparativo_cenario = comparar_por_coluna(eventos, "cenario", equipe=config.equipe_principal)
    comparativo_pressao = comparar_por_coluna(eventos, "sob_pressao", equipe=config.equipe_principal)
    comparativo_estado_placar = comparar_por_coluna(eventos, "estado_placar", equipe=config.equipe_principal)
    comparativo_fase = comparar_por_coluna(eventos, "fase_jogo", equipe=config.equipe_principal)
    duplas = ranking_duplas(eventos, equipe=config.equipe_principal)
    caminhos_3 = ranking_caminhos_posse(eventos, equipe=config.equipe_principal, tamanho_caminho=3)
    caminhos_4 = ranking_caminhos_posse(eventos, equipe=config.equipe_principal, tamanho_caminho=4)
    zonas = ranking_zonas(eventos, equipe=config.equipe_principal)
    pivos = inferir_pivos(eventos, equipe=config.equipe_principal)
    robustez_jogadores = analisar_robustez_jogadores(G_geral, pivos=pivos)
    robustez_arestas = analisar_robustez_arestas(G_geral, top_n=15)

    saidas: Dict[str, Path] = {}
    arquivos = {
        "eventos_enriquecidos": (eventos, "eventos_futsal_enriquecidos.csv"),
        "metricas_coletivas_geral": (metricas_geral, "metricas_coletivas_geral.csv"),
        "metricas_coletivas_zonas": (metricas_zonas, "metricas_coletivas_zonas.csv"),
        "ranking_jogadores": (ranking, "ranking_jogadores.csv"),
        "comunidades": (comunidades, "comunidades.csv"),
        "resumo_temporal": (resumo_temp, "resumo_temporal.csv"),
        "mudancas_temporais": (mudancas, "mudancas_temporais.csv"),
        "comparativo_cenario": (comparativo_cenario, "comparativo_cenario.csv"),
        "comparativo_pressao": (comparativo_pressao, "comparativo_pressao.csv"),
        "comparativo_estado_placar": (comparativo_estado_placar, "comparativo_estado_placar.csv"),
        "comparativo_fase_jogo": (comparativo_fase, "comparativo_fase_jogo.csv"),
        "ranking_duplas": (duplas, "ranking_duplas.csv"),
        "ranking_caminhos_3": (caminhos_3, "ranking_caminhos_3_jogadores.csv"),
        "ranking_caminhos_4": (caminhos_4, "ranking_caminhos_4_jogadores.csv"),
        "ranking_zonas": (zonas, "ranking_zonas.csv"),
        "robustez_jogadores": (robustez_jogadores, "robustez_jogadores.csv"),
        "robustez_arestas": (robustez_arestas, "robustez_arestas.csv"),
        "fluxo_maximo_resumo": (fluxo_resumo, "fluxo_maximo_resumo.csv"),
        "fluxo_maximo_arestas": (fluxo_arestas, "fluxo_maximo_arestas.csv"),
        "corte_minimo_arestas": (corte_minimo, "corte_minimo_arestas.csv"),
    }

    for chave, (df, nome_arquivo) in arquivos.items():
        caminho = config.saida_dir / nome_arquivo
        exportar_csv(df, caminho, config.encoding)
        saidas[chave] = caminho

    relatorio_md = gerar_relatorio_markdown(
        eventos=eventos,
        G_geral=G_geral,
        metricas_geral=metricas_geral,
        ranking=ranking,
        resumo_temp=resumo_temp,
        mudancas=mudancas,
        comparativo_cenario=comparativo_cenario,
        comparativo_pressao=comparativo_pressao,
        robustez_jogadores=robustez_jogadores,
        robustez_arestas=robustez_arestas,
        caminhos=caminhos_3,
        comunidades=comunidades,
        fluxo_resumo=fluxo_resumo,
        corte_minimo=corte_minimo,
        config=config,
    )
    relatorio_path = config.saida_dir / "relatorio_interpretativo.md"
    relatorio_path.write_text(relatorio_md, encoding="utf-8")
    saidas["relatorio"] = relatorio_path

    if config.gerar_graficos:
        grafo_path = config.saida_dir / "grafo_rede_passes_geral.png"
        salvar_grafo_passes(G_geral, eventos, grafo_path, "Rede Geral de Passes - Futsal", equipe=config.equipe_principal)
        saidas["grafico_rede_passes"] = grafo_path

        evolucao_path = config.saida_dir / "evolucao_temporal.png"
        salvar_evolucao_temporal(resumo_temp, evolucao_path)
        saidas["grafico_evolucao_temporal"] = evolucao_path

        zonas_path = config.saida_dir / "matriz_zonas.png"
        salvar_heatmap_zonas(zonas, zonas_path)
        saidas["grafico_matriz_zonas"] = zonas_path

        fluxo_path = config.saida_dir / "fluxo_maximo_corte_minimo.png"
        salvar_grafo_fluxo(G_fluxo, fluxo_dict, corte_minimo, fluxo_path)
        saidas["grafico_fluxo_maximo"] = fluxo_path

    return saidas


def imprimir_resumo_console(saidas: Dict[str, Path], config: Config) -> None:
    print("\n==================== ANÁLISE CONCLUÍDA ====================")
    print(f"Pasta de saída: {config.saida_dir}")
    print("\nArquivos principais:")
    principais = [
        "eventos_enriquecidos",
        "metricas_coletivas_geral",
        "ranking_jogadores",
        "resumo_temporal",
        "comparativo_cenario",
        "comparativo_pressao",
        "robustez_jogadores",
        "robustez_arestas",
        "fluxo_maximo_resumo",
        "corte_minimo_arestas",
        "ranking_caminhos_3",
        "relatorio",
    ]
    for chave in principais:
        if chave in saidas:
            print(f"- {chave}: {saidas[chave]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Análise de futsal usando Teoria dos Grafos")
    parser.add_argument("--eventos", default="eventos_futsal.csv", help="Caminho do CSV de eventos")
    parser.add_argument("--saida", default="resultados_grafos_futsal", help="Pasta de saída dos resultados")
    parser.add_argument("--janela", type=int, default=5, help="Tamanho da janela temporal em minutos")
    parser.add_argument("--equipe", default="nosso", help="Nome da equipe principal no CSV")
    parser.add_argument("--sem-graficos", action="store_true", help="Não gerar gráficos PNG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        eventos_csv=Path(args.eventos),
        saida_dir=Path(args.saida),
        tamanho_janela_min=args.janela,
        equipe_principal=args.equipe.lower().strip(),
        gerar_graficos=not args.sem_graficos,
    )
    saidas = executar_analise(config)
    imprimir_resumo_console(saidas, config)


if __name__ == "__main__":
    main()
