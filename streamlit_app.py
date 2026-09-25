"""Display only: all beliefs and transition probabilities come from the frozen network."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import streamlit as st

from runtime.engine import SEEDS, verify_integrity
from runtime.schema import changes, edge_pairs, observed_next, validate_scenario
from runtime.ui_adapter import predict_streamlit_payload


ROOT = Path(__file__).resolve().parent
EXAMPLES = [ROOT / "examples" / f"station_{n}.json" for n in (7, 8, 9, 10)]
st.set_page_config(page_title="TWM · станция", page_icon="🔭", layout="wide")


@st.cache_resource
def trusted_hashes() -> dict[str, str]:
    return verify_integrity()


@st.cache_data(show_spinner="Нейросеть рассчитывает прогноз…")
def predict_cached(history_json: str, actions: tuple[int, ...], seed: int) -> dict:
    return predict_streamlit_payload(history_json, actions, seed)


def state_table(nodes: list[int], edges: list[int]) -> None:
    n = len(nodes)
    pairs = edge_pairs(n)
    adjacency = {(i, j): value for value, (i, j) in zip(edges, pairs)}
    st.write("**Узлы:** " + " · ".join(f"{i}: {'1' if v else '0'}" for i, v in enumerate(nodes)))
    st.caption("Матрица направленных связей: строка — откуда, столбец — куда; 1 — включена.")
    st.dataframe(
        [{"из / в": str(i), **{str(j): "—" if i == j else str(adjacency[(i, j)]) for j in range(n)}} for i in range(n)],
        hide_index=True, use_container_width=True,
    )


def show_differences(before_nodes: list[int], before_edges: list[int],
                     after_nodes: list[int], after_edges: list[int]) -> None:
    n = len(before_nodes)
    node_diff = changes(before_nodes, after_nodes, [f"узел {i}" for i in range(n)])
    edge_diff = changes(before_edges, after_edges, [f"{i} → {j}" for i, j in edge_pairs(n)])
    st.write(f"Узлы: **{len(node_diff)}** изменений · связи: **{len(edge_diff)}** изменений")
    st.caption("; ".join(node_diff) if node_diff else "Узлы без изменений")
    with st.expander("Какие связи изменились"):
        st.write("; ".join(edge_diff) if edge_diff else "Связи без изменений")


def main() -> None:
    st.title("TWM · интерактивная станция")
    st.caption("Канонические веса R12, фиксированный протокол rollout R13. Динамику рассчитывает нейросеть; интерфейс лишь показывает её ответ. Примеры взяты из синтетических миров.")
    try:
        hashes = trusted_hashes()
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Проверка целостности не прошла: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Сценарий")
        choice = st.selectbox("Наблюдения", [p.stem for p in EXAMPLES] + ["Свой JSON"])
        if choice == "Свой JSON":
            uploaded = st.file_uploader("История из 8 наблюдений", type="json")
            if uploaded is None:
                st.info("Загрузите JSON с n_nodes и history; recorded необязателен.")
                st.stop()
            raw = uploaded.getvalue()
            source_key = "upload:" + hashlib.sha256(raw).hexdigest()
        else:
            path = next(p for p in EXAMPLES if p.stem == choice)
            raw = path.read_bytes()
            source_key = choice
        try:
            scenario = validate_scenario(json.loads(raw.decode("utf-8")))
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            st.error(f"Невозможно прочитать сценарий: {exc}")
            st.stop()
        if st.session_state.get("source_key") != source_key:
            st.session_state.source_key = source_key
            st.session_state.history = scenario["history"].copy()
            st.session_state.recorded_index = 0
        if st.button("Начать сценарий заново"):
            st.session_state.history = scenario["history"].copy()
            st.session_state.recorded_index = 0
        seed = st.selectbox("Зафиксированный seed", SEEDS)
        st.caption("Все три seed — отдельные проверенные пары весов. Показанные вероятности belief не гарантируют калибровку.")

    history = st.session_state.history
    current = history[-1]
    nodes, edges = current["nodes_after"], current["edges_after"]
    recorded = scenario.get("recorded", [])
    cursor = st.session_state.recorded_index
    next_recorded = recorded[cursor] if cursor < len(recorded) else None
    st.subheader("Текущее наблюдаемое состояние")
    state_table(nodes, edges)
    st.caption("Действие — выбор узла по индексу. Эти двоичные узлы и связи пока не обозначают реальные двери, предметы или язык.")

    left, right = st.columns([1, 2])
    with left:
        default_action = next_recorded["action"] if next_recorded else 0
        action = st.selectbox("Действие над узлом", list(range(scenario["n_nodes"])),
                              index=default_action, key=f"action-{source_key}-{cursor}")
        horizon = st.slider("Шагов модели", 1, 16, value=3)
        plan_text = st.text_input("Действия после первого (через запятую; пусто — повторять)", "")
        if plan_text.strip():
            try:
                tail = [int(v.strip()) for v in plan_text.split(",")]
            except ValueError:
                st.error("Введите номера узлов через запятую.")
                st.stop()
            if len(tail) != horizon - 1 or any(not 0 <= a < scenario["n_nodes"] for a in tail):
                st.error(f"Нужно указать {horizon - 1} действий с номерами 0–{scenario['n_nodes'] - 1}.")
                st.stop()
            actions = (action, *tail)
        else:
            actions = (action,) * horizon
        if next_recorded:
            st.caption(f"Следующее сохранённое действие: узел {next_recorded['action']}. Факт доступен только для этого действия из текущего состояния.")
        else:
            st.caption("Дальнейших реальных наблюдений в примере нет.")
    try:
        output = predict_cached(json.dumps(history, ensure_ascii=False), actions, seed)
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Inference не выполнен: {exc}")
        st.stop()
    first = output["steps"][0]

    with right:
        st.subheader("Прогноз после выбранного действия")
        show_differences(nodes, edges, first["nodes_after"], first["edges_after"])
        st.write("Предсказанные состояния узлов: " + " ".join(str(v) for v in first["nodes_after"]))
        st.caption("Правило отображения модели из R13: вероятность ≥ 0,5 даёт 1. Дальнейшие шаги используют собственные предсказания модели.")
        observed = observed_next(scenario, cursor, nodes, edges, action)
        if observed is None:
            st.info("Для этого действия и состояния наблюдаемого результата нет. Точность сравнить нельзя.")
        else:
            node_errors = sum(x != y for x, y in zip(first["nodes_after"], observed["nodes_after"]))
            edge_errors = sum(x != y for x, y in zip(first["edges_after"], observed["edges_after"]))
            if node_errors + edge_errors == 0:
                st.success("Предсказанное состояние полностью совпало с записанным фактом.")
            else:
                st.error(f"Расхождение с записанным фактом: {node_errors} узлов и {edge_errors} связей.")
            with st.expander("Показать записанное фактическое состояние"):
                state_table(observed["nodes_after"], observed["edges_after"])
            if st.button("Принять фактическое наблюдение и продолжить"):
                st.session_state.history = (history + [observed])[-8:]
                st.session_state.recorded_index = cursor + 1
                st.rerun()

    st.subheader("Belief модели по наблюдаемой истории")
    st.caption("Индексы 0–5 — внутренние обученные контексты двух факторов; это не словесные законы мира. Поддержка альтернатив может быть недооценена.")
    b1, b2 = st.columns(2)
    with b1:
        st.write("Фактор связей")
        st.dataframe([{"контекст": i, "вес": round(v, 5)} for i, v in enumerate(output["edge_belief"])], hide_index=True)
    with b2:
        st.write("Фактор узлов")
        st.dataframe([{"контекст": i, "вес": round(v, 5)} for i, v in enumerate(output["node_belief"])], hide_index=True)

    st.subheader("Рекурсивная симуляция")
    st.dataframe([{
        "шаг": i + 1, "действие": item["action"],
        "узлы 1": sum(item["nodes_after"]), "связи 1": sum(item["edges_after"]),
        "узлов изменилось": sum(x != y for x, y in zip(item["nodes_before"], item["nodes_after"])),
        "связей изменилось": sum(x != y for x, y in zip(item["edges_before"], item["edges_after"])),
    } for i, item in enumerate(output["steps"])], hide_index=True, use_container_width=True)
    with st.expander("История реальных наблюдений"):
        st.dataframe([{
            "шаг": i + 1, "действие": t["action"],
            "узлов изменилось": sum(x != y for x, y in zip(t["nodes_before"], t["nodes_after"])),
            "связей изменилось": sum(x != y for x, y in zip(t["edges_before"], t["edges_after"])),
        } for i, t in enumerate(history)], hide_index=True, use_container_width=True)
        st.json(history)
    with st.expander("Технические данные · сырые вероятности и происхождение"):
        st.write("Seed:", seed)
        st.json({k: v for k, v in hashes.items() if k.startswith("weights/") and f"seed{seed}" in k})
        if observed is not None:
            truth = observed["nodes_after"] + observed["edges_after"]
            guesses = first["nodes_after"] + first["edges_after"]
            probabilities = first["nodes_prob"] + first["edges_prob"]
            nll = -sum(
                y * math.log(min(max(p, 1e-6), 1 - 1e-6))
                + (1 - y) * math.log(1 - min(max(p, 1e-6), 1 - 1e-6))
                for y, p in zip(truth, probabilities)
            ) / len(truth)
            st.json({"full_state_exact": guesses == truth,
                     "normalized_hamming": sum(x != y for x, y in zip(guesses, truth)) / len(truth),
                     "nll_per_bit": nll})
        st.json(output)


if __name__ == "__main__":
    main()
