"""
STEM-проєкт, варіант 4. Етап 2: експерименти з багатоканальною моделлю піцерії.

Пошук оптимального розкладу печей за періодами (максимум чистого прибутку за день),
порівняння зі сталою кількістю печей, один прогін і серія зі 100 прогонів,
перевірка адекватності. Графіки будує plots2.py (з results2.pkl).
"""
import pickle
import warnings

import numpy as np
import pandas as pd

from pizza_model import PERIODS, STEP, OPEN_H, MAX_OVENS, N_PER, Pizzeria, run_many, fmt
from plots2 import plot_all

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Константи
OVENS = range(1, MAX_OVENS + 1)
N_RUNS = 100
SEED_OPT = 1         # прогони для пошуку оптимуму: seed 1..100
SEED_TEST = 1001     # незалежні прогони для оцінки результатів: seed 1001..1100
SEED_SINGLE = 42

SUMMARY_NAMES = {
    "arrived": "Надійшло замовлень, шт.", "lost": "Втрачено замовлень, шт.",
    "loss_share": "Частка втрачених, %", "accepted": "Прийнято до виконання, шт.",
    "served": "Виконано до 22:00, шт.", "unfinished": "Не виконано до 22:00, шт.",
    "revenue": "Виручка, грн", "oven_hours": "Печо-години, год",
    "cost": "Витрати на печі, грн", "profit": "Чистий прибуток, грн",
    "utilization": "Завантаження печей, %", "mean_queue": "Середня довжина черги, шт.",
    "max_queue": "Максимальна довжина черги, шт.", "mean_wait": "Середнє очікування до початку, хв",
}
SUMMARY_SCALE = {"loss_share": 100, "utilization": 100}

# Аналіз чутливості (адекватність) при оптимальному розкладі
SWEEP_CONFIG = {
    "p_scale": [0.5, 0.75, 1.0, 1.25, 1.5],
    "time_scale": [0.75, 1.0, 1.25, 1.5],
    "queue_max": [0, 3, 5, 10, 20],
}


# 1. ПОШУК ОПТИМАЛЬНОГО РОЗКЛАДУ
class ScheduleEvaluator:
    """Середній денний прибуток для розкладу (за періодами) на тих самих
    seed-ах (метод спільних випадкових чисел), з кешем результатів."""

    def __init__(self, n_runs=N_RUNS, seed0=SEED_OPT):
        self.n_runs, self.seed0 = n_runs, seed0
        self.cache = {}

    def __call__(self, sched):
        key = tuple(int(c) for c in sched)
        if key not in self.cache:
            self.cache[key] = run_many(self.n_runs, seed0=self.seed0, ovens=key)["summary"]
        return self.cache[key]

    def profit(self, sched):
        return self(sched)["profit"].mean()


def optimize_schedule(evaluate, start):
    """Покоординатний спуск: по черзі для кожного періоду перебираємо 1..10 печей,
    решту періодів фіксуємо. Повторюємо, доки розклад не перестане змінюватися.
    На відміну від оптимізації кожного періоду окремо, враховує черги,
    що переходять з одного періоду в наступний."""
    sched = list(start)
    history = [tuple(sched)]
    while True:
        changed = False
        for j in range(N_PER):
            profits = [evaluate.profit(sched[:j] + [c] + sched[j + 1:]) for c in OVENS]
            best = int(np.argmax(profits)) + 1
            if best != sched[j]:
                sched[j], changed = best, True
                history.append(tuple(sched))
        if not changed:
            return np.array(sched), history


def period_profit_curves(evaluate, sched):
    """Для кожного періоду: денний прибуток залежно від кількості печей у ньому
    (інші періоди - за оптимальним розкладом)."""
    sched = list(sched)
    return {j: [evaluate.profit(sched[:j] + [c] + sched[j + 1:]) for c in OVENS]
            for j in range(N_PER)}


# 2. ЕКСПЕРИМЕНТИ
def run_constant_ovens(seed0=SEED_TEST):
    """Стала кількість печей на весь день: c = 1..10."""
    rows = []
    for c in OVENS:
        m = run_many(N_RUNS, seed0=seed0, ovens=c)["summary"]
        rows.append(dict(ovens=c, arrived=m["arrived"].mean(), served=m["served"].mean(),
                         unfinished=m["unfinished"].mean(), loss=m["loss_share"].mean() * 100,
                         util=m["utilization"].mean() * 100, mean_queue=m["mean_queue"].mean(),
                         mean_wait=m["mean_wait"].mean(), revenue=m["revenue"].mean(),
                         cost=m["cost"].mean(), profit=m["profit"].mean(),
                         profit_sd=m["profit"].std(ddof=1)))
    return pd.DataFrame(rows)


def sweep(param, values, sched, seed0=SEED_TEST):
    """Аналіз чутливості: серія прогонів для кожного значення одного параметра."""
    out = []
    for v in values:
        m = run_many(N_RUNS, seed0=seed0, ovens=sched, **{param: v})["summary"]
        out.append(dict(value=v, arrived=m["arrived"].mean(), served=m["served"].mean(),
                        loss=m["loss_share"].mean() * 100, util=m["utilization"].mean() * 100,
                        queue=m["mean_queue"].mean(), wait=m["mean_wait"].mean(),
                        profit=m["profit"].mean()))
    return pd.DataFrame(out)


def check_ovens_trend(const_table):
    """Адекватність: зі збільшенням кількості печей втрати не зростають,
    завантаження печей і очікування не зростають, витрати зростають."""
    def non_increasing(col):
        return bool((np.diff(const_table[col]) <= 1e-9).all())
    return {
        "частка втрачених не зростає": non_increasing("loss"),
        "завантаження печей не зростає": non_increasing("util"),
        "очікування не зростає": non_increasing("mean_wait"),
        "витрати зростають": bool((np.diff(const_table["cost"]) > 0).all()),
    }


# 3. ТАБЛИЦІ
def build_summary_table(summary):
    """Зведена статистика за серією прогонів: середнє, σ, 95% ДІ, мін, макс."""
    rows = []
    for key, title in SUMMARY_NAMES.items():
        v = summary[key].astype(float) * SUMMARY_SCALE.get(key, 1)
        sd = v.std(ddof=1)
        rows.append([title, v.mean(), sd, 1.96 * sd / np.sqrt(len(v)), v.min(), v.max()])
    return pd.DataFrame(rows, columns=["Показник", "Середнє", "Станд. відхилення",
                                       "±95% ДІ", "Мін", "Макс"])


def build_period_table(by_period, sched):
    """Середні показники за періодами дня (усереднення по прогонах)."""
    cols = ["arrived", "lost", "loss_share", "utilization", "mean_queue", "mean_wait",
            "revenue", "cost", "profit"]
    rows = []
    for i, (name, h1, h2, p) in enumerate(PERIODS):
        v = {c: np.array([df.loc[i, c] for df in by_period], dtype=float) for c in cols}
        rows.append([name, f"{h1:02d}:00–{h2:02d}:00", p, sched[i],
                     v["arrived"].mean(), v["lost"].mean(), np.nanmean(v["loss_share"]) * 100,
                     v["utilization"].mean() * 100, v["mean_queue"].mean(),
                     np.nanmean(v["mean_wait"]), v["revenue"].mean(), v["cost"].mean(),
                     v["profit"].mean()])
    return pd.DataFrame(rows, columns=["Період", "Час", "p", "Печей", "Надійшло", "Втрачено",
                                       "Частка втрачених, %", "Завантаження, %", "Сер. черга",
                                       "Сер. очікування, хв", "Виручка, грн", "Витрати, грн",
                                       "Прибуток, грн"])


def build_comparison_table(schemes):
    """Порівняння схем роботи печей (на незалежних прогонах)."""
    rows = []
    for name, m in schemes.items():
        p = m["profit"]
        rows.append([name, p.mean(), 1.96 * p.std(ddof=1) / np.sqrt(len(p)),
                     m["loss_share"].mean() * 100, m["utilization"].mean() * 100,
                     m["oven_hours"].mean(), m["cost"].mean(), m["mean_wait"].mean()])
    return pd.DataFrame(rows, columns=["Схема", "Прибуток, грн", "±95% ДІ",
                                       "Частка втрачених, %", "Завантаження, %",
                                       "Печо-години", "Витрати, грн", "Сер. очікування, хв"])


def build_order_log_table(orders, n_rows=12):
    """Фрагмент логу замовлень (перші n_rows рядків)."""
    log = orders.head(n_rows)
    return pd.DataFrame({
        "№": log["num"], "Тип": log["kind"], "Вартість, грн": log["price"],
        "Час на піч, хв": log["cook"].round(1),
        "Надійшло": [fmt(t) for t in log["t_arrive"]],
        "Початок": [fmt(t) if pd.notna(t) else "—" for t in log["t_start"]],
        "Кінець": [fmt(t) if pd.notna(t) else "—" for t in log["t_finish"]],
        "Очікування, хв": log["wait"].round(1),
        "Статус": np.where(log["lost"], "втрачено",
                           np.where(log["t_finish"].notna(), "виконано", "в роботі")),
    })


def build_step_log_table(steps, start_hour=17, n_rows=7):
    """Фрагмент логу кроків (за замовчуванням — вечірній пік, 17:00–17:30)."""
    k0 = (start_hour - OPEN_H) * 60 // STEP
    table = steps.iloc[k0:k0 + n_rows][["time", "ovens", "arrived", "lost", "done",
                                        "cooking", "queue", "busy"]].copy()
    table.columns = ["Час (початок кроку)", "Печей увімкнено", "Надійшло", "Втрачено",
                     "Виконано", "Печей зайнято", "Черга (кінець кроку)", "Печі працювали, хв"]
    return table


# 4. ВИВЕДЕННЯ ТА ЗБЕРЕЖЕННЯ
def _print_section(title, table):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(table)


def print_summary(r):
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("Оптимальний розклад печей за періодами:", r["opt_sched"].tolist())
    _print_section("ПОРІВНЯННЯ СХЕМ (100 незалежних прогонів)", r["cmp_table"].round(1))
    _print_section("ОПТИМАЛЬНИЙ РОЗКЛАД: СЕРЕДНІ ЗА 100 ПРОГОНІВ", r["summary_table"].round(2))
    _print_section("ПОКАЗНИКИ ЗА ПЕРІОДАМИ",
                   r["period_table"][["Період", "Печей", "Частка втрачених, %",
                                      "Завантаження, %", "Прибуток, грн"]].round(1))
    _print_section("АДЕКВАТНІСТЬ: ВПЛИВ КІЛЬКОСТІ ПЕЧЕЙ",
                   "\n".join(f"{k}: {'так' if ok else 'НІ'}" for k, ok in r["ovens_trend"].items()))
    for param, table in r["sweeps"].items():
        _print_section(f"ЧУТЛИВІСТЬ: {param}", table.round(1))


def save_results(results, path="results2.pkl"):
    with open(path, "wb") as f:
        pickle.dump(results, f)


# 5. ТОЧКА ВХОДУ
def run_scheme(ovens, sched):
    """Серія незалежних прогонів однієї схеми + дані для таблиць і графіків."""
    res = run_many(N_RUNS, seed0=SEED_TEST, ovens=ovens)
    return dict(sched=list(sched), summary=res["summary"],
                period_table=build_period_table(res["by_period"], sched),
                queue_curves=res["queue_curves"], cooking_curves=res["cooking_curves"])


def main(show_plots=True):
    r = {}

    # стала кількість печей (етап 1 - це c = 1)
    r["const_table"] = const_table = run_constant_ovens()
    best_const = int(const_table.loc[const_table["profit"].idxmax(), "ovens"])
    r["best_const"] = best_const

    # пошук оптимального розкладу (seed 1..100), старт - найкраща стала кількість
    evaluate = ScheduleEvaluator()
    opt_sched, r["opt_history"] = optimize_schedule(evaluate, [best_const] * N_PER)
    r["opt_sched"] = opt_sched
    r["period_curves"] = period_profit_curves(evaluate, opt_sched)

    # оцінка трьох схем на незалежних прогонах (seed 1001..1100)
    r["schemes"] = schemes = {
        "1 піч (етап 1)": run_scheme(1, [1] * N_PER),
        f"{best_const} печі весь день": run_scheme(best_const, [best_const] * N_PER),
        "Розклад " + "-".join(map(str, opt_sched)): run_scheme(opt_sched, opt_sched),
    }
    best = schemes["Розклад " + "-".join(map(str, opt_sched))]
    r["summary_table"] = build_summary_table(best["summary"])
    r["period_table"] = best["period_table"]
    r["cmp_table"] = build_comparison_table({k: v["summary"] for k, v in schemes.items()})

    # один прогін
    single = Pizzeria(seed=SEED_SINGLE, ovens=opt_sched).run()
    r["single"] = single
    r["log_table"] = build_order_log_table(single["orders"])
    r["step_log"] = build_step_log_table(single["steps"])

    # адекватність
    r["ovens_trend"] = check_ovens_trend(const_table)
    r["sweeps"] = {p: sweep(p, v, opt_sched) for p, v in SWEEP_CONFIG.items()}

    print_summary(r)
    save_results(r)
    plot_all(r, show=show_plots)


if __name__ == "__main__":
    main()
