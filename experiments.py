"""
STEM-проєкт, варіант 4. Етап 1: експерименти з моделлю однієї печі.
"""
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pizza_model import (PERIODS, STEP, N_STEPS, OPEN_H, CLOSE_H, Pizzeria, run_many, fmt)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Константи
BLUE, ORANGE, RED, GREEN, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#999999"
HOURS = OPEN_H + np.arange(N_STEPS) * STEP / 60            # час початку кожного кроку, год
PERIOD_LABELS = ["ранок", "обід", "день", "вечір", "пізно"]
QUEUE_LIMIT = 10

SUMMARY_NAMES = {
    "arrived": "Надійшло замовлень, шт.", "lost": "Втрачено замовлень, шт.",
    "loss_share": "Частка втрачених, %", "accepted": "Прийнято до виконання, шт.",
    "served": "Виконано до 22:00, шт.", "unfinished": "Не виконано до 22:00, шт.",
    "revenue": "Виручка, грн", "cost": "Витрати на піч, грн", "profit": "Чистий прибуток, грн",
    "utilization": "Завантаження печі, %", "mean_queue": "Середня довжина черги, шт.",
    "max_queue": "Максимальна довжина черги, шт.", "mean_wait": "Середнє очікування до початку, хв",
}
SUMMARY_SCALE = {"loss_share": 100, "utilization": 100}

# Параметри аналізу чутливості: назва параметра моделі -> перелік значень.
# Щоб додати новий експеримент, достатньо додати рядок сюди (код функцій не змінюється).
SWEEP_CONFIG = {
    "p_scale": [0, 0.25, 0.5, 1.0, 1.5, 2.0],
    "time_scale": [0.5, 0.75, 1.0, 1.5, 2.0],
    "queue_max": [0, 3, 5, 10, 20],
    "p_ind": [0.0, 0.25, 0.5, 1.0],
}


# 1. НАЛАШТУВАННЯ ГРАФІКІВ
def configure_style():
    """Глобальні налаштування matplotlib."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
                         "axes.grid": True, "grid.color": "#e3e3e3", "axes.axisbelow": True})


def shade_periods(ax, labels=True):
    """Позначає на графіку межі логічних періодів дня."""
    for i, (_, h1, h2, _) in enumerate(PERIODS):
        if i % 2 == 1:
            ax.axvspan(h1, h2, color=ORANGE, alpha=0.10, lw=0)
        if labels:
            ax.text((h1 + h2) / 2, 1.01, PERIOD_LABELS[i], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10)
    ax.set_xlim(OPEN_H, CLOSE_H)
    ax.set_xticks(range(OPEN_H, CLOSE_H + 1, 2))


def _finish_figure(path, show):
    """Спільне завершення рисунка: компонування, збереження, (опційно) показ."""
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()


# 2. ЗАПУСК МОДЕЛІ
def run_single_experiment(seed=42):
    """Один прогін моделі. Повертає словник з 'orders' та 'steps'."""
    return Pizzeria(seed=seed).run()


def run_batch_experiments(n_runs=100, seed0=1, **model_params):
    """Серія незалежних прогонів (з можливістю змінити параметри моделі)."""
    return run_many(n_runs, seed0=seed0, **model_params)


def sweep(param, values, n_runs=100, seed0=1):
    """Аналіз чутливості: серія прогонів для кожного значення одного параметра."""
    out = []
    for v in values:
        r = run_batch_experiments(n_runs, seed0=seed0, **{param: v})["summary"]
        out.append(dict(value=v, arrived=r["arrived"].mean(), served=r["served"].mean(),
                        loss=r["loss_share"].mean() * 100, util=r["utilization"].mean() * 100,
                        queue=r["mean_queue"].mean(), wait=r["mean_wait"].mean(),
                        profit=r["profit"].mean(), unfinished=r["unfinished"].mean()))
    return pd.DataFrame(out)


def run_sweeps(config=SWEEP_CONFIG):
    """Виконує всі експерименти з аналізу чутливості."""
    return {param: sweep(param, values) for param, values in config.items()}


# 3. ОБЧИСЛЕННЯ ТАБЛИЦЬ ТА СТАТИСТИКИ
def build_order_log_table(orders, n_rows=12):
    """Фрагмент логу замовлень (перші n_rows рядків)."""
    log = orders.head(n_rows).copy()
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
    table = steps.iloc[k0:k0 + n_rows][["time", "arrived", "lost", "done", "queue", "busy"]].copy()
    table.columns = ["Час (початок кроку)", "Надійшло", "Втрачено", "Виконано",
                     "Черга (кінець кроку)", "Піч працювала, хв"]
    return table


def build_summary_table(summary):
    """Зведена статистика за серією прогонів: середнє, σ, 95% ДІ, мін, макс."""
    rows = []
    for key, title in SUMMARY_NAMES.items():
        v = summary[key].astype(float) * SUMMARY_SCALE.get(key, 1)
        sd = v.std(ddof=1)
        rows.append([title, v.mean(), sd, 1.96 * sd / np.sqrt(len(v)), v.min(), v.max()])
    return pd.DataFrame(rows, columns=["Показник", "Середнє", "Станд. відхилення",
                                       "±95% ДІ", "Мін", "Макс"])


def build_period_table(by_period):
    """Середні показники за логічними періодами дня (усереднення по прогонах)."""
    cols_needed = ["arrived", "lost", "loss_share", "utilization", "mean_queue", "mean_wait"]
    rows = []
    for i, (name, h1, h2, p) in enumerate(PERIODS):
        cols = {c: np.array([df.loc[i, c] for df in by_period], dtype=float) for c in cols_needed}
        rows.append([name, f"{h1:02d}:00–{h2:02d}:00", p,
                     np.mean(cols["arrived"]), np.mean(cols["lost"]),
                     np.nanmean(cols["loss_share"]) * 100, np.mean(cols["utilization"]) * 100,
                     np.mean(cols["mean_queue"]), np.nanmean(cols["mean_wait"])])
    return pd.DataFrame(rows, columns=["Період", "Час", "p", "Надійшло", "Втрачено",
                                       "Частка втрачених, %", "Завантаження, %",
                                       "Сер. черга", "Сер. очікування, хв"])


def compute_period_loss_std(by_period):
    """Стандартне відхилення частки втрачених замовлень у кожному періоді, %."""
    return [np.nanstd([df.loc[i, "loss_share"] for df in by_period], ddof=1) * 100
            for i in range(len(PERIODS))]


def compute_theory():
    """Теоретичні значення для перевірки адекватності генератора замовлень."""
    return dict(
        arrived=sum(p * (h2 - h1) * 60 / STEP for _, h1, h2, p in PERIODS),
        cook=0.75 * (15 + 20 + 30) / 3 + 0.25 * (25 + 35 + 50) / 3,
        price=0.75 * (180 + 300) / 2 + 0.25 * (320 + 500) / 2,
        p_ind=0.25,
    )


def build_generator_check(theory, all_orders, summary):
    """Порівняння теоретичних значень із результатами моделі."""
    return pd.DataFrame([
        ["Надійшло замовлень за день, шт.", theory["arrived"], summary["arrived"].mean()],
        ["Частка індивідуальних замовлень", theory["p_ind"], (all_orders["kind"] == "індивідуальна").mean()],
        ["Середній час приготування, хв", theory["cook"], all_orders["cook"].mean()],
        ["Середня вартість піци, грн", theory["price"], all_orders["price"].mean()],
    ], columns=["Величина", "Теоретичне значення", "Модель (100 прогонів)"])


# 4. ПОБУДОВА ГРАФІКІВ
def plot_single_dynamics(steps, path="fig1_single_dynamics.png", show=True):
    """Рис. 1: динаміка одного прогону (черга + кумулятивні підсумки)."""
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.90))

    ax[0].step(HOURS, steps["queue"], where="post", color=BLUE, lw=1.2)
    ax[0].axhline(QUEUE_LIMIT, color=RED, ls="--", lw=0.8)
    ax[0].text(OPEN_H + 0.1, QUEUE_LIMIT + 0.1, f"ліміт черги = {QUEUE_LIMIT}", color=RED, fontsize=10, va="bottom")
    shade_periods(ax[0])
    ax[0].set_ylim(0, 11.8)
    ax[0].set_xlabel("Час доби, год"); ax[0].set_ylabel("Довжина черги, замовлень")
    ax[0].set_title("а) Довжина черги", y=1.1)

    ax[1].step(HOURS, steps["arrived"].cumsum(), where="post", color=BLUE, label="надійшло")
    ax[1].step(HOURS, steps["done"].cumsum(), where="post", color=GREEN, label="виконано")
    ax[1].step(HOURS, steps["lost"].cumsum(), where="post", color=RED, label="втрачено")
    shade_periods(ax[1])
    ax[1].set_xlabel("Час доби, год"); ax[1].set_ylabel("Кумулятивна кількість, шт.")
    ax[1].set_title("б) Замовлення наростаючим підсумком", y=1.1)
    ax[1].legend(loc="upper left", fontsize=10, frameon=False)

    _finish_figure(path, show)


def plot_single_histograms(orders, path="fig2_single_hist.png", show=True):
    """Рис. 2: гістограми одного прогону."""
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.45))
    ax[0].hist(orders["cook"], bins=10, color=BLUE, edgecolor="white")
    ax[0].set_title("а) Час приготування"); ax[0].set_xlabel("хв"); ax[0].set_ylabel("Кількість замовлень")
    ax[1].hist(orders["price"], bins=10, color=ORANGE, edgecolor="white")
    ax[1].set_title("б) Вартість піци"); ax[1].set_xlabel("грн")
    ax[2].hist(orders["wait"].dropna(), bins=10, color=GREEN, edgecolor="white")
    ax[2].set_title("в) Очікування до початку"); ax[2].set_xlabel("хв")
    _finish_figure(path, show)


def plot_batch_histograms(summary, path="fig3_hist_100.png", show=True):
    """Рис. 3: розподіл підсумків за серією прогонів."""
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.60))

    ax[0].hist(summary["profit"], bins=12, color=BLUE, edgecolor="white")
    ax[0].axvline(summary["profit"].mean(), color=RED, ls="--", lw=1,
                  label=f"середнє = {summary['profit'].mean():.0f}")
    ax[0].set_title("а) Чистий прибуток за день"); ax[0].set_xlabel("грн"); ax[0].set_ylabel("Кількість прогонів")
    ax[0].legend(fontsize=10, frameon=False)

    ax[1].hist(summary["loss_share"] * 100, bins=12, color=ORANGE, edgecolor="white")
    ax[1].axvline(summary["loss_share"].mean() * 100, color=RED, ls="--", lw=1,
                  label=f"середнє = {summary['loss_share'].mean() * 100:.1f}%")
    ax[1].set_title("б) Частка втрачених замовлень"); ax[1].set_xlabel("%")
    ax[1].legend(fontsize=10, frameon=False)

    _finish_figure(path, show)


def plot_period_results(period_table, period_loss_std, queue_curves,
                        path="fig4_period_100.png", show=True):
    """Рис. 4: втрати за періодами та середня довжина черги протягом дня."""
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.90))

    x = np.arange(len(PERIODS))
    ax[0].bar(x, period_table["Частка втрачених, %"], yerr=period_loss_std, color=BLUE,
              capsize=3, error_kw=dict(lw=0.8))
    ax[0].set_xticks(x); ax[0].set_xticklabels(PERIOD_LABELS)
    ax[0].set_ylabel("Частка втрачених, %"); ax[0].set_ylim(0, 105)
    ax[0].set_title("а) Втрати за періодами (середнє ±σ)")

    m, sd = queue_curves.mean(axis=0), queue_curves.std(axis=0, ddof=1)
    ax[1].plot(HOURS, m, color=BLUE, lw=1.2, label="середнє")
    ax[1].fill_between(HOURS, np.clip(m - sd, 0, None), np.minimum(m + sd, QUEUE_LIMIT),
                       color=BLUE, alpha=0.2, label="±σ")
    ax[1].axhline(QUEUE_LIMIT, color=RED, ls="--", lw=0.8)
    shade_periods(ax[1]); ax[1].set_ylim(0, 11.8)
    ax[1].set_xlabel("Час доби, год"); ax[1].set_ylabel("Довжина черги, шт.")
    ax[1].set_title("б) Довжина черги протягом дня", y=1.1)
    ax[1].legend(loc="lower right", fontsize=10, frameon=False)

    _finish_figure(path, show)


# 5. ВИВЕДЕННЯ В КОНСОЛЬ
def _print_section(title, table):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(table)


def print_summary(summary_table, period_table, generator_check):
    """Друкує ключові таблиці у консоль."""
    pd.set_option("display.width", 220, "display.max_columns", 30)
    _print_section("СЕРЕДНІ ПОКАЗНИКИ ЗА 100 ПРОГОНІВ", summary_table.round(2))
    _print_section("ВТРАТИ ТА ЗАВАНТАЖЕННЯ ПО ПЕРІОДАХ",
                   period_table[["Період", "Час", "Частка втрачених, %", "Завантаження, %"]].round(2))
    _print_section("ПЕРЕВІРКА АДЕКВАТНОСТІ МОДЕЛІ", generator_check.round(3))


# 6. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
def save_results(results, path="results.pkl"):
    """Серіалізує словник результатів у pickle."""
    with open(path, "wb") as f:
        pickle.dump(results, f)


# 7. ТОЧКА ВХОДУ
def main(show_plots=True):
    configure_style()

    # один прогін
    single = run_single_experiment(seed=42)
    orders, steps = single["orders"], single["steps"]
    log_table = build_order_log_table(orders)
    step_log = build_step_log_table(steps)
    plot_single_dynamics(steps, show=show_plots)
    plot_single_histograms(orders, show=show_plots)

    # 100 прогонів
    base = run_batch_experiments(100, seed0=1)
    summary = base["summary"]
    summary_table = build_summary_table(summary)
    period_table = build_period_table(base["by_period"])
    period_loss_std = compute_period_loss_std(base["by_period"])
    plot_batch_histograms(summary, show=show_plots)
    plot_period_results(period_table, period_loss_std, base["queue_curves"], show=show_plots)

    # перевірка адекватності та аналіз чутливості
    theory = compute_theory()
    generator_check = build_generator_check(theory, base["orders"], summary)
    sweeps = run_sweeps()

    # вивід і збереження
    print_summary(summary_table, period_table, generator_check)
    save_results(dict(single=single, log_table=log_table, step_log=step_log,
                      summary_table=summary_table, period_table=period_table,
                      generator_check=generator_check, sweeps=sweeps, theory=theory))


if __name__ == "__main__":
    main()
    