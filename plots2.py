"""
STEM-проєкт, варіант 4. Етап 2: рисунки для звіту (4 рисунки, по одному графіку).

Запуск окремо: python plots2.py - перебудовує рисунки з results2.pkl
без повторного моделювання.
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np

from pizza_model import PERIODS, STEP, N_STEPS, OPEN_H, CLOSE_H, QUEUE_MAX, make_schedule

# Кольори закріплені за сутностями й однакові на всіх рисунках
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # розклад / стала к-сть / 1 піч
RED = "#e34948"                                        # втрати, ліміт черги
LIGHT = "#dcdbd5"                                      # "увімкнено", інші варіанти
INK, INK2 = "#0b0b0b", "#52514e"                       # текст
PEAK_BG = "#fbefe6"                                    # фон пікових періодів

HOURS = OPEN_H + np.arange(N_STEPS) * STEP / 60
PERIOD_LABELS = ["ранок", "обід", "день", "вечір", "пізно"]
FIGSIZE = (9.6, 4.8)


# ДОПОМІЖНЕ
def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": "#ecebe6", "axes.axisbelow": True,
        "axes.edgecolor": "#b8b7b0", "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9.5,
    })


def money(x):
    """13030.2 -> '13 030'"""
    return f"{x:,.0f}".replace(",", " ")


def ovens_word(n):
    return "піч" if n == 1 else "печі" if n < 5 else "печей"


def time_axis(ax, extra=None):
    """Вісь часу 9:00-22:00, пікові періоди - світлим фоном, назви періодів над графіком
    (extra - другий рядок під назвою періоду, наприклад показник за період)."""
    for i, (_, h1, h2, _) in enumerate(PERIODS):
        if i % 2 == 1:
            ax.axvspan(h1, h2, color=PEAK_BG, lw=0, zorder=0)
        text = PERIOD_LABELS[i] if extra is None else f"{PERIOD_LABELS[i]}\n{extra[i]}"
        ax.text((h1 + h2) / 2, 1.01, text, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5, color=INK2)
        if i:
            ax.axvline(h1, color="#c9c8c1", lw=0.8, ls=":", zorder=1)
    ax.set_xlim(OPEN_H, CLOSE_H)
    ax.set_xticks(range(OPEN_H, CLOSE_H + 1))
    ax.set_xticklabels([f"{h}:00" if h % 2 == 1 else "" for h in range(OPEN_H, CLOSE_H + 1)])
    ax.set_xlabel("час доби")


def label(ax, x, y, text, color=INK, **kw):
    """Підпис значення на графіку з білою підкладкою (читається поверх ліній)."""
    ax.text(x, y, text, color=color, fontsize=9.5, ha=kw.pop("ha", "center"),
            va=kw.pop("va", "bottom"), bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                                 ec="none", alpha=0.85), **kw)


def _finish(fig, ax, title, name, show, pad=22):
    ax.set_title(title, pad=pad)
    fig.tight_layout()
    fig.savefig(name, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _scheme(r, i):
    return list(r["schemes"].values())[i]


def _sched_name(r):
    return "-".join(map(str, r["opt_sched"]))


# РИС. 1. ОПТИМАЛЬНИЙ РОЗКЛАД ПЕЧЕЙ
def plot_schedule(r, name="s2_fig1_schedule.png", show=True):
    """Скільки печей увімкнено в кожному періоді і наскільки вони зайняті (100 прогонів)."""
    sched, data = r["opt_sched"], _scheme(r, 2)
    busy = data["cooking_curves"].mean(axis=0)
    table = data["period_table"]
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.fill_between(HOURS, make_schedule(sched), step="post", color=LIGHT, lw=0,
                    label="печей увімкнено (за розкладом)")
    ax.plot(HOURS, busy, color=BLUE, lw=2, label="печей зайнято (середнє за 100 прогонів)")
    period_of_step = make_schedule(np.arange(1, len(PERIODS) + 1)) - 1
    for i, (_, h1, h2, p) in enumerate(PERIODS):
        top = max(sched[i], busy[period_of_step == i].max())
        label(ax, (h1 + h2) / 2, top + 0.12, f"{sched[i]} {ovens_word(sched[i])}\n"
              f"завант. {table['Завантаження, %'][i]:.0f} %", fontweight="bold")
    time_axis(ax)
    ax.set_ylim(0, max(sched) + 1.5)
    ax.set_yticks(range(0, max(sched) + 2))
    ax.set_ylabel("кількість печей")
    ax.legend(loc="upper left")
    _finish(fig, ax, f"Оптимальний розклад печей {_sched_name(r)}", name, show)


# РИС. 2. ПРИБУТОК: СТАЛА КІЛЬКІСТЬ ПЕЧЕЙ VS РОЗКЛАД
def plot_profit(r, name="s2_fig2_profit.png", show=True):
    const, best_c = r["const_table"], r["best_const"]
    sched_profit = _scheme(r, 2)["summary"]["profit"].mean()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ci = 1.96 * const["profit_sd"] / np.sqrt(100)
    colors = [ORANGE if c == best_c else (AQUA if c == 1 else LIGHT) for c in const["ovens"]]
    ax.bar(const["ovens"], const["profit"] / 1000, yerr=ci / 1000, color=colors, width=0.7,
           error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
    for c, v in zip(const["ovens"], const["profit"] / 1000):
        ax.text(c, v + (0.35 if v >= 0 else -0.35), f"{v:.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9.5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axhline(sched_profit / 1000, color=BLUE, ls="--", lw=1.8)
    label(ax, 10.45, sched_profit / 1000 + 0.3,
          f"розклад {_sched_name(r)}: {sched_profit / 1000:.1f}", color=BLUE, ha="right",
          fontweight="bold")
    best_v = const.loc[const["ovens"] == best_c, "profit"].iloc[0]
    ax.annotate("", xy=(best_c + 0.38, best_v / 1000), xytext=(best_c + 0.38, sched_profit / 1000),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.2, shrinkA=0, shrinkB=0))
    ax.text(best_c + 0.55, (best_v + sched_profit) / 2000, f"+{money(sched_profit - best_v)} грн",
            va="center", fontsize=9.5, color=BLUE, fontweight="bold")

    ax.set_xticks(list(const["ovens"]))
    ax.set_xticklabels([f"{c}\n(етап 1)" if c == 1 else str(c) for c in const["ovens"]])
    ax.set_xlabel("кількість печей, увімкнених на весь день")
    ax.set_ylabel("чистий прибуток, тис. грн/день")
    ax.set_ylim(top=sched_profit / 1000 + 2.5)
    ax.grid(axis="x", visible=False)
    _finish(fig, ax, "Прибуток за день: стала кількість печей проти розкладу (100 прогонів)",
            name, show, pad=10)


# РИС. 3. ЧЕРГА: СЕРЕДНЄ ЗА 100 ПРОГОНІВ
def plot_queue(r, name="s2_fig3_queue.png", show=True):
    data = _scheme(r, 2)
    q, table = data["queue_curves"], data["period_table"]
    mean = q.mean(axis=0)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.fill_between(HOURS, np.percentile(q, 10, axis=0), np.percentile(q, 90, axis=0),
                    color=ORANGE, alpha=0.2, lw=0, label="у цих межах 80 % прогонів")
    ax.plot(HOURS, mean, color=ORANGE, lw=2, label="середня довжина черги")
    ax.axhline(QUEUE_MAX, color=RED, ls="--", lw=1.2)
    label(ax, OPEN_H + 0.1, QUEUE_MAX + 0.15, "ліміт черги: далі замовлення втрачаються",
          color=RED, ha="left")
    k = int(mean.argmax())
    ax.annotate(f"максимум {mean[k]:.1f} о {int(HOURS[k])}:{round(HOURS[k] % 1 * 60):02d}",
                (HOURS[k], mean[k]), xytext=(HOURS[k] + 0.6, mean[k] + 1.4),
                fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    time_axis(ax, [f"черга {table['Сер. черга'][i]:.1f}\nвтрати "
                   f"{table['Частка втрачених, %'][i]:.0f} %" for i in range(len(PERIODS))])
    ax.set_ylim(0, QUEUE_MAX + 1.8)
    ax.set_ylabel("замовлень у черзі")
    ax.legend(loc="upper right")
    _finish(fig, ax, f"Черга протягом дня при розкладі {_sched_name(r)} (100 прогонів)",
            name, show, pad=46)


# РИС. 4. ОДИН ПРОГІН: ЗАМОВЛЕННЯ НАРОСТАЮЧИМ ПІДСУМКОМ
def plot_single_run(r, name="s2_fig4_single_run.png", show=True):
    steps, s = r["single"]["steps"], r["single"]["summary"]
    end = HOURS[-1] + STEP / 60
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for col, color, text, dy in [("arrived", BLUE, "надійшло", 1.5), ("done", AQUA, "виконано", -1.5),
                                 ("lost", RED, "втрачено", 0)]:
        y = steps[col].cumsum()
        ax.step(np.append(HOURS, end), np.append(y, y.iloc[-1]), where="post", color=color,
                lw=2)
        ax.text(end + 0.1, y.iloc[-1] + dy, f"{text}: {int(y.iloc[-1])}", color=color,
                fontsize=9.5, va="center", fontweight="bold")
    gap_k = int((steps["arrived"].cumsum() - steps["done"].cumsum()).idxmax())
    a, d = steps["arrived"].cumsum()[gap_k], steps["done"].cumsum()[gap_k]
    ax.annotate("", xy=(HOURS[gap_k], d), xytext=(HOURS[gap_k], a),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
    ax.text(HOURS[gap_k] + 0.12, d - 1.5, f"{a - d} замовлень чекають\nабо печуться",
            color=INK2, fontsize=9.5, va="top")
    arrived = steps.groupby("period")["arrived"].sum()
    time_axis(ax, [f"надійшло +{arrived[i]}" for i in range(len(PERIODS))])
    ax.set_xlim(OPEN_H, CLOSE_H + 1.7)
    ax.set_ylabel("замовлень з початку дня")
    _finish(fig, ax, f"Один робочий день: прибуток {money(s['profit'])} грн "
                     f"(виручка {money(s['revenue'])} − печі {money(s['cost'])})",
            name, show, pad=34)


def plot_all(r, show=True):
    configure_style()
    plot_schedule(r, show=show)
    plot_profit(r, show=show)
    plot_queue(r, show=show)
    plot_single_run(r, show=show)


if __name__ == "__main__":
    with open("results2.pkl", "rb") as f:
        plot_all(pickle.load(f))
