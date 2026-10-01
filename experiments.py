"""
STEM-проєкт, варіант 4. Етап 1: експерименти з моделлю однієї печі.
Запуск:  python experiments.py
"""
import pickle
import warnings
 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
warnings.filterwarnings("ignore", category=RuntimeWarning)
 
from pizza_model import (PERIODS, STEP, N_STEPS, OPEN_H, CLOSE_H, Pizzeria, run_many, fmt)
 
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
                     "axes.grid": True, "grid.color": "#e3e3e3", "axes.axisbelow": True})
BLUE, ORANGE, RED, GREEN, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#999999"
HOURS = OPEN_H + np.arange(N_STEPS) * STEP / 60            # час початку кожного кроку, год
 
 
def shade_periods(ax, labels=True):
    """Позначає на графіку межі логічних періодів дня."""
    short = ["ранок", "обід", "день", "вечір", "пізно"]
    for i, (_, h1, h2, _) in enumerate(PERIODS):
        if i % 2 == 1:
            ax.axvspan(h1, h2, color=ORANGE, alpha=0.10, lw=0)
        if labels:
            ax.text((h1 + h2) / 2, 1.01, short[i], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10)
    ax.set_xlim(OPEN_H, CLOSE_H)
    ax.set_xticks(range(OPEN_H, CLOSE_H + 1, 2))
 
 
# ============================================================ 1. ОДИН ПРОГІН
single = Pizzeria(seed=42).run()
orders, steps = single["orders"], single["steps"]
 
# --- фрагмент логу замовлень (перші 12 рядків)
log = orders.head(12).copy()
log_table = pd.DataFrame({
    "№": log["num"], "Тип": log["kind"], "Вартість, грн": log["price"],
    "Час на піч, хв": log["cook"].round(1),
    "Надійшло": [fmt(t) for t in log["t_arrive"]],
    "Початок": [fmt(t) if pd.notna(t) else "—" for t in log["t_start"]],
    "Кінець": [fmt(t) if pd.notna(t) else "—" for t in log["t_finish"]],
    "Очікування, хв": log["wait"].round(1),
    "Статус": np.where(log["lost"], "втрачено", np.where(log["t_finish"].notna(), "виконано", "в роботі")),
})
# --- фрагмент логу кроків (вечірній пік, 17:00-17:30)
k0 = (17 - OPEN_H) * 60 // STEP
step_log = steps.iloc[k0:k0 + 7][["time", "arrived", "lost", "done", "queue", "busy"]].copy()
step_log.columns = ["Час (початок кроку)", "Надійшло", "Втрачено", "Виконано", "Черга (кінець кроку)", "Піч працювала, хв"]
 
# --- рис. 1: динаміка одного прогону
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.90))
ax[0].step(HOURS, steps["queue"], where="post", color=BLUE, lw=1.2)
ax[0].axhline(10, color=RED, ls="--", lw=0.8)
ax[0].text(OPEN_H + 0.1, 10.25, "ліміт черги = 10", color=RED, fontsize=10)
shade_periods(ax[0])
ax[0].set_ylim(0, 11.8)
ax[0].set_xlabel("Час доби, год"); ax[0].set_ylabel("Довжина черги, замовлень")
ax[0].set_title("а) Довжина черги", y=1.1)
cum_arr = steps["arrived"].cumsum(); cum_lost = steps["lost"].cumsum(); cum_done = steps["done"].cumsum()
ax[1].step(HOURS, cum_arr, where="post", color=BLUE, label="надійшло")
ax[1].step(HOURS, cum_done, where="post", color=GREEN, label="виконано")
ax[1].step(HOURS, cum_lost, where="post", color=RED, label="втрачено")
shade_periods(ax[1])
ax[1].set_xlabel("Час доби, год"); ax[1].set_ylabel("Кумулятивна кількість, шт.")
ax[1].set_title("б) Замовлення наростаючим підсумком", y=1.1)
ax[1].legend(loc="upper left", fontsize=10, frameon=False)
plt.tight_layout(); plt.savefig("fig1_single_dynamics.png", dpi=150); plt.show()
 
# --- рис. 2: гістограми одного прогону
fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.45))
ax[0].hist(orders["cook"], bins=10, color=BLUE, edgecolor="white")
ax[0].set_title("а) Час приготування"); ax[0].set_xlabel("хв"); ax[0].set_ylabel("Кількість замовлень")
ax[1].hist(orders["price"], bins=10, color=ORANGE, edgecolor="white")
ax[1].set_title("б) Вартість піци"); ax[1].set_xlabel("грн")
ax[2].hist(orders["wait"].dropna(), bins=10, color=GREEN, edgecolor="white")
ax[2].set_title("в) Очікування до початку"); ax[2].set_xlabel("хв")
plt.tight_layout(); plt.savefig("fig2_single_hist.png", dpi=150); plt.show()
 
# ================================================================ 2. 100 ПРОГОНІВ
base = run_many(100, seed0=1)
S = base["summary"]
names = {
    "arrived": "Надійшло замовлень, шт.", "lost": "Втрачено замовлень, шт.",
    "loss_share": "Частка втрачених, %", "accepted": "Прийнято до виконання, шт.",
    "served": "Виконано до 22:00, шт.", "unfinished": "Не виконано до 22:00, шт.",
    "revenue": "Виручка, грн", "cost": "Витрати на піч, грн", "profit": "Чистий прибуток, грн",
    "utilization": "Завантаження печі, %", "mean_queue": "Середня довжина черги, шт.",
    "max_queue": "Максимальна довжина черги, шт.", "mean_wait": "Середнє очікування до початку, хв",
}
scale = {"loss_share": 100, "utilization": 100}
rows = []
for key, title in names.items():
    v = S[key].astype(float) * scale.get(key, 1)
    rows.append([title, v.mean(), v.std(ddof=1), 1.96 * v.std(ddof=1) / np.sqrt(len(v)), v.min(), v.max()])
summary_table = pd.DataFrame(rows, columns=["Показник", "Середнє", "Станд. відхилення", "±95% ДІ", "Мін", "Макс"])
 
# --- по періодах (середнє за прогонами)
bp = base["by_period"]
per_rows = []
for i, (name, h1, h2, p) in enumerate(PERIODS):
    cols = {c: np.array([df.loc[i, c] for df in bp], dtype=float) for c in
            ["arrived", "lost", "loss_share", "utilization", "mean_queue", "mean_wait"]}
    per_rows.append([name, f"{h1:02d}:00–{h2:02d}:00", p, np.mean(cols["arrived"]), np.mean(cols["lost"]),
                     np.nanmean(cols["loss_share"]) * 100, np.mean(cols["utilization"]) * 100,
                     np.mean(cols["mean_queue"]), np.nanmean(cols["mean_wait"])])
period_table = pd.DataFrame(per_rows, columns=["Період", "Час", "p", "Надійшло", "Втрачено",
                                               "Частка втрачених, %", "Завантаження, %",
                                               "Сер. черга", "Сер. очікування, хв"])
period_loss_std = [np.nanstd([df.loc[i, "loss_share"] for df in bp], ddof=1) * 100 for i in range(len(PERIODS))]
 
# --- рис. 3: розподіл підсумків за 100 прогонами
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.60))
ax[0].hist(S["profit"], bins=12, color=BLUE, edgecolor="white")
ax[0].axvline(S["profit"].mean(), color=RED, ls="--", lw=1, label=f"середнє = {S['profit'].mean():.0f}")
ax[0].set_title("а) Чистий прибуток за день"); ax[0].set_xlabel("грн"); ax[0].set_ylabel("Кількість прогонів")
ax[0].legend(fontsize=10, frameon=False)
ax[1].hist(S["loss_share"] * 100, bins=12, color=ORANGE, edgecolor="white")
ax[1].axvline(S["loss_share"].mean() * 100, color=RED, ls="--", lw=1,
              label=f"середнє = {S['loss_share'].mean() * 100:.1f}%")
ax[1].set_title("б) Частка втрачених замовлень"); ax[1].set_xlabel("%")
ax[1].legend(fontsize=10, frameon=False)
plt.tight_layout(); plt.savefig("fig3_hist_100.png", dpi=150); plt.show()
 
# --- рис. 4: по періодах та протягом дня
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.90))
x = np.arange(len(PERIODS))
ax[0].bar(x, period_table["Частка втрачених, %"], yerr=period_loss_std, color=BLUE, capsize=3,
          error_kw=dict(lw=0.8))
ax[0].set_xticks(x); ax[0].set_xticklabels(["ранок", "обід", "день", "вечір", "пізно"])
ax[0].set_ylabel("Частка втрачених, %"); ax[0].set_ylim(0, 105)
ax[0].set_title("а) Втрати за періодами (середнє ±σ)")
qc = base["queue_curves"]
m, sd = qc.mean(axis=0), qc.std(axis=0, ddof=1)
ax[1].plot(HOURS, m, color=BLUE, lw=1.2, label="середнє")
ax[1].fill_between(HOURS, np.clip(m - sd, 0, None), np.minimum(m + sd, 10), color=BLUE, alpha=0.2, label="±σ")
ax[1].axhline(10, color=RED, ls="--", lw=0.8)
shade_periods(ax[1]); ax[1].set_ylim(0, 11.8)
ax[1].set_xlabel("Час доби, год"); ax[1].set_ylabel("Довжина черги, шт.")
ax[1].set_title("б) Довжина черги протягом дня", y=1.1)
ax[1].legend(loc="lower right", fontsize=10, frameon=False)
plt.tight_layout(); plt.savefig("fig4_period_100.png", dpi=150); plt.show()
 
# ===================================================== 3. ПЕРЕВІРКА АДЕКВАТНОСТІ
allo = base["orders"]
theory = dict(
    arrived=sum(p * (h2 - h1) * 60 / STEP for _, h1, h2, p in PERIODS),
    cook=0.75 * (15 + 20 + 30) / 3 + 0.25 * (25 + 35 + 50) / 3,
    price=0.75 * (180 + 300) / 2 + 0.25 * (320 + 500) / 2, p_ind=0.25)
generator_check = pd.DataFrame([
    ["Надійшло замовлень за день, шт.", theory["arrived"], S["arrived"].mean()],
    ["Частка індивідуальних замовлень", theory["p_ind"], (allo["kind"] == "індивідуальна").mean()],
    ["Середній час приготування, хв", theory["cook"], allo["cook"].mean()],
    ["Середня вартість піци, грн", theory["price"], allo["price"].mean()],
], columns=["Величина", "Теоретичне значення", "Модель (100 прогонів)"])
 
 
def sweep(param, values):
    out = []
    for v in values:
        r = run_many(100, seed0=1, **{param: v})["summary"]
        out.append(dict(value=v, arrived=r["arrived"].mean(), served=r["served"].mean(),
                        loss=r["loss_share"].mean() * 100, util=r["utilization"].mean() * 100,
                        queue=r["mean_queue"].mean(), wait=r["mean_wait"].mean(), profit=r["profit"].mean(),
                        unfinished=r["unfinished"].mean()))
    return pd.DataFrame(out)
 
 
sweeps = {
    "p_scale": sweep("p_scale", [0, 0.25, 0.5, 1.0, 1.5, 2.0]),
    "time_scale": sweep("time_scale", [0.5, 0.75, 1.0, 1.5, 2.0]),
    "queue_max": sweep("queue_max", [0, 3, 5, 10, 20]),
    "p_ind": sweep("p_ind", [0.0, 0.25, 0.5, 1.0]),
}
 
if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(single["summary"]); print(single["by_period"]); print(log_table); print(step_log)
    print(summary_table.round(2)); print(period_table.round(2)); print(generator_check.round(3))
    for k, v in sweeps.items():
        print(k); print(v.round(2))
    with open("results.pkl", "wb") as f:
        pickle.dump(dict(single=single, log_table=log_table, step_log=step_log, summary_table=summary_table,
                         period_table=period_table, generator_check=generator_check, sweeps=sweeps,
                         theory=theory), f)
 