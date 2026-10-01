"""
STEM-проєкт, варіант 4: оптимізація роботи піцерії.
Етап 1. Імітаційна модель обслуговування ОДНІЄЮ піччю (один канал).

Крок модельного часу - 5 хв, тривалість моделювання - 1 робочий день (09:00-22:00).
"""
import collections

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- ПАРАМЕТРИ
STEP = 5                                   # крок модельного часу, хв
OPEN_H, CLOSE_H = 9, 22                    # замовлення приймаються з 09:00 до 22:00
N_STEPS = (CLOSE_H - OPEN_H) * 60 // STEP  # кількість кроків за день = 156

# Логічні періоди дня: (назва, початок, кінець, p - ймовірність надходження
# замовлення за один крок часу)
PERIODS = [
    ("Ранковий спад", 9, 11, 0.15),
    ("Обідній пік", 11, 14, 0.60),
    ("Денний спад", 14, 17, 0.30),
    ("Вечірній пік", 17, 21, 0.70),
    ("Пізній вечір", 21, 22, 0.20),
]

P_INDIVIDUAL = 0.25                        # ймовірність "індивідуального" замовлення (Бернуллі)
COOK_TIME = {                              # час перебування піци в печі, хв: (min, mode, max)
    "стандартна": (15, 20, 30),            # трикутний розподіл
    "індивідуальна": (25, 35, 50),
}
PRICE = {                                  # вартість піци, грн: (min, max), рівномірний розподіл
    "стандартна": (180, 300),
    "індивідуальна": (320, 500),
}
OVEN_COST = 200.0                          # вартість утримання однієї ввімкненої печі, грн/год
QUEUE_MAX = 10                             # максимальна кількість замовлень у черзі

DEFAULTS = dict(p_scale=1.0, time_scale=1.0, p_ind=P_INDIVIDUAL,
                queue_max=QUEUE_MAX, oven_cost=OVEN_COST)

# Номер періоду для кожного кроку моделі
STEP_PERIOD = np.array([
    next(i for i, (_, h1, h2, _) in enumerate(PERIODS)
         if h1 <= OPEN_H + k * STEP / 60 < h2)
    for k in range(N_STEPS)
])


def fmt(minutes):
    """Хвилини від 09:00 -> рядок 'ГГ:ХХ'."""
    m = int(round(minutes))
    return f"{OPEN_H + m // 60:02d}:{m % 60:02d}"


# ------------------------------------------------------------------ МОДЕЛЬ
class Order:
    """Замовлення: усі випадкові характеристики генеруються в момент надходження."""

    def __init__(self, num, kind, price, cook, t_arrive, period):
        self.num, self.kind, self.price, self.cook = num, kind, price, cook
        self.t_arrive, self.period = t_arrive, period
        self.t_start = None
        self.t_finish = None
        self.remaining = cook
        self.lost = False


class Pizzeria:
    """Піцерія з однією піччю та спільною чергою обмеженої довжини."""

    def __init__(self, seed=None, **params):
        self.par = {**DEFAULTS, **params}
        self.rng = np.random.default_rng(seed)
        self.queue = collections.deque()   # замовлення, що чекають (без того, що в печі)
        self.oven = None                   # замовлення, яке зараз у печі
        self.orders = []                   # усі замовлення, що надійшли
        self.steps = []                    # статистика по кроках

    # --- генерація випадкових величин
    def new_order(self, now, period):
        individual = self.rng.binomial(1, self.par["p_ind"])           # Бернуллі
        kind = "індивідуальна" if individual else "стандартна"
        price = round(self.rng.uniform(*PRICE[kind]))
        cook = round(self.rng.triangular(*COOK_TIME[kind]) * self.par["time_scale"], 2)
        order = Order(len(self.orders) + 1, kind, price, cook, now, period)
        self.orders.append(order)
        return order

    # --- початок приготування
    def start_cooking(self, order, moment):
        order.t_start = moment
        self.oven = order

    # --- надходження замовлення на початку кроку
    def arrival(self, k, now):
        period = STEP_PERIOD[k]
        p = min(1.0, PERIODS[period][3] * self.par["p_scale"])
        if self.rng.binomial(1, p) == 0:                               # Бернуллі
            return 0, 0
        order = self.new_order(now, period)
        if self.oven is None:                                          # піч вільна
            self.start_cooking(order, now)
        elif len(self.queue) < self.par["queue_max"]:                  # є місце в черзі
            self.queue.append(order)
        else:                                                          # черга заповнена
            order.lost = True
            return 1, 1
        return 1, 0

    # --- робота печі протягом кроку (STEP хв)
    def service(self, now):
        time_left, clock, busy, done = STEP, now, 0.0, 0
        while time_left > 1e-9 and self.oven is not None:
            order = self.oven
            used = min(order.remaining, time_left)
            order.remaining -= used
            time_left -= used
            clock += used
            busy += used
            if order.remaining <= 1e-9:                                # піца готова
                order.t_finish = clock
                self.oven = None
                done += 1
                if self.queue:                                         # беремо наступне з черги
                    self.start_cooking(self.queue.popleft(), clock)
        return busy, done

    # --- один прогін на весь день
    def run(self):
        for k in range(N_STEPS):
            now = k * STEP
            arrived, lost = self.arrival(k, now)
            busy, done = self.service(now)
            self.steps.append(dict(k=k, time=fmt(now), period=STEP_PERIOD[k],
                                   arrived=arrived, lost=lost, done=done,
                                   queue=len(self.queue), busy=busy))
        return self.results()

    # --- підсумки прогону
    def results(self):
        orders = pd.DataFrame([dict(
            num=o.num, kind=o.kind, price=o.price, cook=o.cook, period=o.period,
            t_arrive=o.t_arrive, t_start=o.t_start, t_finish=o.t_finish, lost=o.lost)
            for o in self.orders])
        steps = pd.DataFrame(self.steps)
        if orders.empty:
            orders = pd.DataFrame(columns=["num", "kind", "price", "cook", "period",
                                           "t_arrive", "t_start", "t_finish", "lost"])
        orders = orders.astype({"price": float, "cook": float, "period": int, "t_arrive": float,
                                "t_start": float, "t_finish": float, "lost": bool})
        orders["wait"] = orders["t_start"] - orders["t_arrive"]
        served = orders[orders["t_finish"].notna()]
        arrived = len(orders)
        lost = int(orders["lost"].sum())
        busy_total = steps["busy"].sum()
        revenue = float(served["price"].sum())
        cost = self.par["oven_cost"] * (CLOSE_H - OPEN_H)
        summary = dict(
            arrived=arrived, lost=lost, accepted=arrived - lost, served=len(served),
            unfinished=arrived - lost - len(served),
            loss_share=lost / arrived if arrived else np.nan,
            revenue=revenue, cost=cost, profit=revenue - cost,
            utilization=busy_total / (N_STEPS * STEP),
            mean_queue=steps["queue"].mean(), max_queue=int(steps["queue"].max()),
            mean_wait=orders["wait"].mean() if orders["wait"].notna().any() else np.nan,
        )
        n_per = len(PERIODS)
        by_period = pd.DataFrame(dict(
            period=[p[0] for p in PERIODS],
            arrived=[int((orders["period"] == i).sum()) for i in range(n_per)],
            lost=[int(orders.loc[orders["period"] == i, "lost"].sum()) for i in range(n_per)],
            utilization=[steps.loc[steps["period"] == i, "busy"].sum()
                         / ((steps["period"] == i).sum() * STEP) for i in range(n_per)],
            mean_queue=[steps.loc[steps["period"] == i, "queue"].mean() for i in range(n_per)],
            mean_wait=[orders.loc[orders["period"] == i, "wait"].mean() for i in range(n_per)],
        ))
        by_period["loss_share"] = by_period["lost"] / by_period["arrived"].replace(0, np.nan)
        self.check(summary, orders, steps)
        return dict(summary=summary, by_period=by_period, orders=orders, steps=steps)

    # --- перевірка внутрішньої узгодженості прогону
    def check(self, s, orders, steps):
        assert s["arrived"] == s["lost"] + s["accepted"]
        assert s["accepted"] == s["served"] + s["unfinished"] and s["unfinished"] >= 0
        assert steps["queue"].max() <= self.par["queue_max"]
        assert (orders["wait"].dropna() >= -1e-9).all()
        assert 0 <= s["utilization"] <= 1 + 1e-9
        done = orders[orders["t_finish"].notna()]
        assert np.allclose(done["t_finish"] - done["t_start"], done["cook"], atol=1e-6)


def run_many(n_runs=100, seed0=1, **params):
    """Серія незалежних прогонів: повертає таблицю підсумків та допоміжні масиви."""
    summaries, by_period, queue_curves, waits, orders_all = [], [], [], [], []
    for r in range(n_runs):
        res = Pizzeria(seed=seed0 + r, **params).run()
        summaries.append(res["summary"])
        by_period.append(res["by_period"])
        queue_curves.append(res["steps"]["queue"].to_numpy())
        waits.append(res["orders"]["wait"].dropna().to_numpy())
        orders_all.append(res["orders"])
    return dict(summary=pd.DataFrame(summaries), by_period=by_period,
                queue_curves=np.array(queue_curves), waits=np.concatenate(waits),
                orders=pd.concat(orders_all, ignore_index=True))
