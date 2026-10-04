"""
STEM-проєкт, варіант 4: оптимізація роботи піцерії.
Етап 1. Імітаційна модель обслуговування ОДНІЄЮ піччю (один канал).

Крок модельного часу - 5 хв, тривалість моделювання - 1 робочий день (09:00-22:00).
"""
import collections
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ПАРАМЕТРИ
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

ORDER_COLUMNS = ["num", "kind", "price", "cook", "period",
                 "t_arrive", "t_start", "t_finish", "lost"]

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


# ДАНІ
@dataclass
class Order:
    """Замовлення: лише дані, без логіки генерації чи обслуговування."""
    num: int
    kind: str
    price: float
    cook: float
    t_arrive: float
    period: int
    t_start: Optional[float] = None
    t_finish: Optional[float] = None
    lost: bool = False
    remaining: float = field(init=False)

    def __post_init__(self):
        self.remaining = self.cook


# ГЕНЕРАЦІЯ ЗАМОВЛЕНЬ
class OrderGenerator:
    """Відповідає лише за випадкові величини:
    чи надійшло замовлення і які в нього характеристики.
    """

    def __init__(self, rng, par):
        self.rng = rng
        self.par = par

    def arrival_probability(self, period):
        return min(1.0, PERIODS[period][3] * self.par["p_scale"])

    def order_arrives(self, period):
        """Чи надійшло замовлення на цьому кроці (Бернуллі)."""
        return self.rng.binomial(1, self.arrival_probability(period)) == 1

    def create(self, num, now, period):
        """Створює замовлення з випадковими типом, вартістю та часом приготування."""
        individual = self.rng.binomial(1, self.par["p_ind"])           # Бернуллі
        kind = "індивідуальна" if individual else "стандартна"
        price = round(self.rng.uniform(*PRICE[kind]))
        cook = round(self.rng.triangular(*COOK_TIME[kind]) * self.par["time_scale"], 2)
        return Order(num, kind, price, cook, now, period)


#  ПІЧ + ЧЕРГА
class OvenStation:
    """Піч та черга обмеженої довжини: правила прийому й обслуговування замовлень."""

    def __init__(self, queue_max):
        self.queue_max = queue_max
        self.queue = collections.deque()   # замовлення, що чекають (без того, що в печі)
        self.oven = None                   # замовлення, яке зараз у печі

    @property
    def queue_length(self):
        return len(self.queue)

    def _start_cooking(self, order, moment):
        order.t_start = moment
        self.oven = order

    def accept(self, order, now):
        """Приймає замовлення (в піч або в чергу). False - якщо черга заповнена."""
        if self.oven is None:
            self._start_cooking(order, now)
        elif len(self.queue) < self.queue_max:
            self.queue.append(order)
        else:
            return False
        return True

    def work(self, now):
        """Робота печі протягом одного кроку.
        Повертає (хвилин роботи, виконано замовлень).
        """
        time_left, clock, busy, done = STEP, now, 0.0, 0
        while time_left > 1e-9 and self.oven is not None:
            order = self.oven
            used = min(order.remaining, time_left)
            order.remaining -= used
            time_left -= used
            clock += used
            busy += used
            if order.remaining <= 1e-9:              # піца готова
                order.t_finish = clock
                self.oven = None
                done += 1
                if self.queue:                       # беремо наступне з черги
                    self._start_cooking(self.queue.popleft(), clock)
        return busy, done


# ПІДСУМКИ
def build_orders_frame(orders):
    """Список замовлень -> типізована таблиця з колонкою 'wait'."""
    frame = pd.DataFrame([{c: getattr(o, c) for c in ORDER_COLUMNS} for o in orders])
    if frame.empty:
        frame = pd.DataFrame(columns=ORDER_COLUMNS)
    frame = frame.astype({"price": float, "cook": float, "period": int, "t_arrive": float,
                          "t_start": float, "t_finish": float, "lost": bool})
    frame["wait"] = frame["t_start"] - frame["t_arrive"]
    return frame


def summarize_run(orders, steps, oven_cost):
    """Підсумкові показники одного прогону."""
    served = orders[orders["t_finish"].notna()]
    arrived = len(orders)
    lost = int(orders["lost"].sum())
    revenue = float(served["price"].sum())
    cost = oven_cost * (CLOSE_H - OPEN_H)
    return dict(
        arrived=arrived, lost=lost, accepted=arrived - lost, served=len(served),
        unfinished=arrived - lost - len(served),
        loss_share=lost / arrived if arrived else np.nan,
        revenue=revenue, cost=cost, profit=revenue - cost,
        utilization=steps["busy"].sum() / (N_STEPS * STEP),
        mean_queue=steps["queue"].mean(), max_queue=int(steps["queue"].max()),
        mean_wait=orders["wait"].mean() if orders["wait"].notna().any() else np.nan,
    )


def summarize_by_period(orders, steps):
    """Показники прогону в розрізі логічних періодів дня."""
    n_per = len(PERIODS)
    table = pd.DataFrame(dict(
        period=[p[0] for p in PERIODS],
        arrived=[int((orders["period"] == i).sum()) for i in range(n_per)],
        lost=[int(orders.loc[orders["period"] == i, "lost"].sum()) for i in range(n_per)],
        utilization=[steps.loc[steps["period"] == i, "busy"].sum()
                     / ((steps["period"] == i).sum() * STEP) for i in range(n_per)],
        mean_queue=[steps.loc[steps["period"] == i, "queue"].mean() for i in range(n_per)],
        mean_wait=[orders.loc[orders["period"] == i, "wait"].mean() for i in range(n_per)],
    ))
    table["loss_share"] = table["lost"] / table["arrived"].replace(0, np.nan)
    return table


def check_consistency(summary, orders, steps, queue_max):
    """Перевірка внутрішньої узгодженості прогону."""
    checks = [
        (summary["arrived"] == summary["lost"] + summary["accepted"], "arrived != lost + accepted"),
        (summary["accepted"] == summary["served"] + summary["unfinished"]
         and summary["unfinished"] >= 0, "accepted != served + unfinished"),
        (steps["queue"].max() <= queue_max, "черга перевищила ліміт"),
        ((orders["wait"].dropna() >= -1e-9).all(), "від'ємний час очікування"),
        (0 <= summary["utilization"] <= 1 + 1e-9, "завантаження поза [0, 1]"),
    ]
    done = orders[orders["t_finish"].notna()]
    checks.append((np.allclose(done["t_finish"] - done["t_start"], done["cook"], atol=1e-6),
                   "t_finish - t_start != cook"))
    for ok, message in checks:
        if not ok:
            raise AssertionError(message)


# МОДЕЛЬ
class Pizzeria:
    """Виконує один прогін: відлік часу, виклик генератора та печі, запис статистики кроків.

    Залежності (генератор, піч) можна підставити ззовні - наприклад, для тестів
    або для моделі з кількома печами.
    """

    def __init__(self, seed=None, generator=None, station=None, **params):
        self.par = {**DEFAULTS, **params}
        self.rng = np.random.default_rng(seed)
        self.generator = generator or OrderGenerator(self.rng, self.par)
        self.station = station or OvenStation(self.par["queue_max"])
        self.orders = []                   # усі замовлення, що надійшли
        self.steps = []                    # статистика по кроках

    # надходження замовлення на початку кроку: (надійшло, втрачено)
    def arrival(self, k, now):
        period = STEP_PERIOD[k]
        if not self.generator.order_arrives(period):
            return 0, 0
        order = self.generator.create(len(self.orders) + 1, now, period)
        self.orders.append(order)
        order.lost = not self.station.accept(order, now)
        return 1, int(order.lost)

    # один прогін на весь день
    def run(self):
        for k in range(N_STEPS):
            now = k * STEP
            arrived, lost = self.arrival(k, now)
            busy, done = self.station.work(now)
            self.steps.append(dict(k=k, time=fmt(now), period=STEP_PERIOD[k],
                                   arrived=arrived, lost=lost, done=done,
                                   queue=self.station.queue_length, busy=busy))
        return self.results()

    # підсумки прогону
    def results(self):
        orders = build_orders_frame(self.orders)
        steps = pd.DataFrame(self.steps)
        summary = summarize_run(orders, steps, self.par["oven_cost"])
        by_period = summarize_by_period(orders, steps)
        check_consistency(summary, orders, steps, self.par["queue_max"])
        return dict(summary=summary, by_period=by_period, orders=orders, steps=steps)


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
