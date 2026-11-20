"""GTA VI trophy rarity forecaster (make.fun: gta6-trophy-rarity-forecast-01).

What the program does
---------------------
For every case file handed to it, it predicts, for each of the game's
PlayStation 5 trophies, five quantiles of that trophy's earn rate (the
percentage of tracked players who hold the trophy) at the case's target
moment.

How it thinks about the numbers
-------------------------------
A printed rate is not a free-floating statistic.  It is a ratio:

    rate = 100 * earners / players_tracked

where "earners" is how many tracked players hold the trophy and
"players_tracked" is how many players the tracker follows for the game.
The tracker does not update that page continuously.  It refreshes in
jumps: the tracked-player count sits flat for a while, then leaps when
fresh profiles are folded in.  At a leap the denominator grows much
faster than the numerator, so most rates DROP at a refresh even though
nobody has lost a trophy.  Between refreshes the opposite drift wins:
earners keep accumulating while the denominator is frozen, so rates
creep up.

Two consequences drive the design:

* Earners never truly fall.  The measured earner count (rate * players
  / 100) is roughly continuous across a refresh while the player count
  jumps, so earners and players are forecast separately and only then
  divided back into a rate.
* A quiet history is not a safe history.  A case cut before the first
  refresh shows flat players and flat rates, yet the target sits on the
  far side of refreshes nobody has seen yet.  The spread therefore
  carries a refresh-risk prior even when no refresh was observed.

Everything below is standard library only, fully deterministic, and
fast: a handful of least-squares fits per trophy.
"""

import json
import math
import os
import sys
from pathlib import Path
from statistics import NormalDist

# The grader launches this file with flags that keep its own directory
# off the import path, so put that directory back at the front in case
# helper modules are ever added next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
RATE_FLOOR = 0.01
RATE_CEIL = 99.99

# Spread ingredients, all in logit units.
WALK_PER_SQRT_DAY = 0.45      # plain random-walk drift of a rate
PRIOR_JUMP_LOGIT = 1.0        # assumed size of one unseen refresh jump
PRIOR_REFRESHES_PER_DAY = 2.0  # assumed refresh cadence when none seen
SIGMA_MIN = 0.35
SIGMA_MAX = 2.3
SIGMA_DOWN_FACTOR = 1.15      # dilution makes downward misses likelier
SIGMA_UP_FACTOR = 0.92

EARNER_DAMPING = 0.7          # shrink fitted earner growth past the data
MEDIAN_BLEND_NEW = 0.75       # weight on the decomposition median
MAX_PLAYER_GROWTH = 25.0      # cap on forecast growth of tracked players


def clamp(value, low, high):
    return low if value < low else high if value > high else value


def logit_of_rate(rate_percent):
    """Log-odds of a percentage rate, clamped away from 0 and 100."""
    p = clamp(rate_percent, RATE_FLOOR, RATE_CEIL) / 100.0
    return math.log(p / (1.0 - p))


def rate_from_logit(logit_value):
    """Inverse of logit_of_rate; returns a percentage."""
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    exp_term = math.exp(logit_value)
    return 100.0 * exp_term / (1.0 + exp_term)


def round4(value):
    """Round a non-negative number to 4 decimals, halves away from zero."""
    return math.floor(value * 10000.0 + 0.5) / 10000.0


def parse_moment(text):
    """Parse an ISO-8601 timestamp into epoch seconds (None if absent)."""
    if not text:
        return None
    try:
        from datetime import datetime

        cleaned = str(text).replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned).timestamp()
    except (ValueError, TypeError):
        return None


def fit_line(xs, ys):
    """Least-squares straight line; returns (intercept, slope) or None."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return (mean_y - slope * mean_x, slope)


def line_residual_rms(xs, ys, fit):
    """Root-mean-square residual of a fitted line over its own points."""
    if fit is None or not xs:
        return 0.0
    intercept, slope = fit
    total = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return math.sqrt(total / len(xs))


def forecast_players(hours, players, target_hour, last_players):
    """Project the tracked-player count to the target hour.

    Player growth on a log-time axis is close to linear in the launch
    window, so ln(players) is fitted against ln(1 + hours).  A fit that
    is missing, degenerate or points downhill is ignored: the tracked
    cohort does not shrink, so the last count is the floor.
    """
    # Compress runs of identical counts to their final read first.
    # Between refreshes the page is re-read many times with the same
    # player count, and feeding every repeat into the fit would let a
    # long flat stretch outvote the jumps that carry the information.
    points = []
    for h, p in zip(hours, players):
        if not p or p <= 0:
            continue
        if points and points[-1][1] == p:
            points[-1] = (h, p)
        else:
            points.append((h, p))
    if len(points) >= 2:
        xs = [math.log1p(h) for h, _ in points]
        ys = [math.log(p) for _, p in points]
        fit = fit_line(xs, ys)
        if fit is not None and fit[1] > 0.0:
            guess = math.exp(fit[0] + fit[1] * math.log1p(target_hour))
            return clamp(guess, last_players, last_players * MAX_PLAYER_GROWTH)
    return float(last_players)


def forecast_earners(hours, earners, target_hour, last_earners, players_pred):
    """Project one trophy's earner count to the target hour.

    The same log-time line is fitted to ln(earners + 0.5), but the part
    of the growth that lies beyond the final observation is shrunk by
    EARNER_DAMPING: launch-week enthusiasm fades, and an undamped line
    would promise more finishers than the week can deliver.  Returns
    (earners_pred, residual_rms_of_fit).
    """
    points = [(h, e) for h, e in zip(hours, earners) if e is not None and e > 0]
    if len(points) < 2:
        return float(last_earners), 0.0
    xs = [math.log1p(h) for h, _ in points]
    ys = [math.log(e + 0.5) for _, e in points]
    fit = fit_line(xs, ys)
    if fit is None:
        return float(last_earners), 0.0
    rms = line_residual_rms(xs, ys, fit)
    intercept, slope = fit
    x_last = xs[-1]
    x_target = math.log1p(target_hour)
    y_pred = intercept + slope * x_last + EARNER_DAMPING * slope * (x_target - x_last)
    guess = math.exp(y_pred) - 0.5
    guess = clamp(guess, 0.6 * last_earners, players_pred)
    return guess, rms


def refresh_spread(hours, players, rates, horizon_days):
    """Logit-scale spread owed to tracker refreshes.

    Where the player count changed between two reads, the rate jump
    across that boundary is measured directly.  Those jumps, scaled by
    how many refreshes the horizon is expected to contain, give the
    refresh term.  When history contains no refresh at all, a prior
    stands in for the jumps that have simply not happened yet.
    """
    change_idx = [i for i in range(1, len(players))
                  if players[i] is not None and players[i - 1] is not None
                  and players[i] != players[i - 1]]
    jumps = []
    for i in change_idx:
        before, after = rates[i - 1], rates[i]
        if before is not None and after is not None:
            jumps.append(abs(logit_of_rate(after) - logit_of_rate(before)))
    span_days = 0.0
    if len(hours) >= 2:
        span_days = (hours[-1] - hours[0]) / 24.0
    if change_idx and span_days > 0.0:
        frequency = len(change_idx) / span_days
        expected = frequency * horizon_days
    elif change_idx:
        expected = float(len(change_idx))
    else:
        expected = PRIOR_REFRESHES_PER_DAY * horizon_days
    mean_jump = (sum(jumps) / len(jumps)) if jumps else PRIOR_JUMP_LOGIT
    return mean_jump * math.sqrt(max(1.0, expected)) * 0.5


def forecast_trophy(hours, players, rates, target_hour, horizon_days, levels):
    """Five quantile rates for a single trophy."""
    # Last usable observation of this trophy.
    last_rate = None
    last_earners = 0.0
    last_players = None
    for p, r in zip(players, rates):
        if p is not None:
            last_players = p
        if r is not None and last_players:
            last_rate = r
            last_earners = r * last_players / 100.0
    if last_players is None:
        last_players = 0.0

    earners_series = [
        (r * p / 100.0) if (r is not None and p) else None
        for p, r in zip(players, rates)
    ]

    players_pred = forecast_players(hours, players, target_hour, last_players)
    earners_pred, fit_rms = forecast_earners(
        hours,
        [e if e is not None else 0.0 for e in earners_series],
        target_hour,
        last_earners,
        players_pred,
    )

    if players_pred > 0 and last_rate is not None:
        decomposed = clamp(100.0 * earners_pred / players_pred,
                           RATE_FLOOR, RATE_CEIL)
        median_logit = (
            MEDIAN_BLEND_NEW * logit_of_rate(decomposed)
            + (1.0 - MEDIAN_BLEND_NEW) * logit_of_rate(last_rate)
        )
    elif last_rate is not None:
        median_logit = logit_of_rate(last_rate)
    else:
        median_logit = logit_of_rate(1.0)

    walk = WALK_PER_SQRT_DAY * math.sqrt(max(horizon_days, 0.0))
    refresh = refresh_spread(hours, players, rates, horizon_days)
    residual = max(fit_rms, 0.1)
    sigma = clamp(math.sqrt(walk * walk + refresh * refresh + residual * residual),
                  SIGMA_MIN, SIGMA_MAX)

    normal = NormalDist()
    quantiles = []
    for level in levels:
        try:
            z = normal.inv_cdf(level)
        except Exception:
            z = 0.0
        side = SIGMA_DOWN_FACTOR if z < 0 else SIGMA_UP_FACTOR if z > 0 else 1.0
        value = rate_from_logit(median_logit + z * sigma * side)
        quantiles.append(round4(clamp(value, RATE_FLOOR, 100.0)))
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return quantiles


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    levels = case.get("levels") or DEFAULT_LEVELS

    first_ts = None
    for entry in history:
        first_ts = parse_moment(entry.get("at") or entry.get("slot"))
        if first_ts is not None:
            break
    if first_ts is None:
        first_ts = parse_moment(case.get("at")) or 0.0

    hours = []
    players = []
    known_players = None
    for entry in history:
        ts = parse_moment(entry.get("at") or entry.get("slot"))
        hours.append(((ts - first_ts) / 3600.0) if ts is not None else 0.0)
        count = entry.get("players_tracked")
        if count is not None:
            known_players = count
        players.append(known_players)

    target_ts = parse_moment(case.get("target"))
    if target_ts is not None:
        target_hour = max((target_ts - first_ts) / 3600.0, 0.0)
    else:
        horizon_min = case.get("horizon_minutes") or 0
        base_ts = parse_moment(case.get("at")) or first_ts
        target_hour = max((base_ts - first_ts) / 3600.0 + horizon_min / 60.0, 0.0)
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0
    if horizon_days <= 0 and hours:
        horizon_days = max(target_hour - hours[-1], 0.0) / 24.0

    rows = []
    for idx, trophy in enumerate(trophies):
        rates = []
        for entry in history:
            series = entry.get("rates") or []
            rates.append(series[idx] if idx < len(series) else None)
        q = forecast_trophy(hours, players, rates, target_hour,
                            horizon_days, levels)
        rows.append({"id": trophy.get("id", idx + 1), "q": q})
    return {"trophies": rows}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not in_dir.is_dir():
        return
    for path in sorted(in_dir.glob("*.in")):
        with path.open("r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        case_id = case.get("id") or path.stem
        out_path = out_dir / (str(case_id) + ".out")
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"))


if __name__ == "__main__":
    main()
