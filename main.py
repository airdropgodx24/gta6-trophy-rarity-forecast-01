import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob
import json
import math
import statistics
from datetime import datetime, timezone


def parse_time(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def clamp(value, lo, hi):
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def logit_of_rate(rate_percent):
    p = clamp(rate_percent, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def rate_from_logit(logit_value):
    if logit_value >= 0:
        p = 1.0 / (1.0 + math.exp(-logit_value))
    else:
        e = math.exp(logit_value)
        p = e / (1.0 + e)
    return p * 100.0


def fit_line(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return ys[0], 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    slope = sxy / sxx
    intercept = my - slope * mx
    residuals = [ys[i] - (intercept + slope * xs[i]) for i in range(n)]
    rms = math.sqrt(sum(r * r for r in residuals) / n)
    return intercept, slope, rms


def process_case(case):
    trophies = case.get("trophies") or []
    trophy_ids = []
    for t in trophies:
        if isinstance(t, dict):
            trophy_ids.append(t.get("id"))
        else:
            trophy_ids.append(t)
    n_troph = len(trophy_ids)
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    history = case.get("history") or []

    case_at = parse_time(case.get("at"))
    target_at = parse_time(case.get("target"))

    horizon_minutes = case.get("horizon_minutes")
    if horizon_minutes is not None:
        try:
            horizon_days = float(horizon_minutes) / 1440.0
        except (TypeError, ValueError):
            horizon_days = 0.0
    elif case_at is not None and target_at is not None:
        horizon_days = (target_at - case_at).total_seconds() / 86400.0
    else:
        horizon_days = 0.0
    if horizon_days < 0:
        horizon_days = 0.0
    horizon_hours = horizon_days * 24.0

    hist_times = []
    hist_players = []
    hist_rates = []
    for row in history:
        if not isinstance(row, dict):
            continue
        t = parse_time(row.get("at") or row.get("slot"))
        hist_times.append(t)
        try:
            players = float(row.get("players_tracked"))
        except (TypeError, ValueError):
            players = 0.0
        hist_players.append(players)
        hist_rates.append(row.get("rates") or [])

    if not hist_times or hist_times[0] is None:
        first_time = case_at
    else:
        first_time = hist_times[0]

    def hours_since_first(t):
        if t is None or first_time is None:
            return 0.0
        return (t - first_time).total_seconds() / 3600.0

    def log_time(t):
        h = hours_since_first(t)
        if h < 0:
            h = 0.0
        return math.log(1.0 + h)

    if target_at is not None:
        x_target = log_time(target_at)
    elif first_time is not None and hist_times:
        last_t = hist_times[-1]
        base_h = hours_since_first(last_t) if last_t is not None else 0.0
        x_target = math.log(1.0 + max(0.0, base_h + horizon_hours))
    else:
        x_target = math.log(1.0 + max(0.0, horizon_hours))

    x_last = log_time(hist_times[-1]) if hist_times else 0.0
    last_players = hist_players[-1] if hist_players else 0.0

    comp_x = []
    comp_y = []
    if hist_players:
        run_last_idx = []
        i = 0
        n_hist = len(hist_players)
        while i < n_hist:
            j = i
            while j + 1 < n_hist and hist_players[j + 1] == hist_players[i]:
                j += 1
            run_last_idx.append(j)
            i = j + 1
        for idx in run_last_idx:
            if hist_players[idx] > 0:
                comp_x.append(log_time(hist_times[idx]))
                comp_y.append(math.log(hist_players[idx]))
    if comp_x:
        p_intercept, p_slope, _p_rms = fit_line(comp_x, comp_y)
        players_pred_raw = math.exp(p_intercept + p_slope * x_target)
    else:
        players_pred_raw = last_players
    if last_players > 0:
        players_pred = clamp(players_pred_raw, last_players, 25.0 * last_players)
    else:
        players_pred = max(players_pred_raw, 0.0)
    if players_pred <= 0:
        players_pred = last_players if last_players > 0 else 1.0

    refresh_indices = []
    for i in range(1, len(hist_players)):
        if hist_players[i] != hist_players[i - 1]:
            refresh_indices.append(i)
    n_refresh = len(refresh_indices)
    if n_refresh > 0 and hist_times and hist_times[0] is not None and hist_times[-1] is not None:
        duration_days = (hist_times[-1] - hist_times[0]).total_seconds() / 86400.0
        if duration_days > 1e-9:
            refresh_per_day = n_refresh / duration_days
        else:
            refresh_per_day = 2.0
        expected_refreshes = refresh_per_day * horizon_days
    else:
        expected_refreshes = 2.0 * horizon_days
    if expected_refreshes < 0:
        expected_refreshes = 0.0

    global_jumps = []
    for i in refresh_indices:
        prev_rates = hist_rates[i - 1] if i - 1 < len(hist_rates) else []
        cur_rates = hist_rates[i] if i < len(hist_rates) else []
        for k in range(min(len(prev_rates), len(cur_rates), n_troph)):
            a = prev_rates[k]
            b = cur_rates[k]
            if a is None or b is None:
                continue
            try:
                af = float(a)
                bf = float(b)
            except (TypeError, ValueError):
                continue
            if 0.0 < af < 100.0 and 0.0 < bf < 100.0:
                global_jumps.append(abs(logit_of_rate(bf) - logit_of_rate(af)))
    global_mean_jump = sum(global_jumps) / len(global_jumps) if global_jumps else None

    normal = statistics.NormalDist()
    out_trophies = []

    for ti in range(n_troph):
        tid = trophy_ids[ti]

        rates_series = []
        for r in hist_rates:
            if ti < len(r):
                v = r[ti]
                if v is None:
                    rates_series.append(None)
                else:
                    try:
                        rates_series.append(float(v))
                    except (TypeError, ValueError):
                        rates_series.append(None)
            else:
                rates_series.append(None)

        last_rate = None
        for v in reversed(rates_series):
            if v is not None:
                last_rate = v
                break

        if last_rate is not None and last_rate >= 100.0:
            out_trophies.append({"id": tid, "q": [100.0 for _ in levels]})
            continue

        earners_series = []
        for i in range(len(hist_players)):
            r = rates_series[i] if i < len(rates_series) else None
            if r is None or hist_players[i] <= 0:
                earners_series.append(None)
            else:
                earners_series.append(r * hist_players[i] / 100.0)

        last_earners = None
        for v in reversed(earners_series):
            if v is not None:
                last_earners = v
                break
        if last_earners is None:
            last_earners = 0.0

        e_xs = []
        e_ys = []
        for i in range(len(earners_series)):
            e = earners_series[i]
            if e is not None and e > 0:
                e_xs.append(log_time(hist_times[i]))
                e_ys.append(math.log(e + 0.5))
        if e_xs:
            e_intercept, e_slope, e_rms = fit_line(e_xs, e_ys)
        else:
            e_intercept, e_slope, e_rms = 0.0, 0.0, 0.0

        if e_xs and last_earners > 0:
            growth = e_slope * (x_target - x_last)
            y_pred = math.log(last_earners + 0.5) + 0.7 * growth
            earners_pred = math.exp(y_pred) - 0.5
        elif e_xs:
            y_fit_last = e_intercept + e_slope * x_last
            growth = e_slope * (x_target - x_last)
            y_pred = y_fit_last + 0.7 * growth
            earners_pred = math.exp(y_pred) - 0.5
        else:
            earners_pred = 0.0

        if earners_pred < 0.6 * last_earners:
            earners_pred = 0.6 * last_earners
        if earners_pred < 0:
            earners_pred = 0.0
        if earners_pred > players_pred:
            earners_pred = players_pred

        raw_median = 100.0 * earners_pred / players_pred if players_pred > 0 else 0.0
        raw_median = clamp(raw_median, 0.01, 99.99)

        if last_rate is not None:
            blended_logit = 0.75 * logit_of_rate(raw_median) + 0.25 * logit_of_rate(clamp(last_rate, 0.01, 99.99))
            median_rate = clamp(rate_from_logit(blended_logit), 0.01, 99.99)
        else:
            median_rate = raw_median

        rw_term = 0.45 * math.sqrt(horizon_days) if horizon_days > 0 else 0.0

        if n_refresh > 0:
            per_jumps = []
            for i in refresh_indices:
                a = rates_series[i - 1] if i - 1 < len(rates_series) else None
                b = rates_series[i] if i < len(rates_series) else None
                if a is not None and b is not None and 0.0 < a < 100.0 and 0.0 < b < 100.0:
                    per_jumps.append(abs(logit_of_rate(b) - logit_of_rate(a)))
            if per_jumps:
                mean_jump = sum(per_jumps) / len(per_jumps)
            elif global_mean_jump is not None:
                mean_jump = global_mean_jump
            else:
                mean_jump = 1.0
            refresh_term = mean_jump * math.sqrt(expected_refreshes) if expected_refreshes > 0 else 0.0
        else:
            eff_expected = max(expected_refreshes, 0.25)
            refresh_term = 1.0 * math.sqrt(eff_expected)

        resid_term = max(e_rms, 0.1)
        sigma = math.sqrt(rw_term * rw_term + refresh_term * refresh_term + resid_term * resid_term)
        sigma = clamp(sigma, 0.35, 2.3)

        median_logit = logit_of_rate(median_rate)
        quantiles = []
        for level in levels:
            try:
                lv = float(level)
            except (TypeError, ValueError):
                lv = 0.5
            if lv <= 0.0:
                qv = 0.01
            elif lv >= 1.0:
                qv = 100.0
            elif lv == 0.5:
                qv = median_rate
            else:
                side = sigma * 1.15 if lv < 0.5 else sigma * 0.92
                z = normal.inv_cdf(lv)
                qv = rate_from_logit(median_logit + side * z)
            qv = clamp(qv, 0.01, 100.0)
            qv = round(float(qv), 4)
            quantiles.append(qv)

        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]

        out_trophies.append({"id": tid, "q": quantiles})

    return {"trophies": out_trophies}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(in_dir, "*.in")))
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = process_case(case)
        case_id = case.get("id")
        if case_id is None:
            case_id = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, str(case_id) + ".out")
        payload = json.dumps(result, separators=(",", ":"))
        if len(payload.encode("utf-8")) > 65536:
            # Never truncate: that would produce invalid JSON. Re-round
            # the quantiles to fewer decimals until the payload fits,
            # re-enforcing monotonicity after each re-rounding.
            for decimals in (3, 2, 1, 0):
                for entry in result["trophies"]:
                    rounded = [round(float(v), decimals) for v in entry["q"]]
                    for i in range(1, len(rounded)):
                        if rounded[i] < rounded[i - 1]:
                            rounded[i] = rounded[i - 1]
                    entry["q"] = rounded
                payload = json.dumps(result, separators=(",", ":"))
                if len(payload.encode("utf-8")) <= 65536:
                    break
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(payload)


if __name__ == "__main__":
    main()
