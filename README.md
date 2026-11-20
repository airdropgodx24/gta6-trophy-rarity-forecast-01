# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/airdropgodx24/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/airdropgodx24/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entered on [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) for the bounty "Forecast how many GTA VI players earn each trophy in week one" ($250 in USDC).

## What it does

`main.py` reads each case file it is given — a history of reads of the game's Exophase trophy page, oldest first — and writes, for every one of the 55 PlayStation 5 trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of that trophy's earn rate at the case's target moment: the percentage of tracked players who will hold the trophy then.

## The method, in plain terms

A printed earn rate is a ratio, not a raw count:

    rate = 100 * earners / players_tracked

That one fact shapes the whole forecast, because the two sides of the ratio move in completely different ways during a launch week.

**Players tracked** only moves when the tracker refreshes its page. Between refreshes the count is frozen; at a refresh it jumps as new profiles are folded in. **Earners** of a trophy behave oppositely: nobody ever un-earns a trophy, so the earner count only grows, and it grows roughly continuously — including straight through a refresh, where the measured earners (rate × players ÷ 100) barely move while the player count leaps.

So the program never extrapolates the rate itself. For each trophy it:

1. Rebuilds the earner series from the history (rate × players ÷ 100 at every read) alongside the player series.
2. Forecasts players at the target by fitting ln(players) against ln(1 + hours since the first read) — launch growth is close to straight on that axis — with the result floored at the last count and capped at 25× it.
3. Forecasts earners the same way, but with the fitted growth beyond the last observation damped to 70%: first-week enthusiasm fades, and an undamped line promises more finishers than the week can produce. Earners are floored at 60% of the last count (they cannot truly fall) and capped at the forecast player count.
4. Divides the two forecasts back into a median rate, then blends it 75/25 on the logit scale with the last observed rate, so a shaky fit cannot drag the forecast far from what was actually seen.

## Assumptions

- Earners never decrease in reality; any dip in the measured series is a tracking artefact, not players losing trophies.
- The tracker will keep refreshing during the forecast horizon at roughly the cadence seen in the history, and each refresh dilutes rates in the same direction as the ones already observed.
- Log-time (ln(1 + hours)) is a reasonable axis for launch-week growth of both players and earners over a horizon of a few days.
- A trophy sitting at 100% has earners equal to players, so its earners grow with the player count and its rate stays near the top instead of collapsing.

## Uncertainty, stated honestly

The five quantiles are built on the logit scale, where a move from 1% to 2% counts about as much as a move from 33% to 50% — rare trophies matter as much as common ones. The spread combines three independent sources of doubt in quadrature:

- a plain random-walk drift of 0.45 logit units per square-root day;
- a refresh term: the mean absolute logit jump actually measured across reads where the player count changed, scaled by how many refreshes the horizon should contain. If the history shows **no** refresh at all, the program does not conclude the future is calm — it falls back on a prior jump of 1.0 logit units at about two refreshes a day. A flat launch-night history is exactly the situation where a narrow forecast would be dishonest;
- the residual scatter of the earners fit itself, so a trophy whose growth has been erratic gets a wider band than a steady one.

The total is clamped to a sane range, and the band is deliberately asymmetric: the downward side is widened by 15% and the upward side narrowed slightly, because refresh dilution pushes rates down far more often than anything pushes them up during a launch week.

Nothing here is fitted to the answers — there are no answers yet. The constants above were chosen from the structure of the problem (a ratio, a jumping denominator, a one-way numerator), not tuned against any target data.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network, deterministic: the same case always produces the same forecast file.

---

<sub>Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty.</sub>
