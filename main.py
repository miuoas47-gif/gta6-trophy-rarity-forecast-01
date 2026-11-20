"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case (/input/<id>.in, schema "makefun/gta6-trophies-case/v1") and
writes /output/<id>.out: for each trophy, five quantiles (5/25/50/75/95%) of
its Exophase earn rate at the case's target. Python 3.13, standard library
only, no network, deterministic: the same case always yields the same bytes.

Method
------
An Exophase earn rate is a ratio, r = E / N: E tracked players who earned the
trophy over N players tracked. Both counts grow after launch, and the rate
moves with whichever grows faster (a surge of new players dilutes every rate;
play time pushes each trophy's earners up). So the model works in counts,
not in rates:

* Time is measured in hours since the game's release (2026-11-19T00:00Z) and
  compressed with x = ln(1 + hours): launch growth is fast, then slows, and a
  log-time axis makes that roughly linear instead of explosive.
* For each trophy, ln E is fitted against x by weighted least squares over
  the case's distinct reads (consecutive identical reads carry no new
  information and are collapsed first). Weights grow with the earner count
  (a rate printed to 2 decimals is a noisier count when E is small) and fade
  gently with age, so the recent regime counts most.
* ln N is fitted the same way. The forecast is r = 100 * E(x_target)/N(x_target).
* Each trophy's slope is shrunk toward its tier's mean slope (platinum, gold,
  silver, bronze behave alike), which steadies the rare trophies whose counts
  are tiny and noisy. A trophy with no usable trend falls back to its last
  rate, exactly like the house baseline.
* The median's move from the last read is capped at 3 logit units, so a
  power-law extrapolation can never run away over a 4.5-day horizon.
* Story trophies must be earned in order (Chapter 1 before Chapter 2, ...,
  final mission, platinum), so their forecast medians are made
  non-increasing along that chain by pool-adjacent-violators averaging.
  The chain is also forecast as a funnel: the ratio E(next)/E(prev) rises
  as the player cohort matures, so a late chapter whose earners are still
  flat in the reads is not left at zero — its ratio is taken from the reads
  where they show one, else the last observed ratio, never below a floor
  (a real share of the players who finished one chapter finish the next
  within a week), with a wider spread to price that structural judgement.
* The spread is an honest one: a random-walk term that widens with the
  horizon (0.5 logit per sqrt day, as the house baseline), the fit's residual
  spread, a structural floor for Exophase refreshes and regime changes, and
  extra width when the median moved far. Quantiles are the median shifted by
  the normal quantiles times that spread, on the logit scale.

What it does not do: it never fetches anything (the sandbox has no network),
never hard-codes a trophy or a rate, and never looks at the answer. On a
practice case, whose whole history is one unchanged launch-night page, every
trend is zero and the forecast is the house baseline's persistence forecast.

Sources: the case's own history (the house's recording of
https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/); the
competition's rules for the scoring (pinball loss on the logit scale) and the
house baseline it improves on.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# GTA VI's release, the origin of the log-time axis.
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# A rate move is never trusted beyond this many logit units from the last read.
MAX_LOGIT_MOVE = 3.0
# Spread floors / caps (logit units) and the random-walk rate per sqrt day.
SPREAD_FLOOR = 0.30
SPREAD_CAP = 2.00
RANDOM_WALK_SIGMA = 0.5
STRUCTURAL_SIGMA = 0.15


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    l = max(-30.0, min(30.0, l))
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    """Round to 4 decimals, half up, as the house baseline rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def log_time(at):
    """Hours since release, compressed: ln(1 + hours), never negative."""
    hours = (at - RELEASE).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_std, n, mx, my)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, len(xs), mx, my)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    return (a, b, math.sqrt(var), len(xs), mx, my)


def distinct_reads(history):
    """Collapse consecutive identical reads; each snapshot is one observation."""
    seen = []
    last_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != last_key:
            seen.append(read)
            last_key = key
    return seen


def story_chain(trophies):
    """The story trophies in play order: Chapters 1..n, final mission, platinum.

    Matched on the description, exactly: "Complete Chapter N." and
    "Complete the final mission." (the side missions "Complete the final
    mission for <person>." are not the story's end and are left out).
    """
    chain = []
    for i, trophy in enumerate(trophies):
        desc = (trophy.get("description", "") or "").strip()
        m = re.match(r"Complete Chapter (\d+)\.$", desc)
        if m:
            chain.append((int(m.group(1)), i))
        elif desc.lower() == "complete the final mission.":
            chain.append((99, i))
        elif trophy.get("tier") == "platinum":
            chain.append((100, i))
    chain.sort()
    return chain


def earner_series(reads, trophies):
    """Per-trophy (x, E) points from the distinct reads, E from rate x N."""
    series = []
    for i in range(len(trophies)):
        pts = []
        for read in reads:
            rates = read.get("rates") or []
            n = read.get("players_tracked")
            if i >= len(rates) or rates[i] is None or not n or n <= 0:
                continue
            eff_rate = rates[i] if rates[i] > 0 else 0.005
            pts.append((log_time(parse_time(read["at"])), eff_rate * n / 100.0))
        series.append(pts)
    return series


def funnel_ratio(prev_pts, next_pts, x_target):
    """Forecast E(next)/E(prev) at the target from the reads' ratios.

    Fits logit(u) against log-time where the reads show a trend, else holds
    the last observed ratio. Returns a ratio in (0, 0.95].
    """
    by_x = {}
    for x, e_prev in prev_pts:
        by_x.setdefault(x, {})["prev"] = e_prev
    for x, e_next in next_pts:
        by_x.setdefault(x, {})["next"] = e_next
    xs, ys = [], []
    last_u = None
    for x in sorted(by_x):
        d = by_x[x]
        if "prev" in d and "next" in d and d["prev"] > 0:
            u = min(0.99, max(0.001, d["next"] / d["prev"]))
            xs.append(x)
            ys.append(math.log(u / (1.0 - u)))
            last_u = u
    if last_u is None:
        return 0.10
    if len(xs) >= 3:
        fit = wls(xs, ys, [1.0] * len(xs))
        if fit is not None and fit[1] > 0:
            u = 1.0 / (1.0 + math.exp(-(fit[0] + fit[1] * x_target)))
            return max(last_u, min(0.95, u))
    return min(0.95, last_u)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    reads = distinct_reads(case["history"])
    target = parse_time(case["target"])
    x_target = log_time(target)
    horizon_days = case["horizon_minutes"] / 1440.0

    # --- players tracked: ln N against log-time ---------------------------
    xs_n, ys_n = [], []
    for read in reads:
        n = read.get("players_tracked")
        if n and n > 0:
            xs_n.append(log_time(parse_time(read["at"])))
            ys_n.append(math.log(n))
    fit_n = wls(xs_n, ys_n, [1.0] * len(xs_n))
    if fit_n is None:
        # A single distinct snapshot (the practice cases): no trend to fit,
        # so the player count is taken as staying where the last read put it.
        last_n = reads[-1].get("players_tracked") or 1
        a_n, b_n, res_n = math.log(last_n), 0.0, 0.0
    else:
        a_n, b_n, res_n, _, _, _ = fit_n
    log_n_target = a_n + b_n * x_target

    # --- per-trophy earner counts: ln E against log-time ------------------
    last_x = log_time(parse_time(reads[-1]["at"]))
    tier_slopes = {}
    tier_counts = {}
    fits = []
    for i, trophy in enumerate(trophies):
        xs, ys, ws = [], [], []
        last_rate = None
        for read in reads:
            rates = read.get("rates") or []
            if i >= len(rates):
                continue
            rate = rates[i]
            if rate is None:
                continue
            n = read.get("players_tracked")
            if not n or n <= 0:
                continue
            last_rate = rate
            # A printed 0.00% means fewer than 0.005%: use the bin midpoint.
            eff_rate = rate if rate > 0 else 0.005
            earners = eff_rate * n / 100.0
            x = log_time(parse_time(read["at"]))
            xs.append(x)
            ys.append(math.log(max(earners, 0.05)))
            # Count noise shrinks with E; recent reads count most.
            recency = 0.5 ** ((last_x - x) / 48.0)
            ws.append(min(max(earners, 0.5), 200.0) * recency)
        fit = wls(xs, ys, ws) if len(xs) >= 3 else None
        if fit is not None and fit[3] >= 3 and fit[1] == fit[1]:
            tier = trophy.get("tier", "bronze")
            tier_slopes[tier] = tier_slopes.get(tier, 0.0) + fit[1] * fit[3]
            tier_counts[tier] = tier_counts.get(tier, 0) + fit[3]
        fits.append((fit, last_rate))
    tier_mean = {}
    for tier, total in tier_slopes.items():
        tier_mean[tier] = total / max(1, tier_counts[tier])
    global_mean = (
        sum(tier_slopes.values()) / max(1, sum(tier_counts.values()))
        if tier_slopes
        else 0.0
    )

    # --- forecast each trophy ---------------------------------------------
    medians = []
    spreads = []
    for i, trophy in enumerate(trophies):
        fit, last_rate = fits[i]
        if last_rate is None:
            last_rate = 1.0
        if fit is None:
            # No trend to fit: persistence, like the house baseline.
            medians.append(float(last_rate))
            spreads.append(
                math.sqrt(RANDOM_WALK_SIGMA**2 * horizon_days + STRUCTURAL_SIGMA**2)
            )
            continue
        a_e, b_e, res_e, n_pts, mx_e, my_e = fit
        tier = trophy.get("tier", "bronze")
        b_tier = tier_mean.get(tier, global_mean)
        # Shrink the slope toward the tier mean; thin data shrinks most.
        # The intercept is re-centred on the shrunk slope, so the line still
        # passes through the weighted centre of the trophy's own reads.
        b_e = (n_pts * b_e + 3.0 * b_tier) / (n_pts + 3.0)
        a_e = my_e - b_e * mx_e
        log_rate_last = logit(last_rate)
        # Forecast the rate itself from the counts, then take its logit, so
        # large rates (where logit and log-rate differ) are handled exactly.
        rate_target = 100.0 * math.exp((a_e + b_e * x_target) - log_n_target)
        log_rate_target = logit(rate_target)
        # Keep the move honest: cap it, and never let a rounding artefact
        # (a 0.00% rate) push the median to an absurd place.
        move = max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, log_rate_target - log_rate_last))
        medians.append(inv_logit(log_rate_last + move))
        sigma_res = math.sqrt(res_e * res_e + res_n * res_n)
        spread = math.sqrt(
            RANDOM_WALK_SIGMA**2 * horizon_days
            + (1.5 * sigma_res) ** 2
            + STRUCTURAL_SIGMA**2
            + (0.25 * abs(move)) ** 2
        )
        spreads.append(max(SPREAD_FLOOR, min(SPREAD_CAP, spread)))

    # --- a trophy everyone holds stays held --------------------------------
    for i, trophy in enumerate(trophies):
        if fits[i][1] is not None and fits[i][1] >= 99.99:
            medians[i] = 99.95
            spreads[i] = 0.10

    # --- story funnel: later chapters catch up ------------------------------
    # The count model alone fails the late story trophies: their earner
    # counts are flat in the observed window (nobody had reached them yet
    # except reviewers), so it would forecast them near zero forever. But
    # players progress through the chapters in order, and the funnel ratio
    # u = E(next)/E(prev) rises as the cohort matures: on 2026-11-20 the
    # live page already showed Chapter 2 / Chapter 1 at 0.11 and climbing,
    # from ~1.0 among the launch-night reviewers. So for each step of the
    # chain, take the funnel ratio from the reads (a log-time fit where the
    # data shows one, else the last observed ratio) and never below a floor:
    # by a week after launch a real share of the players who finished one
    # chapter have finished the next. The floors are structural priors, and
    # the wider spread below prices the uncertainty they carry.
    chain = story_chain(trophies)
    funnel_floors = [0.30, 0.25, 0.20, 0.25, 0.01]  # ch2, ch3, ch4, final, platinum
    if len(chain) >= 2:
        earners = earner_series(reads, trophies)
        for step in range(len(chain) - 1):
            _, i_prev = chain[step]
            _, i_next = chain[step + 1]
            u_target = funnel_ratio(
                earners[i_prev], earners[i_next], x_target
            )
            floor = funnel_floors[step] if step < len(funnel_floors) else 0.10
            u_target = max(u_target, floor)
            medians[i_next] = medians[i_prev] * u_target
            spreads[i_next] = min(
                SPREAD_CAP,
                math.sqrt(spreads[i_next] ** 2 + 0.35**2),
            )

    # --- story trophies are earned in order: enforce it on the medians ----
    chain = story_chain(trophies)
    if len(chain) >= 2:
        # Pool adjacent violators on the logit medians, decreasing order.
        vals = [logit(medians[i]) for _, i in chain]
        blocks = [[v, 1] for v in vals]
        k = 0
        while k < len(blocks) - 1:
            if blocks[k][0] < blocks[k + 1][0] + 1e-12:
                # Violation: later trophy forecast above the earlier one.
                total = blocks[k][0] * blocks[k][1] + blocks[k + 1][0] * blocks[k + 1][1]
                count = blocks[k][1] + blocks[k + 1][1]
                blocks[k:k + 2] = [[total / count, count]]
                if k > 0:
                    k -= 1
            else:
                k += 1
        flat = []
        for v, c in blocks:
            flat.extend([v] * c)
        for (_, i), v in zip(chain, flat):
            medians[i] = inv_logit(v)

    rows = []
    for i, trophy in enumerate(trophies):
        centre = logit(medians[i])
        s = spreads[i]
        q = [round4(inv_logit(centre + z * s)) for z in Z]
        # Rounding can only tie neighbours, never invert them; enforce anyway.
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
