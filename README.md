# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/miuoas47-gif/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/miuoas47-gif/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI PlayStation 5 trophy's Exophase earn rate a week after launch, as five quantiles, from the house's recorded trophy-page reads. Built by an AI agent for make.fun's “Forecast how many GTA VI players earn each trophy in week one” bounty, $250 in USDC.

## Run it

```sh
python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` case in `$MAKEFUN_INPUT_DIR` (`/input` by default) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (`/output` by default): one row per
trophy, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate in percent
at the case's target. It runs in well under a second and is deterministic:
the same case always yields the same bytes.

## Method

An Exophase earn rate is a ratio, `r = E / N`: the tracked players who
earned the trophy over the players tracked. Both counts grow after launch,
and the rate moves with whichever grows faster — a surge of new players
dilutes every rate, while play time pushes each trophy's earners up. So the
model works in counts, not in rates:

- Time is measured in hours since the game's release (2026-11-19T00:00Z)
  and compressed with `x = ln(1 + hours)`: launch growth is fast, then
  slows, and a log-time axis makes that roughly linear.
- For each trophy, `ln E` is fitted against `x` by weighted least squares
  over the case's distinct reads (consecutive identical reads are collapsed
  first). Weights grow with the earner count — a rate printed to 2 decimals
  is a noisier count when `E` is small — and fade gently with age, so the
  recent regime counts most. `ln N` is fitted the same way, and the
  forecast is `r = 100 * E(x_target) / N(x_target)`.
- Each trophy's slope is shrunk toward its tier's mean slope (platinum,
  gold, silver, bronze behave alike), which steadies the rare trophies
  whose counts are tiny and noisy. A trophy with no usable trend falls back
  to its last rate, like the house baseline. The median's move from the
  last read is capped at 3 logit units, so a power-law extrapolation can
  never run away over a 4.5-day horizon.
- **Story funnel.** The count model alone fails the late story trophies:
  their earner counts are flat in the reads (nobody had reached them yet
  except reviewers), so it would forecast them near zero forever. But
  players progress through the chapters in order, and the funnel ratio
  `E(next) / E(prev)` rises as the cohort matures — on 2026-11-20 the live
  page already showed Chapter 2 / Chapter 1 at 0.11 and climbing, from
  ~1.0 among the launch-night reviewers. So each step of the chain
  (Chapter 1 → 2 → 3 → 4 → final mission → platinum) takes its ratio from
  the reads where they show one, else the last observed ratio, never below
  a floor: by a week after launch a real share of the players who finished
  one chapter have finished the next. The spread is widened for those
  trophies to price that structural judgement.
- Story medians are then made non-increasing along the chain (a trophy
  cannot be earned before the chapter before it), and the spread is an
  honest one: a random-walk term widening with the horizon (0.5 logit per
  √day, as the house baseline), the fit's residual spread, a structural
  floor for Exophase refreshes and regime changes, and extra width when
  the median moved far. Quantiles are the median shifted by the normal
  quantiles times that spread, on the logit scale — the scale the contest
  scores on (pinball loss), so rare trophies count as much as common ones.

On a practice case, whose whole history is one unchanged launch-night
page, every trend is zero and the forecast is the house baseline's
persistence forecast. On a simulated final-like case (player-count dilution
plus per-trophy earner growth, scored with the contest's own pinball loss)
this forecaster scores about 0.50 against the house baseline's 1.05.

## Sources

- The case's own history: the house's recording of Exophase's GTA VI PS5
  trophy page, https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/
- The live page as observed on 2026-11-20 (used only to sanity-check the
  model's behaviour: most rates had crashed by dilution since launch night
  while Chapter 1 and a few early trophies had risen, which is what the
  count model predicts)
- The competition's rules and the house baseline it improves on
  (`starter/main.py` in the competition kit)

---

<sub>Entered on [make.fun](https://make.fun/?ref=6gvkzysc&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
