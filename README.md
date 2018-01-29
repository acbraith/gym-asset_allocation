# [OpenAI Gym](https://gym.openai.com/) Environment for the [Asset Allocation Problem](https://en.wikipedia.org/wiki/Asset_allocation)

This environment implements the Asset Allocation problem.

Currently there are no transaction costs, and the assets are fixed, with data pulled from `quandl`.

For each assset, a random year of data is used for each episode (the year used can differ for each asset).

The portfolio is rebalanced at the start of each day. On a `step`, we provide `action` describing how we want to balance the portfolio at the start of the day. As an observation we are then given the days `OHLCV` for each asset, and our reward is the fractional change in portfolio value over the course of the day.

An episode terminates after 1 year of trading.

## Observations

The observation at each timestep is a `numpy` array of shape `(n, 5)`, containing the day's `OHLCV` for each of the `n` assets.

## Actions

The action given is the relative allocation of funds between each asset. Each `0 ≤ aᵢ ≤ 1`, and if `Σaᵢ > 1` we normalise such that `Σaᵢ = 1`. If `Σaᵢ < 1` the remaining funds are put into 'cash', whose value does not change with time.

Rebalancing is done at the start of a day, so assets are bought and sold at the day's `open` price.

## Rewards

The reward given is `(valueₜ - valueₜ₋₁) / valueₜ₋₁`, ie the change in portfolio value from the start of day `t-1` to the start of day `t`.

## TODO

* Try out a baselines agent
* Add more assets
    * 'short' for each asset, with price series mimicing the result from shorting the asset?
* Add transaction costs
    * Adds some complexity with rebalancing portfolio
* Add more than just `OHLCV` data
    * eg some fundamental stuff, technical indicators, whatever