# [OpenAI Gym](https://gym.openai.com/) Environment for the [Asset Allocation Problem](https://en.wikipedia.org/wiki/Asset_allocation)

This environment implements the Asset Allocation problem.

Currently there are no transaction costs, and the assets are fixed, with data pulled from `quandl`.

For each assset, a random year of data is used for each episode (the year used can differ for each asset).

The portfolio is rebalanced at the start of each day. On a `step`, we provide `action` describing how we want to balance the portfolio at the start of the day. As an observation we are then given the days `OHLCV` for each asset, and our reward is the fractional change in portfolio value over the course of the day.

An episode terminates after 1 year of trading.

## Observations

The observation at each timestep is a `numpy` array of shape `(n, 5)`, containing the day's `OHLCV` for each of the `n` assets.

## Actions

The action given is the relative allocation of funds between each asset. If `Σaᵢ > 1` we normalise `a`. If `Σaᵢ < 1` the remaining funds are put into 'cash', whose value does not change with time. If `aᵢ < 0`, `aᵢ` is treated as `0`.

Rebalancing is done at the start of a day, so assets are bought and sold at the day's `Open` price.

## Rewards

The reward given is `(valueᶜˡᵒˢᵉ - valueᵒᵖᵉⁿ) / valueᵒᵖᵉⁿ` for `open` and `close` prices on the same day.

This potentially leads to the true value of the portfolio not exactly following what would be expected from the rewards, due to the following day's `open` not being equal to previous day's `close`.

The reward function has been chosen as it is to avoid the actor being able to infer some information on the following day's `open` price from it's reward.

## TODO

* Work with more than just `close` prices
* Add more assets
* Add transaction costs
    * Adds some complexity with rebalancing portfolio