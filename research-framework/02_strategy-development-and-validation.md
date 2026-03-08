# 2. Strategy Development and Validation

If markets are competitive and strong persistent alpha is rare, then the main challenge in systematic trading is not generating ideas but testing them without fooling ourselves. A large share of apparent strategy performance comes from noise, hidden exposures, favorable sample periods, or implementation assumptions that do not survive contact with reality. For that reason, strategy development should be treated as a process of disciplined filtering rather than performance maximization.

This note outlines that process. The objective is not to produce the most attractive backtest, but to design a workflow that gives weak ideas many opportunities to fail before capital is committed.

## 2.1 From intuition to testable hypothesis

A strategy should begin with a hypothesis, not with a search for profitable patterns. The hypothesis does not need to be perfect, but it should at least explain why a signal might contain information or why a rule might improve the management of risk.

Useful hypotheses often come from one of four sources:

- **Risk premia**: expected returns earned for bearing compensated risk.
- **Behavioral effects**: delayed reaction, overreaction, underreaction, or positioning pressure.
- **Structural frictions**: rebalancing flows, liquidity segmentation, roll dynamics, or institutional constraints.
- **Implementation improvements**: better sizing, timing, or execution relative to simpler alternatives.

A hypothesis should be stated in a form that can be challenged. For example:

> If realized volatility rises sharply after a sustained trend break, reducing exposure may improve risk-adjusted returns relative to a constant-notional benchmark.

This is better than beginning with a vague claim such as “volatility seems useful.” A good hypothesis implies what should be measured, what benchmark should be used, what data are required, and what result would count as failure.

## 2.2 Define the target before looking at the result

Before running a backtest, it helps to define what success would mean. Otherwise it becomes too easy to reinterpret the objective after seeing the numbers.

The target may involve one or more of the following:

- positive net returns after realistic costs,
- improved Sharpe-like performance relative to a benchmark,
- lower drawdowns at similar return,
- better crisis behavior,
- lower volatility of returns,
- or improved diversification when combined with other strategies.

The key is that the objective should match the economic purpose of the strategy. A risk-reduction overlay should not be judged only by raw return. A directional strategy should not be defended solely because it lowers turnover. A timing strategy should be compared with an appropriate passive or simpler active benchmark.

Clarity on this point makes later evaluation much more honest.

## 2.3 Data as part of the model

Data should not be treated as a neutral input. The choice of data source, cleaning method, sampling frequency, contract construction, and corporate handling can materially change the apparent performance of a strategy. In that sense, the dataset is part of the model.

The first questions should be basic:

- What instrument is being traded?
- What exact fields are used?
- At what frequency are decisions made?
- What information was available at the decision time?
- Are prices tradeable, settlement-based, indicative, or revised?
- Are missing values, roll effects, or timestamp issues handled explicitly?

A useful discipline is to define the tradable object before defining the signal. In futures, for example, the researcher must specify whether the strategy is based on front contracts, back-adjusted continuous series, excess return series, or some roll-aware implementation. Different constructions answer different economic questions.

Poor data discipline creates false confidence faster than almost anything else in systematic research.

## 2.4 Signal design should stay economically interpretable

It is tempting to generate dozens of indicators, transform them repeatedly, and search for combinations that maximize in-sample performance. The result may look sophisticated while being little more than parameterized noise.

A better practice is to keep the signal interpretable. The researcher should be able to explain:

- what information the signal is using,
- why that information might matter,
- how often the signal changes,
- and which market conditions should help or hurt it.

This does not mean every strategy needs a fully proven theoretical foundation. It does mean that a rule should have enough structure that failure is informative. If a strategy works only because of a highly specific parameter combination with no clear interpretation, then its future behavior is difficult to trust.

Interpretability also helps when live performance deviates from the backtest. A transparent signal gives the researcher a better chance of understanding whether the problem is due to the market regime, execution, data construction, or the original hypothesis itself.

## 2.5 Build the backtest to reject the idea, not to sell it

The purpose of a backtest is often misunderstood. It is not a marketing document. It is a stress environment for the hypothesis.

A useful backtest should be built with enough realism that poor ideas fail quickly. At a minimum, this usually means accounting for:

- transaction costs and slippage,
- realistic execution timing,
- position limits or leverage bounds,
- liquidity constraints where relevant,
- contract rolls and instrument changes,
- and any delays between signal formation and execution.

Even if the assumptions are imperfect, it is better to state them clearly than to hide them behind idealized fills.

The backtest should also preserve the actual decision sequence of the strategy. Information used at time $t$ must be available at time $t$. Any use of future data, revised data, or post-close information to justify same-day trading decisions weakens the whole exercise.

In a skeptical framework, the backtest is designed so that the strategy must earn the right to be taken seriously.

## 2.6 Separate development from evaluation

One of the easiest ways to fool yourself is to use the same data repeatedly for both model development and final judgment. Even without explicit overfitting, repeated exposure to the same history shapes intuition, parameter choices, and what the researcher notices.

For that reason, strategy development should distinguish between:

- **development data**, used to form and refine the rule,
- **validation data**, used to test whether the idea survives outside the development sample,
- and, where possible, **truly untouched holdout or forward data**, used for final evaluation.

This separation does not remove all overfitting risk, but it reduces the chance that a strategy’s apparent robustness is merely familiarity with a particular history.

In time series settings, this often implies walk-forward logic, rolling windows, anchored expansions, or other methods that respect time ordering. The exact method may vary, but the principle is constant: a strategy should be judged on data that did not participate in its construction.

## 2.7 Robustness matters more than the best parameter set

A fragile strategy can often be made to look impressive by selecting the right lookback, threshold, rebalance rule, or sample start date. That is why the best-looking specification is often less interesting than the stability of nearby specifications.

Robustness questions include:

- Does the result survive small changes in lookback windows?
- Does it depend on one crisis period or one strong trend?
- Does it collapse after modest cost increases?
- Does changing the rebalance day matter too much?
- Does the idea remain coherent on related instruments or adjacent markets?
- Is performance broad-based, or concentrated in a small number of outlier trades?

A strategy that remains directionally sensible across small perturbations is more credible than one whose performance depends on precise tuning.

The same logic applies to complexity. Adding filters, thresholds, and overlays may improve the in-sample chart while reducing the chance that the strategy is learning anything durable. Complexity should be earned, not assumed.

## 2.8 Evaluation should be multi-dimensional

No single statistic can summarize whether a strategy is worth pursuing. Average return, Sharpe ratio, and drawdown each capture only part of the picture.

A more complete evaluation considers several dimensions:

- **Return**: level, consistency, and contribution by period.
- **Risk**: volatility, drawdown depth, drawdown duration, tail behavior.
- **Implementation burden**: turnover, leverage needs, operational complexity.
- **Exposure**: market beta, directional bias, volatility sensitivity, regime dependence.
- **Stability**: sensitivity to assumptions, concentration of returns, out-of-sample decay.

The point is not to maximize every metric. It is to understand the trade-offs. A strategy with moderate returns but stable behavior, low complexity, and coherent exposures may be more useful than one with excellent headline performance and hidden fragility.

## 2.9 Benchmarks are part of the research design

A strategy should never be evaluated in isolation. The relevant question is rarely “did it make money?” but rather “did it improve on something simpler or more appropriate?”

Benchmark choice depends on the role of the strategy:

- a directional futures model may be compared with passive exposure,
- a volatility-targeting overlay may be compared with constant exposure,
- a trend signal may be compared with a buy-and-hold or simple moving-average baseline,
- a portfolio sleeve may be compared with an equal-risk or equal-weight alternative.

Benchmarks discipline interpretation. They prevent the researcher from confusing general market tailwinds with signal quality, and they help clarify what part of the result comes from the rule itself rather than from broad exposure.

A sophisticated strategy that cannot outperform a simpler benchmark after costs may still be educational, but it should not be presented as a robust improvement.


## 2.10 From backtest to implementation


A strategy is not complete when the backtest is finished. There is a gap between research performance and live performance, and that gap should be expected rather than treated as an anomaly.

Some sources of live deviation are unavoidable:

- signal timestamps may differ from research assumptions,
- execution prices may be worse,
- market impact may appear,
- broker data may not match research data,
- contract rolls may be handled differently,
- risk controls may intervene,
- and software or operational failures may create drift.

The research process should therefore include a translation step: how exactly does the tested rule become a live process? This includes signal timing, order logic, sizing conventions, exception handling, data dependencies, and monitoring.

A strategy that cannot be translated cleanly from notebook to execution framework is not ready, no matter how attractive the research output appears.

## 2.11 Monitoring is part of validation

Validation does not end when trading begins. Live performance provides new information about whether the original hypothesis was sound, whether implementation matches design, and whether the strategy behaves as expected under current market conditions.

This does not mean reacting emotionally to every period of underperformance. It means monitoring the aspects of behavior that matter:

- is turnover in line with research?
- are realized exposures close to expected exposures?
- are costs materially higher than assumed?
- are signals arriving on schedule?
- is performance deterioration broad-based or explained by regime conditions?
- is the live path within a plausible distribution of the tested strategy?

The point of monitoring is not to rescue a weak strategy with discretionary overrides. It is to distinguish between normal variation, implementation error, and evidence that the underlying idea is weaker than believed.

## 2.12 Research discipline over backtest aesthetics

The final standard for a strategy should not be whether the equity curve looks persuasive. The standard should be whether the idea was developed in a way that reduces self-deception.

A disciplined process does not guarantee success. It does, however, improve the odds that failure is informative and that success is more than an artifact of curve fitting.

In that sense, the central task of strategy development is not to discover the most impressive historical pattern, but to create a framework in which weak ideas are rejected early, plausible ideas are challenged rigorously, and any surviving strategy can be implemented with clear understanding of what it is supposed to do.

The next step is to examine one of the areas where this discipline matters most in practice: futures trading and the special challenges of backtesting and implementation in that setting.