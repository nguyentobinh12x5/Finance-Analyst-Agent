# Understanding the Quantitative System Intuitively 🇬🇧

This document will explain the **Architectural Mindset using everyday analogies.**

---

## Part 1: Decoding Feature Engineering (The 14 Health Indicators)
Instead of letting the AI blindly read news, we force it to evaluate a company through 14 distinct financial indicators (defined in our `features` array). We categorize these into 4 Groups to let the AI forecast if the stock will skyrocket or crash in the next 3 months:

### Group 1: Profitability Health (Do you make real money?)
- **EPS (Earnings Per Share)**: The company made a $10M profit this year. Divided by 10M shareholders, how much real cash did one share generate?
- **ROE (Return on Equity)**: If you inject $100 of pure cash into the company, how much interest is returned to you a year later? For example, Vinpearl (VPL) had a distorted ROE back in 2008 due to huge capital tied up in newly built hotels.
- **net_income_ratio (Net Profit Margin)**: For every $100 of revenue, after deducting rent, power, taxes... how much net profit actually goes into the pocket?

### Group 2: Valuation (Are we buying an overpriced bubble?)
- **PE (Price to Earning)**: The company makes $1k a year in profit. The market is pricing the stock at $20k to own it outright. Thus, PE = 20 (You must hold the stock for 20 years to break even). A very high PE often signals an overbought bubble (or sky-high expectations).
- **PB (Price to Book)**: If the company liquidates all its factories and assets into scrap, it gets $1B (Book Value). But the market currently prices the firm at $2B. Thus, PB = 2.

### Group 3: Liquidity (Will it go bankrupt from missing cash?)
- **cur_ratio (Current Ratio)**, **quick_ratio**, & **cash_ratio**: These simply ask: *If creditors aggressively demand their short-term loans back tomorrow morning, does the company have enough liquid assets / cash in the safe to pay instantly?* If the `cash_ratio` is too low, the company is extremely vulnerable to sudden death.
- **acc_rec_turnover (Accounts Receivable Turnover)**: Translates to **"Debt Collecting Skill"**. A high ratio means the company sells goods on credit and successfully claws back cash into its vault swiftly! A low ratio means the money is tied up by customers, clogging the cash flow!

### Group 4: Debt Burden (Overleveraged?)
- **debt_ratio (Debt/Asset)** & **debt_to_equity**: Starting a business costs $10, but the boss borrowed $8 from creditors. If a crisis like Covid hits and they can't afford the interest payments, the company is dead!

👉 **How AI Learns:** It digests the combination of these 14 numbers from 30 VN30 Companies across 3 brutal years to unearth one rule: *"A company with PE < 15, lightning-fast debt collection, and high cash -> Projected for explosive y_return growth."*

---

## Part 2: System Processing Decisions (Domain Decisions)

### Decision 1: Why compute `Log Return` instead of normal Percentages (%)?
**The Story:** If you have $100M. Tomorrow the market crashes **-50%**, you are left with $50M. To reclaim your original $100M, the market next day must skyrocket by **+100%**. (The numbers -50 and +100 are completely asymmetric in arithmetic percentages, so if fed into an AI, Deep Learning algorithms will hallucinate variance errors).
**The Decision:** Use `Logarithmic %`. In the log domain, falling by 50% is `-0.693`, and doubling is `+0.693`. It eradicates the lies of Compound Interest, allowing returns to be perfectly "Additive"!

### Decision 2: The Original Sin of "Look-ahead Bias" (Data Leakage)
**The Sin:** In normal Data Science (like lane drawing for self-driving cars), we usually pour all data into a bucket and randomly shuffle it (`Shuffle = True`), then pick 80% to Learn and 20% to Train/Test.
If we do this with Stocks: It means you let the AI Robot study "2024 Financial Reports" beforehand, and then throw the "2022 Stock Exam" at it to solve! It becomes a Time Machine! Any high score it achieves is absolutely fabricated when used with real money in the real world. => We will not use the standard random shuffle and 80/20 split, but apply Walk-Forward instead.

**The Antidote (`StandardScaler` Feature):** Scaling values to the same range.

### Decision 3: Walk-Forward Validation (The Invigilator Running Machine)
Instead of splitting Data 80% Train / 20% Test outright. You decided to design the **Walk-Forward Rolling Window** method.
**Analogy:** Imagine a Stock Professor. He has historical University Exams from 2011 to 2024.
- Step 1: He gives historical materials from 2011-2015 to 4 students (4 ML Models). Makes them study hard. Then confiscates the materials. Hands them the 2016 Exam to solve fresh. Logs the scores (Predictions File). Wipes the Students' memory.
- Step 2: He gathers fresh unlearned students, gives them materials from 2012-2016. Confiscates. Makes them take the 2017 exam. Logs scores. Wipes memory.
👉 This extremely brutal and closed process ensures the AI will never ever learn by peeking at future Exams, providing a 100% transparent and lethal sandbox!
=> In my project, I chose Walk-Forward with a `window_size = 12`, meaning a fixed 12 quarters, moving forward 1 quarter to test. However, there is another method called **Expanding window**, which accumulates data to train for the next iteration after each test. The choice of Walk-Forward is because in financial markets, "Market Regime Shifts" occur frequently, meaning the market will have different phases (e.g., growth phase, recession phase, sideways phase). If using an Expanding window, the model will be influenced by previous phases, leading to inaccurate predictions. Meanwhile, Walk-Forward Validation helps the model adapt to the different changing phases of the market.

### Decision 4: The Great Wall between Train / Trade
Instead of mixing execution code directly in the AI file. I built the `backtest_engine.py` shell which receives the Weights Matrix.
- The Machine Learning File (ML) remains completely unaware of "Real Dollars Won/Lost". It just outputs an Edict: Tray A holding FPT is looking good, allocate 20% Seed Capital.
- The Arena `bt` framework is an independent Mercenary. It reads the Edict, and walks onto the Live Exchange Simulator. It penalizes your account with Broker Commissions, hits you with Slippage (Failing to match trades at perfect prices) if you chase orders.
The resulting `CAGR` and `Max Drawdown` figures thrown out by the `backtest_engine` are Realized Equity. This is money you can actually withdraw from the bank, not just an academic MSE mathematical hallucination!
