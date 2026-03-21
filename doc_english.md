# The Anatomy of Quant Models: An Intuitive Guide 🇬🇧

If you are a regular Software Engineer or someone just stepping into Quantitative Finance, looking at this massive codebase might make you wonder: *"Why overcomplicate things? Why aren't we just dumping everything into a single script?"*

This document will stray away from dry Mathematical formulas. Instead, it explains the **Architectural Mindset using everyday analogies.**

---

## Part 1: Decoding Feature Engineering (The 14 Health Indicators)
Instead of letting the AI blindly read news, we force it to evaluate a company through 14 distinct "X-Ray" financial indicators (defined in our `features` array). We categorize these into 4 Pathology Groups to let the AI forecast if the stock will skyrocket or crash in the next 3 months:

### Group 1: Profitability Health (Do you make real money?)
- **EPS (Earnings Per Share)**: The company made a $10M profit this year. Divided by 10M shareholders, how much real value did one share generate?
- **ROE (Return on Equity)**: If you inject $100 of pure flesh-and-blood cash into the company, how much interest is returned to you a year later? 
- **net_income_ratio (Net Profit Margin)**: For every $100 of revenue, after deducting rent, power, taxes... how much net cash actually goes into the boss's pocket?

### Group 2: Valuation (Are we buying an overpriced bubble?)
- **PE (Price to Earnings)**: The company makes $1k a year in profit. The market is shouting the stock price is $20k to own it. PE = 20 (You must hold the stock for 20 years to break even). A very high PE often signals an overbought bubble (or sky-high expectations).
- **PB (Price to Book)**: If the company liquidates all its factories and assets into scrap, it gets $1B (Book Value). The market currently prices the firm at $2B. Thus, PB = 2.

### Group 3: Liquidity Bloodline (Will it go bankrupt tomorrow from missing cash?)
- **cur_ratio (Current Ratio)**, **quick_ratio**, & **cash_ratio**: These simply ask: *If creditors aggressively demand their short-term loans back tomorrow morning, does the company have enough liquid cash in the safe to pay instantly?* A terrible `cash_ratio` implies sudden death vulnerability.
- **acc_rec_turnover (Accounts Receivable Turnover)**: Translates to **"Debt Collecting Skill"**. A high ratio means the company sells goods on credit and successfully claws back cash into its vault swiftly!

### Group 4: Debt Burden (Overleveraged?)
- **debt_ratio (Debt/Asset)** & **debt_to_equity**: Starting a factory costs $10, but the boss borrowed $8 from the bank. If Covid hits and they can't afford the interest, the company is dead.

👉 **How AI Learns:** It digests this 14-variable cocktail over 30 companies inside the VN30 index across 3 brutal years to unearth one hidden truth: *"Any anonymous creature with PE < 15, lightning-fast debt collection, and high cash -> Is projected for explosive y_return."*

---

## Part 2: Dissecting Domain System Decisions

### Decision 1: Why compute `Log Return` instead of normal Percentages (%)?
**The Story:** If you have $100. Tomorrow the market crashes **-50%**, you are left with $50. To reclaim your $100 base, the market must skyrocket by **+100%**. (The numbers -50 and +100 are heavily asymmetric when using arithmetic percentages, causing Deep Learning algorithms to hallucinate variance errors).
**The Decision:** We use `Logarithmic Returns`. In the log domain, falling by half is `-0.693`, and doubling is `+0.693`. It eradicates the lies of Compound Interest, allowing returns to be perfectly, symmetrically "Additive"!

### Decision 2: The Ultimate Sin of "Look-ahead Bias" (Data Leakage)
**The Sin:** In normal Data Science (like image classification), we usually pour all data into a bucket and randomly shuffle it (`Shuffle = True`) to split 80% Train, 20% Test.
If we do this in Stocks: The AI will memorize the "2024 Financial Crash Report" in the training dataset, and then magically use that memory to take a test on "Stocks in 2022"! It becomes a Time Machine! Any high score it achieves is absolutely fabricated when used with real money in Production.
**The Antidote (`StandardScaler` strictness):** Our Scaling function is explicitly commanded to only call `.fit()` on past Historical Training data. It is absolutely banned from peeking at the statistical peaks/troughs of the Test Set (Future Exams).

### Decision 3: Walk-Forward Validation (The Unforgiving Supervisor)
Instead of splitting Data 80% / 20% evenly, we designed an architecture called **Walk-Forward Rolling Window**.
**Analogy:** Imagine a Stock Professor with historical final exams from 2011 to 2024.
- Step 1: He gives historical books from 2011-2015 to 4 students (4 ML Models). He lets them study. Then he confiscates the books, slams the 2016 Exam on their desks, and grades them. He logs the scores, then **Wipes their Memory**.
- Step 2: He gathers fresh unlearned students, gives them books from 2012-2016. Confiscates. Hands them the 2017 Exam to take blind. Logs score. Wipes memory.
👉 By marching forward sequentially, the AI has perfectly zero mechanical capability of predicting the future since it hasn't seen it yet. A 100% transparent and lethal sandbox!

### Decision 4: The Great Wall between Train and Trade
Instead of blending trading Execution code inside our AI python files, we built the `backtest_engine.py` shell.
- The Machine Learning File remains completely unaware of "Real Dollars Won/Lost". It just outputs an Edict: *Box A holding FPT is good, allocate 20% Seed Capital.*
- The Arena `bt` framework is an independent Mercenary. It reads the Edict, and walks onto the Live Exchange Simulator. It penalizes your account with Broker Commissions, hits you with Slippage (Failing to match trades at perfect prices), and reports real loss.
The resulting `CAGR` and `Max Drawdown` figures thrown out by the backtest engine are Realized Equity. This is money you can actually withdraw from the bank, not just academic MSE mathematical hallucinations!
