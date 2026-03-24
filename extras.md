## PRIMARY DATASETS (kaggle):

1. **"netflix movies and tv shows till 2025"** - curated from tmdb, includes titles/genres through 2025 [DataCamp](https://www.datacamp.com/courses/exploratory-data-analysis-in-python)
2. **"netflix tv shows and movies"** (victor soeiro) - 5k+ titles, 15 columns: justwatch id, imdb score/votes, tmdb popularity/score [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/)
3. **"netflix 2025: user behavior dataset"** - 210k+ records (synthetic but useful for engagement patterns)

## PROJECT ANGLES (pick your poison):

### OPTION A: **content evolution & strategy**
**title:** "streaming platform content strategy: exploratory analysis of netflix catalog dynamics 2019-2025"

**analysis vectors:**
- temporal: release year distribution, addition patterns (covid spike? post-streaming-wars changes?)
- geographic: production country trends (korean wave? regional originals vs licensed?)
- genre evolution: what's declining? (rom-coms dead? true crime explosion?)
- ratings correlation: imdb vs tmdb scores, vote patterns
- content type shift: movies vs series ratio over time
- runtime analysis: are things getting shorter? (tiktok brain impact)

**technical hits:**
- time series decomposition (seasonality in releases)
- correlation matrices (rating vs votes vs popularity)
- geographic clustering (k-means on country production patterns)
- sentiment proxy via ratings distributions
- pareto analysis (80/20 of viewership via votes)

### OPTION B: **competitive positioning** (if multi-platform data)
**title:** "the great fragmentation: comparative analysis of streaming catalog strategies"

**if you can merge multiple platform datasets:**
- content overlap analysis (same titles across platforms)
- genre differentiation strategies
- pricing vs catalog size correlation
- regional availability patterns

**context from search:**
- netflix 19%, prime 20%, disney+ 14% us market share q3 2025 [GitHub](https://github.com/FuadAnalyst/EDA-Exploratory-Data-Analysis)
- netflix: 16 oscar noms 2025, 30 emmys (tied with hbo) [Kaggle](https://www.kaggle.com/code/ekami66/detailed-exploratory-data-analysis-with-python)
- only 21% of global series available on top 8 platforms, 88% of movies not on major streamers [Kaggle](https://www.kaggle.com/code/agrawaladitya/step-by-step-data-preprocessing-eda)

### OPTION C: **quality vs popularity paradox** (my pick)
**title:** "decoupling quality from virality: an exploratory analysis of netflix content performance metrics"

**core question:** does "good" content perform well, or is it all marketing/algorithm?

**analysis:**
- imdb score vs imdb votes (critical acclaim vs mass appeal)
- tmdb popularity vs tmdb score (different audience demographics)
- genre-specific thresholds (what's "good" for reality tv vs prestige drama?)
- outlier detection (cult classics with low votes but high scores)
- recency bias (new content juice vs evergreen library)
- franchise power (sequel/spinoff performance vs standalone)

**viz opportunities:**
- quadrant charts (high quality/low popularity vs viral trash)
- distribution comparison by genre
- temporal heatmaps (when do things peak?)
- network graphs (cast/director collaboration patterns if you have that data)

## DATA QUALITY CHECK:

**pros:**
- established kaggle datasets (well-maintained)
- multiple rating sources (imdb/tmdb triangulation)
- temporal span (can show trends)
- decent size (5k-8k titles = robust analysis)

**cons:**
- no actual viewership numbers (netflix guards this)
- rating bias (self-selection, bots)
- missing values likely in older content
- no cost/budget data (would be gold for roi analysis)

## TECHNICAL STACK:

```python
# core
pandas, numpy

# viz
seaborn, matplotlib
plotly if you want interactive (good for presentations)

# stats
scipy.stats (correlation tests, anova)
sklearn (clustering, pca if you want to flex)

# optional spice
wordcloud for genre/title analysis
networkx if doing collaboration graphs
```

## PROJECT STRUCTURE:

```
1. data acquisition & cleaning
   - handling nulls (rating columns especially)
   - dtype conversions (dates, numeric ratings)
   - duplicate detection (same title, different years?)
   - outlier treatment (0-vote entries, test content)

2. univariate analysis
   - rating distributions (normal? bimodal?)
   - release year trends
   - genre frequencies
   - runtime patterns

3. bivariate/multivariate
   - score vs votes correlation
   - genre vs rating relationships
   - temporal patterns (seasonality, growth)
   - geographic production trends

4. segmentation
   - content clusters (k-means on features)
   - high-performer identification
   - underrated gem detection

5. key findings synthesis
   - pattern summary
   - anomaly highlights
   - strategic implications (if you're feeling spicy)
```

## PRO MOVES:

- **combine datasets:** merge netflix + imdb metadata for richer features
- **external context:** reference streaming wars stats from search (market share trends, strategy shifts)
- **domain insight:** avg viewing time per account dropped from 132 to 106 min/day in 2024 [Kaggle](https://www.kaggle.com/code/ekami66/detailed-exploratory-data-analysis-with-python) - attention economy analysis
- **critical angle:** don't just describe, INTERPRET. why does genre X have rating inflation? what does this mean for content strategy?

## AVOID:

- pure description ("this is a histogram of ratings") - NO
- ml predictions (not EDA scope)
- recommendation systems (different project)
- excessive data cleaning documentation (do it, don't write a novel about it)

## DELIVERABLE TIPS:

- **intro:** set context with streaming wars landscape (use search stats)
- **methodology:** brief, clinical
- **analysis:** 70% of effort - show patterns, correlations, insights
- **viz:** clean, annotated, interpretable (not chart vomit)
- **conclusion:** 3-5 key findings with strategic implications

**final boss move:** if you want MAXIMUM DEPTH, scrape current netflix catalog yourself via unofficial apis (justwatch, reelgood) and compare with kaggle historical data. shows "what changed" longitudinally. might be overkill for 602 but would absolutely body the project.

grab the kaggle datasets tonight (literally - form due today). want me to draft a quick project proposal doc you can submit?