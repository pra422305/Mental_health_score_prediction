# Interview Preparation

## 1-Minute Project Explanation

"I built an end-to-end machine learning project that predicts a mental wellbeing
score from lifestyle and behavioral inputs — things like sleep quality, stress
level, social support, and daily habits. I framed it as a regression problem
since the target is a continuous 0–100 score. I did a full pipeline: data
inspection, EDA to find patterns and check for data leakage, a leakage-safe
preprocessing pipeline using ColumnTransformer, and engineered a few composite
features like a lifestyle score. I trained and compared six regression models —
Ridge Regression came out on top with an R² of about 0.79 — then tuned it with
GridSearchCV and cross-validation. I added feature-importance explainability so
the model isn't a black box, and I built a seven-tab Streamlit dashboard where
you can explore the data interactively and get live predictions with a
disclaimer that this is educational, not diagnostic. It's deployed on Streamlit
Community Cloud."

## Technical Explanation

**Why this problem?** Mental wellbeing prediction is a relatable, socially
relevant regression problem with rich feature engineering opportunities and a
natural story for explainability — all things interviewers like to probe.

**Why regression, not classification?** The target (`Mental_Health_Score`) is
continuous (0–100), so regression is the correct framing. If the dataset instead
had discrete categories (e.g., Low/Medium/High risk), classification would be
appropriate instead.

**Why each model?**
- *Linear/Ridge/Lasso* — fast, interpretable baselines; Ridge/Lasso add
  regularization to control overfitting and handle multicollinearity.
- *Decision Tree* — captures non-linear splits, useful as a baseline for
  tree ensembles.
- *Random Forest* — reduces variance by averaging many trees.
- *Gradient Boosting* — sequentially corrects errors, often strong on tabular data.

**Why preprocessing was required:** Raw data had missing values, mixed
numeric/categorical types, and different scales; models like linear regression
and KNN are scale-sensitive, and most scikit-learn estimators can't handle NaNs
or strings directly.

**Why scaling was required:** Ridge/Lasso regularization penalizes coefficient
magnitude, which is meaningless unless all features are on a comparable scale
— otherwise, a feature measured in the hundreds would be penalized differently
than one measured in single digits.

**Why the final model was selected:** Ridge Regression had the best test R²
(0.79) and the best cross-validation R² (0.757 ± 0.028), meaning it generalizes
consistently — not just a lucky train/test split. Since the target was built
from a broadly linear combination of factors, this is exactly the setting where
linear models outperform trees.

**How overfitting was handled:** Train/test split (80/20), 5-fold cross-validation
during comparison and tuning, regularization (Ridge/Lasso), and comparing CV mean
± std rather than a single test score.

**How missing values were handled:** `SimpleImputer` with median for numeric
columns (robust to outliers) and most-frequent for categorical columns, fitted
only on the training set inside the pipeline to avoid leakage.

**How feature importance works:** For the linear Ridge model, importance is the
absolute value of each standardized coefficient — larger magnitude means the
feature moves the prediction more per unit change. For tree-based models,
`feature_importances_` reflects how much each feature reduces impurity across
all trees/splits.

**How Streamlit deployment works:** The app is pushed to GitHub; Streamlit
Community Cloud pulls the repo, installs `requirements.txt` into a fresh
environment, and runs `app.py`. The app loads the pre-trained `model.pkl` via
`joblib.load` (cached with `@st.cache_resource`) instead of retraining on every
page load, keeping startup fast.

---

## 25+ Interview Questions & Answers

1. **What type of ML problem is this?**
   Supervised regression — predicting a continuous score.

2. **Why not classification?**
   The target is continuous (0–100), not categorical; converting it to bins
   would throw away information unless there's a specific reason to bucket it.

3. **How did you split your data?**
   80/20 train/test split with `random_state=42` for reproducibility.

4. **What is data leakage, and how did you check for it?**
   Leakage is when information from outside the training data (often
   target-correlated info) leaks into features, giving unrealistically good
   scores. I checked pairwise correlation between each feature and the target
   and flagged any above 0.9 — none were found here.

5. **Why use a ColumnTransformer instead of manually preprocessing?**
   It keeps numeric and categorical preprocessing steps bundled with the model
   in one `Pipeline`, so `fit` only ever sees training data, and the exact same
   transformation is guaranteed at inference time — no manual re-implementation
   risk.

6. **What's the difference between Ridge and Lasso?**
   Both add a penalty to coefficient size; Ridge uses an L2 penalty (shrinks
   coefficients smoothly, rarely to zero), Lasso uses an L1 penalty (can shrink
   coefficients exactly to zero, performing feature selection).

7. **Why did Ridge outperform Random Forest here?**
   The synthetic target was generated as a linear combination of features plus
   noise, which is the ideal setting for a linear model; tree ensembles tend to
   shine more on data with strong non-linearities or interactions.

8. **What does R² mean?**
   The proportion of variance in the target explained by the model; 0.79 means
   the model explains about 79% of the variance in wellbeing scores.

9. **What's the difference between MAE and RMSE?**
   MAE is the average absolute error (treats all errors equally); RMSE squares
   errors before averaging, so it penalizes large errors more heavily.

10. **Why use cross-validation instead of a single train/test split?**
    A single split can be lucky or unlucky; k-fold CV averages performance
    across multiple splits, giving a more reliable estimate of generalization.

11. **What hyperparameter did you tune, and how?**
    Ridge's `alpha` (regularization strength) via `GridSearchCV` with 5-fold CV,
    scoring on R².

12. **How much did tuning improve the model?**
    Marginally (R² 0.7900 → 0.7902) — the untuned default alpha was already
    near-optimal for this data, which itself is a valid finding worth reporting.

13. **What is multicollinearity, and did you check for it?**
    When two or more features are highly correlated with each other, which can
    destabilize linear model coefficients. I checked pairwise feature
    correlations and found none above 0.7.

14. **How do you handle missing values?**
    Median imputation for numeric columns (robust to skew/outliers),
    most-frequent imputation for categorical columns — fit only on training
    data.

15. **Why median instead of mean for numeric imputation?**
    Median is robust to outliers/skewed distributions; mean can be pulled by
    extreme values.

16. **What is one-hot encoding, and why use it here?**
    Converts categorical variables (like Gender) into binary indicator columns
    so numeric models can use them; used `handle_unknown="ignore"` so an unseen
    category at inference time doesn't crash the pipeline.

17. **What features did you engineer, and why?**
    Lifestyle_Score, Sleep_Efficiency_Score, and Wellbeing_Support_Score —
    composite scores that summarize related raw features into single,
    more-interpretable signals.

18. **How do you know your engineered features are useful?**
    By checking their correlation with the target and their contribution in
    feature importance after training — several ranked in the top features.

19. **What is the bias-variance tradeoff, and how does it apply here?**
    Bias is error from overly simple assumptions; variance is error from
    sensitivity to training data. Ridge regularization trades a small amount of
    bias for reduced variance, which is why it generalized more consistently
    (lower CV std) than the unregularized linear model in some runs.

20. **How would you detect overfitting in this project?**
    Large gap between training R² and test/CV R²; here, CV mean (0.757) and
    test R² (0.79) are close, so overfitting isn't a major concern.

21. **What would you do if the model underperformed?**
    Gather more/better features, try polynomial or interaction terms, try
    non-linear models with more tuning, or check for label noise in the target.

22. **Why did you cache the model in Streamlit?**
    `@st.cache_resource` avoids reloading the model from disk on every user
    interaction, keeping the app responsive.

23. **How would you deploy this at scale?**
    Wrap the trained pipeline behind a lightweight API (e.g., FastAPI), containerize
    it, and deploy behind a load balancer; keep the Streamlit app as a thin client
    calling that API rather than loading the model in-process.

24. **What ethical considerations does this project raise?**
    It must never be presented as a diagnostic tool; predictions should include
    clear disclaimers, and any real deployment would need informed consent,
    privacy protections for sensitive self-reported data, and human oversight.

25. **How would you extend this into a classification problem?**
    If a labeled dataset provided discrete risk categories, I'd swap regression
    metrics for classification metrics (accuracy, precision, recall, F1,
    ROC-AUC), likely address class imbalance, and use `predict_proba` for
    confidence scores.

26. **Why Streamlit instead of Flask/Django for the dashboard?**
    Streamlit turns Python scripts into interactive web apps with minimal
    boilerplate — ideal for rapid, data-centric dashboards and fast portfolio
    iteration, though Flask/Django would give more control for production-grade
    APIs.

27. **What does `handle_unknown="ignore"` do in OneHotEncoder, and why does it matter for deployment?**
    It prevents a crash if the app receives a category value not seen during
    training (e.g., a new Gender option); without it, `.transform()` would raise
    an error in production.

28. **How did you avoid retraining the model every time the app runs?**
    The trained pipeline is serialized with `joblib.dump` after training and
    loaded once via `joblib.load`, cached with `@st.cache_resource` in
    `app.py`.
