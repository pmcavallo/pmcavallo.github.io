# Cross-Cloud AutoML Shootout: Lessons from AWS, GCP, and BigQuery  

When I kicked off the **Cross-Cloud AutoML Shootout**, the idea was simple: put AWS and GCP side by side, train on the same dataset, and see which delivered the better model with less friction. What started as a straightforward benchmark quickly turned into something bigger, a case study in how different cloud philosophies shape the experience of doing machine learning.  

Just like in banking, where model development often collides with regulatory guardrails, this project revealed how *quotas, hidden constraints, and pricing structures* can be as important as the algorithms themselves.  

I could explain the data, but why not let Gemini do it from within GCP? My simulated data had no descritions to the variables, however, Gemini is able to read the data and and create the description itself:

![Description](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/var_description.png?raw=true)
---

## AWS SageMaker Autopilot: Transparent, Predictable, Done in 30 Minutes  
AWS set the tone early. With Autopilot, training was smooth:  

- **Cost:** About **$10** for 30 minutes of runtime.  
- **Models:** 10 candidate models trained and evaluated.  
- **Transparency:** Pay-per-second billing gave full control of time and budget.  

Screenshots from AWS told the story:

![Leaderboard](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/leaderboard.png?raw=true)

The AWS Autopilot leaderboard summarizes how multiple candidate models performed on the dataset. 
- Default model: The top row is flagged as the default. It’s selected automatically by Autopilot because it offered the best trade-off among metrics (highest F1 at 55.08%).
- Performance range:
    - Accuracy varies between ~53% and ~57%.
    - AUC is consistently low (0.53–0.56), suggesting the models struggled to separate classes well.
    - F1 scores cluster around 33–55%, showing the default model is substantially better than several alternatives.
- Precision vs Recall:
    - Default model: Precision 49.5%, Recall 62.1% — meaning it finds more true positives but at the cost of some false positives.
    - Other models trade precision and recall differently (e.g., models with ~56–57% accuracy tend to have recall closer to 24%).
- Consistency: Many rows repeat near-identical metrics (e.g., accuracy ~56.8%, AUC ~0.559). These are often variants of the same underlying algorithm with slightly different hyperparameters.

![Explainability](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/explainability.png?raw=true)

This graph is AWS Autopilot’s explainability output and shows feature importance + partial dependence:
- Left panel – Column impact (global feature importance):
    - DTI (Debt-to-Income ratio) is the most important predictor, contributing ~23% of the model’s predictive power.
    - Other strong contributors: tenure_months (15%), LTV (11%), age (11%), and credit_lines (6%).
    - ogether, the top 4–5 features dominate the model’s decisions.
- Right panel – Partial dependence plot (DTI vs prediction impact):
    - Shows how changes in DTI affect the predicted probability of the target (default_12m).
    - This is counterintuitive compared to credit risk practice (normally higher DTI → higher default risk). This is a risk with simulated data.

![Metrics](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/metrics.png?raw=true)

The model has learned some signal from the data but struggles with clear separation between classes. Accuracy near 54% suggests it’s only marginally stronger than a majority-class baseline, while the F1 score confirms modest effectiveness in balancing false positives and false negatives.

**Key Lessons:**
  
The message from AWS was clear: *simple, transparent, predictable*.  

---

## GCP Vertex AI AutoML: Quotas, Quotas, and More Quotas  
GCP was a different experience. Instead of cost being the first challenge, it was **quotas**.  

- The first runs failed at the preprocessing stage due to **Dataflow N1 CPU limits**.  
- After that, training failed again due to **Custom Model Training CPU quotas**.  
- Both required manual support requests, even on a paid account.  

Even after approvals (400 N1 CPUs, 150 custom CPUs), another roadblock appeared: some quotas were hidden entirely from the UI. Without a paid support plan, progress stalled.  

This exposed a design philosophy: GCP protects against runaway costs with strict guardrails, but for independent projects, those guardrails can block you entirely.  

---

## The Pivot: BigQuery ML to the Rescue  
Instead of giving up on GCP, I pivoted to **BigQuery ML**. Unlike Vertex AI, BigQuery ML uses straightforward data-scanning quotas, not node hours.  

- **Quotas:** The first **1 TiB per month** is free, and our dataset was tiny (3,500 rows).  
- **Clarity:** No hidden quotas, usage is visible under IAM → Quotas.  
- **Cost control:** You pay for queries, not idle infrastructure.  

Screenshots captured the turnaround:  

![Metrics](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/metrics.png?raw=true)

![Matrix](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/matrix.png?raw=true)

![feature](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/metrics.png?raw=true)

This “positive class threshold” slider is a great feature in BigQuery. What it is:
- By default, classification models output a probability between 0 and 1 for the positive class (here: default_12m = 1).
- The threshold tells the model:
    - If the predicted probability ≥ threshold → classify as 1 (positive), otherwise classify as 0 (negative).
    - The default is 0.5, but you can move it up or down to change how conservative or aggressive the model is in predicting positives.

I then use SQL to find the F1-maximizing threshold, then set the slider to (about) that value:

![f1](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/F1-maximizing.png?raw=true)

The new results are:

![Metrics](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/metrics2.png?raw=true)

![Matrix](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/matrix2.png?raw=true)

Interpretation:
- At threshold = 0.49, the model balances precision and recall in a way that maximizes F1.
- Compared to the default 0.50 threshold, the optimized one slightly boosts recall (the model catches more positives) while keeping precision at an acceptable level.
- This results in a much stronger F1 score (0.739 vs ~0.50 at default).

**Key Lessons:**

With BigQuery ML, we finally had a working GCP comparison point, and a reminder that sometimes the “backup option” is the smarter long-term play.  

---

## Lessons Learned  
This shootout wasn’t just about models. It surfaced the *realities of working across clouds*:  

- **AWS:** Transparent billing, fast results, predictable.  
- **GCP Vertex AI:** Hidden quota gates make it risky for small teams.  
- **BigQuery ML:** A pragmatic middle ground — clear quotas, flexible billing, and enough power for real experiments.  

---

## Why This Matters  
In AI and data science, it’s not just accuracy metrics that matter, it’s whether you can get to those metrics at all. The Shootout shows how **infrastructure choices can shape experimentation itself**.  

Cross-cloud isn’t just a buzzword. It’s a survival skill.  

---
