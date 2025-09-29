# Prompt Engineering Lab: From Zero-Shot to Production-Ready Systems

This notebook documents the journey of moving from basic prompt engineering (zero-shot classification) to a production-ready system that integrates validation, schemas, and retrieval augmentation (RAG).

---

## Step 1: API Setup and First Test Prompt

I started by authenticating with the OpenAI API. This is required because every request I send to the model needs to be tied to my account. The key is stored securely with `getpass` so it never shows up in the code.

```python
import sys, subprocess, getpass, os
from openai import OpenAI

# Upgrade SDK
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "openai>=1.40.0"])

# Get API key
OPENAI_API_KEY = getpass.getpass("Paste your OpenAI API key (starts with 'sk-'): ")
assert OPENAI_API_KEY.startswith("sk-"), "Key must start with 'sk-'"

# Initialize client
client = OpenAI(api_key=OPENAI_API_KEY)
print("✅ OpenAI client ready.")
```

This ensures the environment is ready to interact with the OpenAI API securely.

I verified connectivity by sending a trivial test prompt. The model replied successfully, confirming the setup.

```python
resp = client.responses.create(
    model="gpt-4o",
    input=[{"role": "user", "content": "Say ok"}],
    temperature=0
)
print(resp.output_text)
```

_Output:_

```
OK! How can I assist you today?
```

---

## Step 2: Dataset Creation

Next, I created a dataset of credit policy notes. These snippets represent real-world eligibility, pricing, fraud, and collections policies. Some are straightforward, while others are tricky edge cases designed to test the limits of prompt accuracy.

```python
rows = [
    # --- originals (4) ---
    ("Borrowers with FICO >= 700 qualify for best pricing.", "Pricing"),
    ("Applications with DTI above 45% are not eligible.", "Eligibility"),
    ("Accounts with two missed payments move to collections.", "Collections"),
    ("Flag transactions with mismatched addresses for review.", "Fraud"),

    # --- Pricing vs Eligibility confusers ---
    ("Prime tier applies to FICO 720+ with a 25 bps rate reduction.", "Pricing"),
    ("Minimum credit score 680 required to apply.", "Eligibility"),
    ("Applicants with FICO 740 are offered a points discount.", "Pricing"),
    ("DTI must be <= 36% for prime offers.", "Eligibility"),

    # --- Collections vs Fraud boundary ---
    ("If payment is 60 days late, send a delinquency notice.", "Collections"),
    ("Unrecognized device + inconsistent SSN → escalate for identity verification.", "Fraud"),

    # --- Mixed sentences (contain multiple signals) ---
    ("DTI must be under 40%; rate is 7.1% for approved borrowers.", "Eligibility"),
    ("Late fees apply after 15 days; repeated chargebacks may indicate account takeover.", "Fraud"),

    # --- Negations & hedges ---
    ("No change to APR tiers this quarter.", "Pricing"),
    ("This is not suspected fraud; route to normal collections.", "Collections"),

    # --- Colloquial / indirect phrasing ---
    ("Score in the high 600s? You’re probably in the better-price bucket.", "Pricing"),
    ("When debt eats half your paycheck, we decline the app.", "Eligibility"),

    # --- Lookalikes that should separate ---
    ("Two consecutive NSF events move the account to collections workflow.", "Collections"),
    ("Synthetic name patterns (e.g., repeated vowels) should be auto-flagged.", "Fraud"),

    # --- Edge constraints: LTV/DTI mixes ---
    ("Max LTV is 80% and DTI must be <= 43% for jumbo loans.", "Eligibility"),
    ("Preferred tier gets 0.25% rate cut with autopay enrollment.", "Pricing"),
]

df_ext = pd.DataFrame(rows, columns=["text","label"])
Path("data").mkdir(exist_ok=True, parents=True)
df_ext.to_csv("data/credit_notes_extended.csv", index=False, encoding="utf-8")
len(df_ext), df_ext.head(8)
```

This step is important: we don’t just want “easy” examples; we want borderline cases where the model might confuse categories. This is how we stress-test prompts.

---

## Step 3: Zero-Shot Baseline vs Engineered Prompts

I compared two approaches:

- **Baseline (zero-shot):** plain natural language prompt.  
- **Engineered prompt:** few-shot examples + structured instructions.

![Baseline vs Engineered Confusion Matrix](baseline_vs_engineered.png)

The confusion matrices above show that the baseline confused `Eligibility` with `Pricing` and sometimes mixed `Fraud` with `Collections`. The engineered prompt, by contrast, achieved higher accuracy (0.95 vs 0.90) by reducing these confusions.

This demonstrates that **prompt design matters**: adding structure and examples reduces ambiguity.

---

## Step 4: Schema Enforcement

We can extende the task by asking the model to output structured fields like minimum credit score, max DTI, and category. By enforcing a schema, we ensured that outputs were machine-readable.

```json
{
  "raw_inputs_model": "{\n  \"amount\": 250000,\n  \"rate_annual_pct\": 6.5,\n  \"term_months\": 360,\n  \"income_monthly\": 9000,\n  \"obligations_monthly\": 1200\n}",
  "parsed_inputs": {
    "amount": 250000.0,
    "rate_annual_pct": 6.5,
    "term_months": 360,
    "income_monthly": 9000.0,
    "obligations_monthly": 1200.0
  },
  "tool_result": {
    "monthly_payment": 1580.17,
    "dti": 0.309
  },
  "final_decision": {
    "approve": true,
    "reason": "DTI 0.309 <= 0.36",
    "monthly_payment": 1580.17,
    "dti": 0.309
  }
}
```

Every row passed schema validation (20/20). This matters because free text is unreliable in downstream pipelines; schema outputs let us connect LLMs with databases, dashboards, or loan decisioning systems.

---

## Step 5: Retrieval-Augmented Generation (Mini RAG)

Finally, I implemented a lightweight Retrieval-Augmented Generation system. Instead of trusting the model to recall facts from training data, I grounded answers in a local corpus of policy snippets.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

snips = {
 "001": "DTI cap is typically 36% for prime borrowers.",
 "002": "Minimum credit score 680+ is required for standard eligibility.",
 "003": "Collections process begins after 60 days of nonpayment.",
 "004": "Pricing discounts of 25 bps apply for autopay enrollment."
}

ids, corpus = list(snips.keys()), list(snips.values())
vec = TfidfVectorizer().fit(corpus)
X = vec.transform(corpus)

def retrieve(query, k=1):
    q = vec.transform([query])
    sims = cosine_similarity(q, X).ravel()
    idx = sims.argsort()[::-1][:k]
    return [(ids[i], corpus[i], float(sims[i])) for i in idx]

def answer_with_grounding(question):
    hits = retrieve(question, k=1)
    doc_id, passage, score = hits[0]
    prompt = f"""Answer the question **only** using the passage below.
If the passage is insufficient, say "insufficient".
Return JSON: {{"answer": string, "doc_id": string}}

Passage (doc_id={doc_id}):
{passage}

Question: {question}
"""
    r = client.responses.create(model="gpt-4o", input=[{"role":"user","content":prompt}], temperature=0)
    import json, re
    try:
        obj = json.loads(r.output_text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", r.output_text)
        obj = json.loads(m.group(0)) if m else {"answer": r.output_text.strip(), "doc_id": doc_id}
    obj["retrieval_score"] = round(score,3)
    return obj

print(answer_with_grounding("What is the typical DTI cap?"))
print(answer_with_grounding("When do collections begin?"))
```

The retrieval step finds the **closest policy snippet** to the question, then I pass only that snippet to the model.  

Example output:
```json
{"answer": "36%", "doc_id": "001", "retrieval_score": 0.611}
{"answer": "After 60 days of nonpayment.", "doc_id": "003", "retrieval_score": 0.362}
```

This guarantees that answers are **grounded in context**, not hallucinated.

---

## 8. Conclusion

Throughout this lab we saw the **progression of prompt engineering maturity**:

1. Zero-shot prompts: fast but unreliable.  
2. Engineered prompts: templates + few-shot examples reduce confusions.  
3. Schema enforcement: forces structured, machine-readable outputs.  
4. Validation and guardrails: catch malformed JSON, enforce domain rules.  
5. Tool/logic integration: models cooperate with downstream calculators.  
6. Retrieval augmentation: grounding prompts in real context.  

**Key takeaway:** Prompt engineering is not about “tricking” a model, it’s about **designing an interface** between human intent, model reasoning, and business logic.  
It saves cost, reduces error, speeds workflows, and enables safe deployment.

By running confusion matrices, schema checks, and retrieval grounding, I saw firsthand that prompt engineering is a system design discipline, not just clever wording.  

My process now looks like this:

- Start with something simple.  
- Add worked examples.  
- Impose schemas.  
- Layer in validation.  
- Integrate with tools.  
- Ground in retrieval contexts.  

That’s how we go from “prompting” to **production-ready prompt engineering**. It’s a vital skill because in applied AI, reliability and governance are just as important as creativity.

