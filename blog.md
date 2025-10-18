---
layout: default
title: Blog          
permalink: /blog/
---

Welcome to the blog. Here I share short, practical notes from building my portfolio projects. Expect lessons learned, trade offs I faced, and the reasoning behind architecture and tool choices. If you want the code and demos, visit the [Projects](/projects/) page. Start with the latest post below.

---

# Building a Governed AI System: My Experiment with Codex (10/18/2025)

I decided to test a question I’d been circling for months: can an AI coding assistant like Codex not only accelerate development, but help build a governed, reproducible machine learning system from the ground up? 

I didn’t want boilerplate code or a “copilot” writing syntax. I wanted to see if Codex could act as a structured collaborator; a system that writes, tests, and refines code within real governance boundaries. So I set up the experiment: to build a full orchestration framework for credit risk models, with the same discipline heavily regulated industries demand for audit-ready models.

**The Context**

The project, *Codex Orchestrator*, began with a clear mission: to bring AI-assisted governance to traditional credit risk modeling. I wanted two models; a classic logistic scorecard and a modern XGBoost challenger — running side by side in a sandbox where everything was tracked, reproducible, and observable.

Codex would be my engineering partner: implementing, debugging, and refactoring through structured prompts, while I handled reasoning, architecture, and validation.

**The Collaboration**

The workflow unfolded like a true partnership. I gave Codex concrete goals (“Add calibration persistence with fallbacks”), constraints (“Preserve reproducibility”), and context about how the system was structured. It would then produce incremental diffs, which I reviewed line by line.

At one point, the Streamlit “Run Monitoring Now” button broke; a relative import error that caused the dashboard to fail silently. Within a single iteration, Codex identified the root cause, converted the imports to absolute paths, implemented cached data loaders, and even added test coverage to validate refresh behavior.  

When we introduced calibration, Codex automatically built Platt and Isotonic selection logic, serialized the calibrators, and updated downstream pipelines to use them transparently. The outputs were not just functional, they were contract-compliant, meaning the system now mirrored real-world regulatory workflows.

**The Turning Point**

Halfway through, I realized something subtle but important: Codex wasn’t just generating code, it was preserving governance logic. It never broke reproducibility. Every artifact, models, metrics, and reports, remained deterministic unless explicitly told to vary.  

That led me to a second phase of the experiment: *stochastic mode*. I wanted to see whether Codex could help me introduce variability safely. I asked it to design a toggle in the dashboard that would inject random seeds, rerun monitoring, and regenerate all metrics and plots in real time, but without compromising the default deterministic mode.

The result was elegant. The deterministic (OFF) mode preserved governance sign-off integrity, while the stochastic (ON) mode refreshed visuals dynamically for testing and exploration. Every test still passed. Every cache refreshed. And the system behaved exactly as intended.

**What I Learned**

Working with Codex felt less like outsourcing and more like orchestrating a process. It didn’t replace my expertise, it amplified it. I still needed to reason through model architecture, regulatory logic, and interpretability; Codex handled syntax, testing, and consistency with near-perfect discipline.  

What impressed me most wasn’t speed, it was *stability*. Even as the codebase grew more complex, the system never lost its shape. Each component; calibration, drift monitoring, stochastic mode, and governance scaffolding, connected seamlessly, as if we were co-writing a long-form technical essay in code.

I also learned that governance isn’t a bottleneck when AI is part of the loop. Codex made it faster to stay compliant. By baking governance into each step — artifact versioning, metadata tracking, test enforcement — it turned regulation into a design constraint, not a limitation.

**The Evaluation**

Looking back, the experiment redefined how I think about AI in engineering.  
- Codex *understood structure*: it maintained logical boundaries and respected reproducibility.  
- It *adapted quickly*: when I introduced calibration or monitoring complexity, it expanded the system coherently.  
- And it *taught me discipline*: the more precise my prompts became, the more reliable its outputs were.  

But the biggest insight came from what it didn’t do. Codex never tried to “be creative.” It optimized for clarity and structure, qualities human engineers often lose when rushing through build cycles. It showed that AI doesn’t need to outthink us; it needs to out-organize us.

**Where It Ends (for Now)**

By the end, the *Governed Risk Control Tower* became fully functional. The dashboard ran in-process monitoring, refreshed on demand, and tracked model stability across calibration and drift metrics. The only missing piece was interpretation; an AI-assisted tab that could analyze results in natural language and explain model health in plain terms.  

That’s where I’ll take it next: an AI-driven interpretability layer that can narrate what the metrics mean, flag concerns, and recommend retraining actions. It’s not just about models learning from data anymore, it’s about *systems learning from themselves*.

**Closing Reflection**

This project wasn’t about proving that Codex can code. It was about testing whether AI can collaborate under governance, whether it can operate within the same constraints we expect from human engineers in regulated systems.  

The answer is yes, but only when the human stays in the loop as the orchestrator. Codex executes; I interpret. Codex tests; I validate. Together, we built something that’s more than a project, it’s a case study in what responsible AI development might actually look like.

What started as an experiment with an AI coder ended as a reflection on trust, governance, and human-machine alignment, written not just in Markdown, but in code itself.

---

# Building Trustworthy AI: Why Guardrails Matter More Than You Think (10/10/2025)

I was explaining my CreditIQ project to a friend when he asked the question that stops every AI engineer cold: "Wait, you're letting an AI agent approve loans? What if it screws up?" (Important disclaimer, the [project](https://pmcavallo.github.io/CreditIQ/) uses simulated data).

He wasn't being difficult. He was asking the most important question in AI deployment. This isn't about whether Claude can write good poetry or summarize articles. This is about a system that could approve a predatory loan to someone who can't afford it, deny a good borrower based on spurious correlations, or discriminate against protected groups without anyone noticing until it's too late. To be very clear, this is a personal project done with synthetic data I generated specifically for this experiment.

The question isn't "Can AI do this?" anymore. We know it can. The real question is: "Should we let it, and if so, how?"

For CreditIQ, I spent as much time building guardrails as I did building the AI system itself. What I learned surprised me.

When you first build an AI agent system, the temptation is simple. Get the application data, send it to the agent, return the decision. Trust the AI to make the right call.

This is how disasters happen.

Traditional software fails in obvious ways. It crashes, throws errors, returns the wrong type. You write unit tests, catch the bugs, ship it. AI agents are different. They fail subtly, with plausible-sounding explanations for decisions that are fundamentally wrong. They don't crash, they confidently give you answers that sound reasonable but aren't grounded in your actual data.

A bug in traditional software might approve the wrong loan amount. An unsupervised AI agent might invent a new approval criterion based on proxies for protected attributes, systematically discriminate against a demographic group, and generate explanations that sound perfectly reasonable. All while staying within its API constraints.

You can't just trust an AI system. You have to constrain it.

To deal with this isue, I designed a three-layer trust model. Think of it as concentric circles of safety, each one catching what the previous layer might miss.

The first layer is simple: the agent doesn't get to decide. It recommends. For every edge case that gets routed to the AI agent, there's still a machine learning model providing a second opinion. When the stakes are high, loans over twenty-five thousand dollars, cases where the ML model and agent disagree by more than thirty percentage points, or any conditional approval with modified terms, a human reviewer sees both opinions before making the final call.

This isn't about not trusting AI. It's about understanding that agents handle volume while humans handle stakes. You can't hire enough underwriters to manually review every application, but you also can't abdicate responsibility for decisions that could ruin someone financially. The agent is an expert assistant, not a replacement for judgment.

The second layer constrains what the agent can even consider. There are hard rules it cannot override. If a factor is below acceptance levels, the agent can look at all the alternative data it wants, bank balances, rent payment history, whatever, but it cannot approve the loan. Below that threshold is a regulatory red line, period. 

The third layer is the audit trail. Every single decision logs everything: the full application data, the ML model's prediction with SHAP values showing which features mattered, why it was flagged as an edge case, what the agent analyzed and recommended, what guardrails checked, and what the final decision was. If a customer appeals or a regulator audits or the system starts performing poorly, you can trace backward through the entire decision chain.

This isn't just for compliance. It's how you improve the system. When you see patterns of agent-ML disagreement, or cases where human reviewers override the agent, or demographic groups with different approval rates, the audit trail tells you where to look. You can't fix what you can't see.

Building guardrails for individual decisions isn't enough. You also have to ensure the system doesn't systematically discriminate.

Credit decisions are subject to fair lending laws. You cannot discriminate based on race, gender, age, religion, national origin, marital status, or disability. The problem is that AI agents are trained on human text, which contains all of society's biases. Even if you explicitly exclude protected attributes from your data, the agent might find proxies. Zip code correlates with race through residential segregation. Type of employment can proxy for various protected classes.

So I built automated fairness testing that runs monthly. The system splits decisions by geographic region as a demographic proxy, calculates approval rates for each region, and checks whether any region falls below eighty percent of the highest approval rate. That's the four-fifths rule from EEOC guidance. If there's a violation, the system pauses agent decisions in that region, audits the reasoning for biased patterns, and doesn't resume until we've validated the fix.

The uncomfortable truth is that you can't just test for fairness once at launch. Data distributions drift. Edge cases reveal biases slowly. Systemic fairness requires continuous monitoring, because one fair decision doesn't prove the system is fair.

My first version trusted agent confidence scores. If the agent said it was highly confident in a decision, I'd accept it without additional checks. The problem? Agent confidence doesn't mean the decision is correct. It just means the agent is confident, which could mean it's confidently wrong.

The fix was to stop trusting confidence and start validating outcomes. Track agent-approved loans over time and measure their default rate. Compare to ML-approved loans. If agent decisions start defaulting at a higher rate, something's wrong with the agent's reasoning, regardless of how confident it claims to be. The system needs to alert you before the problem gets worse.

I also didn't initially plan for agent failures. What happens when Claude's API goes down? If agents handle twenty percent of your decisions and the service fails, you can't just stop processing applications. So I added graceful degradation. If the agent times out or errors, fall back to the ML model's decision, log it as a fallback event, and move on. The system has to keep working even when individual components fail.

But the biggest mistake was assuming guardrails were enough. Guardrails prevent bad decisions, but they don't ensure good decisions. It's not enough to check that the agent followed the rules, didn't violate fairness constraints, and provided reasoning. You have to measure whether agent-approved loans actually perform well, whether agent denials were accurate, whether conditional approvals get accepted, whether the explanations help customers understand what happened.

We have to treat agents like models, validating them against ground truth. Run A/B tests comparing agent decisions to human underwriter decisions. Track default rates by decision maker. Monitor customer satisfaction. Measure operational efficiency. If agents don't beat the baselines, they don't belong in production, no matter how sophisticated the guardrails are.

The guardrails I built for CreditIQ apply to any high-stakes AI system. Start by asking: what's the worst thing the AI could do? If the answer involves serious harm to people or the organization, you need constraints.

The three-layer model applies everywhere. Make the AI advisory rather than autonomous, with clear escalation thresholds for when humans should review. Constrain the reasoning space with hard rules the AI cannot override. Log everything comprehensively so you can audit any decision. Then test not just individual decisions but systematic patterns. Does the AI harm any group disproportionately? Does it perform consistently over time? Can you explain what it did and why?

Most importantly, plan for failure. AI will fail. Your job is to fail safely. What happens when the AI is unavailable? How do you disable it quickly if something goes wrong? How do you detect failures early? Who gets alerted and what's the process? These aren't edge cases to figure out later. They're fundamental architecture decisions you make from day one.

Here's what building CreditIQ taught me: good AI isn't just about building powerful models. It's about building trustworthy systems.

The hardest problems aren't technical. They're about trust, fairness, accountability, and safety. You can have the most sophisticated AI in the world, but if people don't trust it, if it systematically harms people, if you can't explain its decisions, or if it fails catastrophically when things go wrong, it doesn't belong in production.

Guardrails aren't an afterthought. They're not something you add once the AI works. They're fundamental to the architecture. Deploying AI without guardrails is like deploying software without error handling. It might work fine in demos, but it will fail spectacularly in production.

The uncomfortable part? Building proper guardrails means your AI system is slower, more expensive, and less autonomous than you initially envisioned. For CreditIQ, routing to agents adds five seconds of latency. Human review adds operational cost. Guardrails occasionally override correct agent decisions. Audit trails increase database storage. The system costs an extra four hundred dollars per month compared to pure ML.

But the alternative is worse. Deploy ungoverned AI in credit decisioning and you risk regulatory fines in the millions, discrimination lawsuits in the tens of millions, reputational damage that's priceless, and actual harm to real people, which is unacceptable. Those five seconds of latency and four hundred dollars a month aren't overhead. They're the cost of responsibility.

We're at an inflection point with AI. Systems like Claude and GPT-4 are capable enough to handle high-stakes decisions. But capability isn't the same as readiness.

The question isn't "Can AI do this?" anymore. It's "How do we deploy AI responsibly?" Because at the end of the day, AI in production isn't about what the system can do. It's about what you trust it to do.

The goal isn't to build AI that replaces humans. It's to build AI that augments humans. Where AI handles volume, humans handle stakes, and systems provide safety. That's not a limitation, that's the architecture. And getting that architecture right matters more than having the most sophisticated model.

I came into this project thinking I could solve hallucination with better prompts. I learned instead that it requires better architecture. Separating factual queries from semantic reasoning. Constraining what the AI can invent. Validating outputs before you trust them. These aren't prompt engineering tricks. They're system design principles.

The intelligence of the AI agent matters, but it's the orchestration that makes it trustworthy. Knowing when to use it, when to bypass it, when to validate its output, that control remains firmly on our side of the keyboard. And honestly, that's worth more than all the conversational polish in the world.

Full project [here](https://pmcavallo.github.io/CreditIQ/)

---

# The Neural Network Challenge: Does Deep Learning Win? (10/05/2025)

After seeing LightGBM beat traditional models (see previous blog), I had to ask: **What about neural networks?**

Tabular data is gradient boosting's home turf. Neural networks dominate images, text, and speech. But Google Research published TabNet in 2019, an attention-based architecture specifically designed for tables. If any neural network could compete, it would be TabNet.

**The Setup**

I trained TabNet with the same features as LightGBM:
- **Architecture**: 5 attention steps, 64-dimensional embeddings
- **Training**: 200 epochs with early stopping
- **Dataset**: Same 10K applications, same train/test split

**The Results**

**TabNet**: 0.7795 AUC (138 seconds)  
**LightGBM**: 0.8032 AUC (0.27 seconds)

LightGBM won by **2.4 percentage points** while training **500x faster**.

**Why This Matters**

This isn't a failure, it's validation. The research literature says "gradient boosting dominates tabular data," but I needed to test it myself. Here's what I learned:

**1. Neural networks need more data**  
TabNet might win on millions of samples. On 10K samples, it underfit while LightGBM captured complex patterns efficiently.

**2. Attention isn't always better than trees**  
TabNet's attention mechanism is elegant - it selects features sequentially like a human analyst. But gradient boosting's greedy splits are more effective for structured data.

**3. Speed matters in production**  
138 seconds vs 0.27 seconds isn't just about convenience. In production, faster training means:
- Quicker experiments during development
- More frequent model retraining
- Lower compute costs at scale

**4. Different models see different patterns**  
TabNet's attention weights told me something interesting: it ranked `num_delinquencies_2yrs` as the #1 feature (0.128 importance), while LightGBM ranked `credit_score` first. Same data, different architecture, different learned patterns.

This is why I tested multiple approaches. Not because I expected TabNet to win, but because **assuming without testing is how you miss opportunities**.

**The Architecture Decision Framework**

So when should you use each approach?

**Use Logistic Regression when:**
- You need maximum interpretability (regulators, stakeholders)
- You have strong feature engineering (manual interactions work)
- Speed is critical (sub-second scoring)

**Use Gradient Boosting when:**
- You need maximum performance on tabular data
- You have mixed data types (categorical + numeric)
- You want automatic feature interaction discovery

**Use Neural Networks when:**
- You have massive datasets (millions+ samples)
- You need multi-task learning (predict multiple outcomes)
- You're doing transfer learning (pre-trained embeddings)

For CreditIQ with 10K synthetic samples? **Gradient boosting wins.** 

For a production lending platform with 10M real applications? I'd test neural networks again, the answer might change.

That's the point. I don't pick architectures based on what's trendy or what I learned in 2020. I test, measure, and choose based on evidence.

**Next: Adding AI Agents to Handle Edge Cases**

Traditional models excel at standard cases. LightGBM finds non-linear patterns. But what about true edge cases that need reasoning?

That's where AI agents come in...

---

# Traditional vs Modern: Which ML Approach Wins? (10/05/2025)

I built three models to answer a fundamental question in ML: 
**Does traditional statistical feature selection still matter?**

In the traditional, "right way", we need to 
check VIF scores, run stepwise selection, justify every feature 
to stakeholders who ask "why is this variable in the model?" It's rigorous. 
It's defensible. It's also... slow.

Modern ML takes a different approach: throw in all the features, let 
regularization handle redundancy, trust the algorithm to find patterns. 
To traditional quants, this sounds reckless. To ML engineers, feature 
selection sounds like unnecessary busywork.

**The Contestants**

**Model 1: Logistic Regression (Full Arsenal)**
- All 45 features, including correlated pairs
- Test AUC: 0.7980
- Training time: 2.9 seconds
- *Problem: Multicollinearity, redundant features*

**Model 2: Logistic Regression (Refined)**
- Statistical feature selection: removed correlated features, kept only significant (p<0.05)
- 15 features (67% reduction)
- Test AUC: 0.7958 (-0.2%)
- Training time: 2.1 seconds
- *Result: Same performance, simpler model*

**Model 3: LightGBM (Modern ML)**
- All 45 features, let the algorithm decide
- Test AUC: 0.8032 (+5.2% vs logistic)
- Training time: ~3 minutes (includes hyperparameter tuning)
- *Winner: Best performance by handling non-linear interactions*

**What This Means**

**For interpretable models:** Feature selection helps. The refined logistic 
regression has identical performance with 67% fewer features, which is easier to 
explain to stakeholders and regulators, faster to score, simpler to maintain.

**For production ML:** Let the algorithm handle complexity. LightGBM 
automatically ignores weak features and finds non-linear patterns that 
linear models miss.

**The hybrid approach:** Use both. Feature engineering (like `income_x_score`, 
which ranked #2) combined with modern ML gives you the best of both worlds.

**SHAP Explainability: The Best of Both Worlds**

Even though LightGBM is more complex, SHAP values provide feature 
attributions for every prediction:

![SHAP](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/shap_waterfall_approved.png?raw=true)

For this approved application (77.9% probability):
- ✅ No delinquencies: +0.19 (strong positive)
- ✅ High income × credit score: +0.16
- ✅ Good credit score (691): +0.13
- ⚠️ High DTI × loan amount: -0.21 (risk factor)

This is **exactly what stakeholders, and especially regulators, want**, modern performance with 
traditional explainability.

For years, I've heard the debate: interpretability vs performance. Traditional 
models vs black-box ML. Regulatory compliance vs innovation.

**It's a false choice.**

You can have gradient boosting performance (0.803 AUC) with per-prediction 
explainability (SHAP). You can engineer features using domain knowledge 
(`income_x_score` ranking #2) while letting algorithms find non-linear 
interactions. You can satisfy strict regulations (e.g. SR 11-7) while deploying modern ML.

That's what I'm building my next project, CreditIQ, this way. Not because it's what I learned and it's "traditionally" used, and not because it's what's trendy in ML. Because it's what actually 
works when you need to ship models that are both accurate and defensible.

---

# When RAG Tools Hallucinate: Building Trust Through Custom Architecture (10/01/2025)

I needed something simple: a way for visitors, and myself, to ask questions about my 27 data science portfolio projects. Natural language queries, accurate answers, maybe a few citations. Easy, right? I reached for Flowise, a popular no-code RAG platform that promised exactly this, conversational AI without the complexity of building from scratch. Within an hour, I had something running. Within two hours, I realized it was confidently lying to me. "Tell me about Project Alpha," it would say, describing in detail a machine learning project I'd never built. Ask about Azure, and it would invent three projects using Microsoft's cloud platform—despite my portfolio containing exactly zero Azure work. The system would generate plausible-sounding descriptions, cite realistic-seeming architectures, and do it all with the unwavering confidence of an LLM that has no idea it's making things up. I could have kept tweaking prompts. Instead, I rebuilt the entire system from scratch.

**Architectural Grounding Over Prompt Engineering**

The core insight: hallucination prevention isn't a prompt engineering problem. It's an architecture problem. Instead of asking the LLM "what are all the project titles?", I store titles as structured metadata during document ingestion. When someone asks for a list, the system queries a SQLite database directly, no LLM involved. It's impossible to hallucinate when you're reading from a database.

For semantic queries, I implemented multiple layers of grounding:

**Layer 1: Metadata Extraction**

Each markdown file in my portfolio has YAML frontmatter with the project title, date, and tags. The ingestion pipeline extracts this before chunking and attaches it to every chunk from that document. The LLM never has to guess what project it's reading about—it's in the metadata.

**Layer 2: Strict System Prompt**

The prompt is blunt: "Answer ONLY from provided context. If context doesn't contain the answer, say 'I don't have sufficient information.' DO NOT add information from your training data. DO NOT make up project names."
Temperature is set to 0. No creativity allowed.

**Layer 3: Response Validation**

Before returning an answer, the system checks: Does it contain citations? Does it reference actual source files? If not, it shows raw excerpts instead of the LLM's response.

**Layer 4: Honest Uncertainty**

When someone asks "What's your most complex project?", the system retrieves relevant documents but refuses to answer. Why? Because "complex" is subjective. Without objective complexity metrics in the retrieved chunks, making a claim would be speculation. The system says "I don't have sufficient information to judge complexity."

Better to be honest than helpful.

**The Cost of Trust**

Building this custom system took about two days. The ongoing cost is $0.02 per month for 100 queries. Flowise would have been "free" (not counting the Pinecone vector store). But free hallucinations aren't worth it when credibility is on the line. The technical implementation uses LangChain for orchestration, Chroma for local vector storage, OpenAI's text-embedding-3-small for embeddings ($0.0006 total), and GPT-4o-mini for generation. The entire system runs on Hugging Face Spaces' free tier, rebuilding the vector database from GitHub on first startup if needed.

**What I Learned**

Pre-built tools optimize for the wrong metrics. Flowise optimizes for conversational engagement. I needed accuracy. Those aren't the same thing. "I don't know" is a feature, not a bug. Honest uncertainty builds more trust than confident fabrication. The best RAG systems admit their limitations. Metadata is architectural truth. By separating factual queries (metadata lookups) from semantic queries (LLM-based), you eliminate entire classes of hallucination. The LLM never gets a chance to invent project titles because it never sees those queries. Context engineering matters more than prompt engineering. Shaping the environment around the model—templates, schemas, retrieval strategies, validation layers—matters more than clever wording inside the prompt. In production, costs mount when outputs need manual fixes, efficiency drops when results drift, and trust erodes when decisions can't be validated.

I came into this thinking I could solve hallucination with better prompts. I learned instead that it requires better architecture. The intelligence of the LLM is real, but uneven. The orchestration—knowing when to use it, when to bypass it, when to validate its output—remains firmly on our side of the keyboard.

The project lives [here](https://huggingface.co/spaces/pmcavallo/portfolio-rag-agent). Ask it about Azure. It will tell you the truth: I don't use it.

That honesty is worth more than all the conversational polish in the world.

---

# From Zero-Shot to Production-Ready: Lessons From a Prompt Engineering Lab (09/28/2025)

This project set out to answer a simple question: how do we turn a free-form prompt into a governed system that consistently produces reliable and auditable outputs? What began as a zero-shot classification task, asking a model to categorize credit policy notes, quickly evolved into a deeper exploration of prompt engineering maturity. By layering engineered prompts, schema enforcement, validation guardrails, tool integration, and retrieval augmentation (RAG), the lab demonstrates the shift from clever phrasing to systems design.

Across the journey, I built confusion matrices to spot systematic errors, validated JSON schemas to enforce structure, added guardrails to catch out-of-range values, integrated deterministic loan calculators for hybrid reasoning, and finally grounded answers in external documents with retrieval scores. Each step reduced ambiguity, improved accuracy, and made the system more trustworthy.

The real lesson was not just accuracy gains (baseline 0.75 to engineered 0.95). It was the discovery that context engineering—shaping the environment around the model with templates, schemas, and retrieval—matters more than tweaking the words inside the prompt. In production, costs mount when outputs need manual fixes, efficiency drops when results drift, and trust erodes when decisions cannot be validated. Prompt engineering, done as systems design, addresses all three.

I came away convinced that prompt engineering is no longer about tricking models with clever wording. It is about designing interfaces between human intent, model reasoning, and business logic. These interfaces save cost, reduce error, speed workflows, and unlock safe deployment.


🔗 [View Full Project](https://pmcavallo.github.io/prompt-engineering/)

---

# When AI Writes Code: Lessons From a Shootout (09/26/2025)

This project began with a simple question: if I handed the same machine learning task to three different AI coding assistants, how differently would they approach it? I set up what I called the RiskBench AI Coding Shootout, bringing Claude Code, GitHub Copilot, and Cursor into the same arena. The goal wasn’t to crown a champion but to learn how their choices shaped data quality, model performance, and the reliability of the final system.

I broke the work into four sprints, moving from scaffolding and data generation to modeling, then to serving predictions and explanations through an API. At first the tasks seemed straightforward, but almost immediately the experiment revealed something deeper: the subtle but decisive impact of dataset quality. Claude’s synthetic data turned out richer, with stronger signals, while Copilot and Cursor produced weaker versions that led to models that looked fine on the surface but collapsed under test. Once I leveled the playing field by having everyone use Claude’s dataset, Copilot’s performance jumped back up, proving that it wasn’t the modeling code that failed but the foundation it was built on.

By the time I reached the serving stage, Claude managed to deliver a functioning API with SHAP explanations, while Copilot stumbled with file paths and fragile module setups, and Cursor hit usage limits. What struck me most wasn’t the technical glitches themselves but how often resilience came down to hidden layers: validating assumptions, handling edge cases, and stitching components together into something that wouldn’t break under pressure. That is where human oversight still matters most.

I came into this wanting to see whether an AI agent could stand in for a human engineer. What I found instead is that these tools are accelerators, not replacements. The intelligence in their code is real, but uneven; the orchestration and judgment remain on our side of the keyboard. The project reminded me that garbage in still means garbage out, that explainability is the bridge to trust, and that the real work is less about producing a working snippet than about building systems that hold together in the real world.

🔗 [View Full Project](https://pmcavallo.github.io/code-agent/)

--------------------

# AI-in-the-Cloud Knowledge Shootout: Perplexity vs NotebookLM (09/14/2025)

This project set out to answer a simple but important question: can AI tools themselves act as orchestrators of cloud knowledge? After building the [Cross-Cloud Shootout](https://pmcavallo.github.io/cross-cloud/) to compare AWS and GCP directly, I wanted to pit two AI copilots — **Perplexity Pro** and **Google NotebookLM** — against each other. The goal was to see how each tool handled real cloud prompts across cost optimization, architecture design, and governance, and whether their answers could be relied on for strategic or operational decision-making.

From the beginning, the experiment was uneven by design. NotebookLM requires a curated corpus, so for each prompt I loaded it with AWS and GCP documentation, pricing pages, and my own Cross-Cloud blog. This meant it behaved like a research assistant: highly aligned to those inputs, precise in tone, but limited to what I gave it. Perplexity, by contrast, could not be given a preloaded corpus. It searched the open web in real time, citing blogs, docs, and technical guides it found along the way. It acted more like a field guide: fast, broad, and pragmatic, but not tied to the framing I wanted. The difference between the two approaches became the story of the shootout.

The test design consisted of six prompts, two in each category. For cost optimization, the tools were asked first to design a budget-constrained ML pipeline (COST_01), and later to compare the economics of training versus inference for a mid-size team (COST_02). For architecture fit, they first explored how to deliver telecom-grade low-latency inference (ARCH_01), then tackled the harder problem of a hybrid control plane spanning AWS and GCP (ARCH_02). Finally, for governance, they compared compliance frameworks and enforcement mechanisms (GOV_01), and closed with the design of a model governance baseline for credit risk PD estimation models (GOV_02).

Across these six steps, the contrast between the tools was striking. NotebookLM consistently produced long, detailed answers that read like policy documents or whitepapers. When asked about cost optimization, it leaned heavily on governance levers, structured playbooks, and predictable billing frameworks, often echoing ideas from my Cross-Cloud shootout. Its answers were grounded and complete, rarely skipping a piece of the prompt. Perplexity, on the other hand, delivered tables, checklists, and concrete numbers. It listed hourly rates for SageMaker and Vertex AI instances, mapped services to categories, and described system architectures with edge placement and failover scenarios. Where NotebookLM was deep, Perplexity was clear; where NotebookLM was structured, Perplexity was practical.

The governance prompts highlighted this divergence most clearly. NotebookLM built out a governance baseline that would not be out of place in a regulatory framework: documentation requirements, approval workflows, lineage and reproducibility, all mapped carefully to AWS and GCP services. Perplexity gave a checklist: start with model cards, enable audit logs, use metadata tracking, monitor for bias and drift. Both were correct, but each spoke to a different audience, the compliance officer on one hand, the practitioner on the other.

By the end of the shootout, the pattern was clear. NotebookLM is at its best when you want controlled, source-aligned synthesis. It “thinks like a policy author.” Perplexity is strongest when you need concise, actionable answers quickly. It “thinks like a practitioner writing a runbook.” Neither tool is a winner on its own; the real value comes from combining them. NotebookLM sets the strategy, while Perplexity supplies the tactics.

The lesson of this project is that LLMs are not yet full orchestrators of cloud strategy. But in tandem, they can support both ends of the spectrum: strategic frameworks and day-to-day execution. In that sense, this AI shootout reflects the broader reality of cloud computing itself — no single provider, tool, or perspective is enough. The power lies in integration, in building systems where strengths complement each other.

🔗 [View Full Project](https://pmcavallo.github.io/ai-in-the-cloud/)

---

# LLM Quiz App: Lessons from Building Across ChatGPT, Gemini, and Claude (09/14/2025)

This project started with a simple idea: I wanted a quiz tracker app. ChatGPT was struggling to incorporate a quiz with a memory component in its daily tasks, so the workaround was to build a small local “agent.” The plan was straightforward: ask questions, grade answers, and persist results in a local database using RAG to recall past attempts. It sounded ambitious, but at its core, it was supposed to be just that, an app that tracks quizzes.  

ChatGPT initially pitched a small web app with an API that I could run locally with a browser UI. Over time, it kept suggesting new features it could not actually implement. After a few days of false starts, we finally had something: an MVP. It worked well enough, with logging, summaries, a session-aware Bottom-3, randomized and non-repeating question selection, and a large question bank. It was not the adaptive LLM-powered engine that was first envisioned, but it did the job.  

Frustrated but still curious, I tried Gemini next. It promised a real-time coding experience in Canvas, with prompts on one side and a live editor on the other. The interface looked sleek, with buttons, a tracker title, and a “fix error” button that popped up after every crash. Gemini would identify bugs, explain them, and claim to fix them, only for the same black screen and “fix error” button to reappear. After two days, I still did not have an app that could run outside of Gemini.  

Gemini also assumed I wanted a multi-user app, so it pushed Firebase for memory and Netlify for hosting. That was not my goal, I only needed local persistence so the tracker would not reset each session. The Firebase and Netlify approach quickly became fragile without a proper build process, running into race conditions and security issues that a production pipeline would normally solve.  

At this point, I pivoted to Claude. Right away, it suggested starting with a self-contained app, a simple HTML file with persistence across browser sessions. That worked immediately on Netlify, no Firebase required. I was impressed with Claude’s stability and its ability to preserve file integrity while implementing fixes. Unlike ChatGPT or Gemini, it did not regress or produce broken versions with each change. But then I hit its hard usage limits. Even on Pro, I was locked out for five hours after hitting the quota. That made iterative work, the style I rely on, impossible. With Claude, you must be strategic, efficient, and very clear in prompts. There is no space for the back-and-forth experimentation I enjoy with ChatGPT.  

Armed with the lessons from Claude, I went back to Gemini and asked for a self-contained app with only session persistence, no API or cloud services. This time it delivered a clean, minimal app with a fun icon-based interface. I gave the same instructions to ChatGPT and it did produce a self-contained app with even more complex graphs than Gemini, albeit a less clean and visually appealing interface. I just had to do the same thing with Claude now. But I was still waiting for my Claude lockout to end.  

---

### Best Practices I Learned for Claude

**Maximize each message**  
- Provide full context and requirements upfront  
- Be comprehensive rather than incremental  
- Bundle related requests together  

**Use strategic prompting**  
- Specify format, length, and expectations  
- Give examples of what you want  
- Use structured prompts with sections  

**Plan sessions carefully**  
- Group related work  
- Prioritize complex tasks first  
- Offload simple tasks to other tools  

---

One of Claude’s limitations is file size. It struggles with large outputs and large inputs, and once its 200k-token context window fills, you need to start a new chat. Unlike ChatGPT, it cannot access artifacts from previous sessions. Every new chat means re-uploading context and files.  

Even with these issues, I liked Claude. It is fast, practical, and makes smart suggestions unprompted. But it cannot stand alone, you would need at least one other LLM alongside it. ChatGPT remains the most versatile for general use, while Gemini offers extra value through Google’s ecosystem and services.  

For this app project, each LLM had strengths and weaknesses. All of them produced usable quiz apps, though none fully achieved the original vision: a self-contained app that also tapped its LLM creator for infinite questions, explanations, and personalized study plans. That dream turned out to be too ambitious.  

Still, I ended up with something real: a working quiz tracker, lessons in cross-LLM development, and a deeper appreciation of how these tools differ not just in features but in workflow philosophy.  

---

# Cross-Cloud AutoML Shootout: Lessons from the Trenches. (09/10/2025)

With the **Cross-Cloud AutoML Shootout** project the idea was straightforward: pit AWS SageMaker Autopilot against Google Cloud’s AutoML, feed them the same dataset, and see who comes out on top in terms of accuracy, speed, and cost. What happened instead turned into a lesson about quotas, governance, and adaptability in cloud AI.

Just like in my credit risk modeling work, where the challenge often isn’t the math itself but the infrastructure and constraints, this shootout was as much about system design as it was about algorithms. AWS and GCP offered very different paths, and those differences reshaped the entire project.

But the project wasn’t just about comparing two black-box services. It was about uncovering how each cloud handles scale, transparency, and control:

- **AWS Autopilot: Fast Start, Pay-as-You-Go**  
  Autopilot trained 10 candidate models in 30 minutes, surfacing a solid performer with ~65% accuracy and ~0.78 ROC AUC. Cost: about $10. The tradeoff: limited visibility into feature importance—good performance, but little interpretability.

- **GCP Vertex AI AutoML: Quota Walls Everywhere**  
  On paper, Vertex AI AutoML should have been the competitor. In practice, hidden quotas derailed every attempt. Even after raising CPU quotas twice, the pipeline kept failing with opaque errors. Without a paid support plan, there was no path forward.

- **Pivot to BigQuery ML: Control Through SQL**  
  Instead of abandoning the project, I pivoted. With BigQuery ML, I wrote SQL directly to train models, engineer features, and evaluate results. The boosted tree model came in slightly weaker (~56% accuracy, ~0.74 ROC AUC), but I gained full transparency, feature importance, and—critically—predictable cost. Under 1 TiB/month, it was effectively free.

## Lessons Learned

- **Cloud Governance Matters**: GCP AutoML’s hidden quotas were a reminder that cloud experiments aren’t just technical, they’re operational.  
- **Transparency vs. Accuracy**: AWS won on performance, but BigQuery ML gave us the interpretability that regulated industries demand.  
- **Cost Awareness**: The shootout underscored how pricing models (per-second vs. node-hour vs. query bytes) drive design decisions.  
- **Adaptability as a Skill**: The pivot itself was a success. Knowing when to change course is part of building resilient AI systems.

The name “Shootout” now reflects more than just a head-to-head test. It captures the reality of working across clouds: the contest isn’t just between models, but between philosophies of control, cost, and transparency.  

Why does this matter? Because in AI, as in finance, **constraints shape outcomes**. 

🔗 [View Full Project](https://pmcavallo.github.io/cross-cloud-ml/)

---

# SignalGraph: Telecom Data Pipelines Reimagined (08/30/2025)

When I launched **SignalGraph**, the idea was simple: treat telecom network data with the same care and structure that financial institutions give to credit risk data. In practice, that meant building a pipeline that could transform raw, messy network logs into a structured system where anomalies, trends, and performance could be analyzed at scale.

Just like in my banking work, where I’ve spent years reconciling multiple servicer feeds and correcting default timing errors, SignalGraph began with the fundamentals: data integrity. The project ingests raw Bronze-layer parquet files, standardizes them in Silver with anomaly flags, and prepares feature-rich Gold outputs. Each step ensures consistency, comparability, and readiness for modeling.

But the project wasn’t just about ETL. It was about bringing together a full ecosystem:

- **Anomaly Detection:** In finance, a default can be seen as an anomaly—a rare, high-impact event that needs to be captured correctly. In telecom, latency spikes and overloaded cells serve a similar role. SignalGraph flags anomalies (latency > 60ms, PRB > 85%) directly in the Silver layer so they are never lost in downstream aggregation.
- **Feature Engineering at Scale:** Gold-layer datasets include hourly features, giving analysts and models the inputs they need for prediction and trend analysis.
- **Modeling and Forecasting:** SignalGraph is designed to support both baseline predictive models (XGBoost, Random Forests) and forecasting tools like Prophet for time-series latency predictions.
- **Graph Analytics:** Using Neo4j, the project explores neighbor effects and centrality—critical for understanding how one weak node can ripple through a telecom network.
- **Governance and Transparency:** Just like my regulatory work at Comerica, every stage of SignalGraph is documented and auditable. The design emphasizes trust, reproducibility, and clarity.

The name *SignalGraph* reflects this dual ambition: it’s not only about signals in the network, but also about connecting the nodes, people, tools, models, and governance into a coherent graph.

Why does this matter? Because in telecom, as in finance, anomalies aren’t just noise. They’re signals—warnings of where risk is building or where performance is degrading. SignalGraph shows how an end-to-end AI/ML system can surface those signals, structure them for decision-making, and keep them flowing into live dashboards and predictive models.

**Explore the project:** [SignalGraph on GitHub](https://pmcavallo.github.io/signalgraph/)

---

# One Project. Three Platforms. (08/16/2025)

NetworkIQ started as a simple idea. I wanted one pipeline that I could move between platforms without rewriting the heart of the work. The goal was to ingest telecom-style telemetry, shape it into clean features, train a baseline model to detect latency anomalies, and give stakeholders a friendly view to explore results. Same project, three homes. Render for speed. AWS for scale and control. GCP for a pragmatic middle ground.

Working across platforms forced me to separate what is essential from what is incidental. The essential parts are the business logic and the modeling flow. The incidental parts are authentication, packaging, endpoints, and scheduling. When I kept that separation clear, the project moved with me.

## Render: first link to share

Render was the shortest path to something I could show. I connected a repository, set a few environment variables, and had a public URL I could send to a colleague the same day. That matters when momentum is everything. Streamlit felt natural here, and the platform handled build and deploy so I could focus on the story the data was telling.

The tradeoff is that Render is not where I want to crunch heavy Spark jobs or run long training runs. It shines when I need a clean demo, a quick what-if tool, or a lightweight API. For NetworkIQ, that meant pushing precomputed features and model artifacts to the app and keeping the heavy lifting elsewhere. For stakeholder conversations this was perfect. People could see the problem, adjust thresholds, and understand operational implications without waiting on infrastructure.

## AWS: scale and controls

When I needed stronger controls and a clearer path to enterprise practices, AWS was the natural step. S3 gave me a stable lake for Parquet data and versioned artifacts. Glue or EMR ran PySpark transforms when feature builds became heavier. SageMaker provided a place to train and track models with the permissioning and logging that risk and security teams expect.

The price of that power is setup time and cognitive load. Policies, roles, networking, and service limits have to be right, and each decision has ripple effects. Once it is in place it feels robust. For NetworkIQ, AWS made sense when I asked how to hand this to another team, how to secure it properly, and how to monitor it over time. This is the platform I would pick for a regulated environment or a high-traffic deployment that cannot fail quietly.

## GCP: pragmatic analytics

On GCP I found a productive middle. Cloud Storage handled the data lake side without fuss. Dataproc and notebooks were straightforward for Spark work. Vertex AI was an opinionated yet flexible path to train and serve. Cloud Run made it simple to put a small service on the internet without managing servers.

For NetworkIQ, the appeal was speed with enough structure to feel professional. I would choose it for teams that want managed services with a simpler day-one experience, especially when BigQuery is part of the analytics stack. It was also an easy place to stitch together a scheduled job and a lightweight service without touching too many knobs.

## What stayed constant

Across all three platforms the core of NetworkIQ did not change. The data model, the way I engineered features, and the way the model consumed them stayed the same. I kept artifacts and data in open formats and wrote the pipeline so that storage, secrets, and endpoints were swappable. That discipline paid off. Moving the project was about replacing adapters and configuration, not rewriting logic.

## What changed

What changed was everything around the edges. Authentication flows were different. Deployment packaging had to follow each platform’s rules. Schedulers had different names and limits. Monitoring and logs lived in different places. None of this changed the value of the project, but it did change how quickly I could iterate and how easily I could meet enterprise expectations.

## When I would choose each

I choose Render when I want a live demo today, when the purpose is conversation, or when a simple app needs to be shareable without ceremony.  
I choose AWS when I need strong governance, integration with enterprise tooling, and room for production growth.  
I choose GCP when I want managed services that get me moving fast, especially for analytics teams that live in notebooks and SQL and want a clean path to serving.

## What I learned about orchestration

The most important lesson from NetworkIQ is that portability is a mindset. If I treat the cloud as the place I run the project rather than the project itself, I retain the freedom to move. That freedom is strategic. It lets me optimize for speed when I am validating an idea and for robustness when I am scaling it. It also keeps the conversation with stakeholders focused on outcomes, not tools.


If you want to see the project narrative and visuals, the project page is here: 🔗 [View Full Project](https://pmcavallo.github.io/network-iq/) 
