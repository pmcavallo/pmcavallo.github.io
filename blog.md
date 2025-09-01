---
layout: default
title: Blog          
permalink: /blog/
---

Welcome to the blog. Here I share short, practical notes from building my portfolio projects. Expect lessons learned, trade offs I faced, and the reasoning behind architecture and tool choices. If you want the code and demos, visit the [Projects](/projects/) page. Start with the latest post below.

---

# SignalGraph: Telecom Data Pipelines Reimagined (08/30/2025)
---

When I launched **SignalGraph**, the idea was simple: treat telecom network data with the same care and structure that financial institutions give to credit risk data. In practice, that meant building a pipeline that could transform raw, messy network logs into a structured system where anomalies, trends, and performance could be analyzed at scale.

Just like in my banking work, where I’ve spent years reconciling multiple servicer feeds and correcting default timing errors, SignalGraph began with the fundamentals: data integrity. The project ingests raw Bronze-layer parquet files, standardizes them in Silver with anomaly flags, and prepares feature-rich Gold outputs. Each step ensures consistency, comparability, and readiness for modeling.

But the project wasn’t just about ETL. It was about bringing together a full ecosystem:

- **Anomaly Detection:** In finance, a default can be seen as an anomaly—a rare, high-impact event that needs to be captured correctly. In telecom, latency spikes and overloaded cells serve a similar role. SignalGraph flags anomalies (latency > 60ms, PRB > 85%) directly in the Silver layer so they are never lost in downstream aggregation.
- **Feature Engineering at Scale:** Gold-layer datasets include hourly features, giving analysts and models the inputs they need for prediction and trend analysis.
- **Modeling and Forecasting:** SignalGraph is designed to support both baseline predictive models (XGBoost, Random Forests) and forecasting tools like Prophet for time-series latency predictions.
- **Graph Analytics:** Using Neo4j, the project explores neighbor effects and centrality—critical for understanding how one weak node can ripple through a telecom network.
- **Governance and Transparency:** Just like my regulatory work at Comerica, every stage of SignalGraph is documented and auditable. The design emphasizes trust, reproducibility, and clarity.

The name *SignalGraph* reflects this dual ambition: it’s not only about signals in the network, but also about connecting the nodes—people, tools, models, and governance—into a coherent graph.

Why does this matter? Because in telecom, as in finance, anomalies aren’t just noise. They’re signals—warnings of where risk is building or where performance is degrading. SignalGraph shows how an end-to-end AI/ML system can surface those signals, structure them for decision-making, and keep them flowing into live dashboards and predictive models.

---

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
