# ML System Design

Notes for ML system design interviews, one topic per document. Each one works through a real problem the way you would in an interview: clarify the requirements, frame the ML problem, design the data and features, pick a model, evaluate it, and serve it at scale.

## How each topic is structured

Every topic follows the same outline, so you can practise a consistent interview flow:

1. **Clarifying requirements**: scope, scale, latency, and constraints
2. **Framing as an ML problem**: objective, inputs and outputs, ML category
3. **Data preparation**: sources, labeling, and feature engineering
4. **Model development**: baselines, architecture choices, and training
5. **Evaluation**: offline metrics, online metrics, and A/B testing
6. **Serving and system architecture**: diagrams, the online and offline pipelines, and scaling
7. **Monitoring and iteration**: drift, retraining, and failure modes
8. **Interview talking points**: trade-offs, follow-up questions, and pitfalls

Diagrams are written in [Mermaid](https://mermaid.js.org/), which GitHub renders inline.

## Topics

| # | Topic | Key concepts |
|---|-------|--------------|
| 1 | [Feed Ranking for a Large-Scale Social Network](topics/feed-ranking.md) | Multi-stage funnel, two-tower retrieval, multi-task ranking (MMoE), fan-out strategies, position bias, network effects in A/B tests |
| 2 | [Improving Feed Ranking to Increase Time Spent](topics/feed-time-spent.md) | Time-spent driver tree, diagnosis, dwell/watch-time prediction, duration bias, session-continuation scoring, long-term value / RL, cannibalization, well-being guardrails |
| 3 | [Video Recommendation System](topics/video-recommendation.md) | Multi-source retrieval, two-tower + log-Q correction, co-watch, content understanding for cold start, position-bias tower, watch-time objectives |

## Contributing

Spotted a mistake or have a better approach? Open an issue or a PR.
