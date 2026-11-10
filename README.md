# ML System Design

Notes for ML system design interviews, one topic per document. Each one works through a real problem the way you would in an interview: clarify the requirements, frame the ML problem, design the data and features, pick a model, evaluate it, and serve it at scale.

> **Note:** These documents, including their diagrams, were written by an AI coding agent ([Claude Code](https://claude.com/claude-code)), directed and reviewed by the repository owner as interview preparation. They're study notes, not authoritative references. Verify important details against the linked papers, and please open an issue if you spot an error.

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

Topics are grouped by problem type, one folder per category.

### Ranking — [`ranking/`](ranking/)

| Topic | Key concepts |
|-------|--------------|
| [Feed Ranking for a Large-Scale Social Network](ranking/feed-ranking.md) | Multi-stage funnel, two-tower retrieval, multi-task ranking (MMoE), fan-out strategies, position bias, network effects in A/B tests |
| [Improving Feed Ranking to Increase Time Spent](ranking/feed-time-spent.md) | Time-spent driver tree, diagnosis, dwell/watch-time prediction, duration bias, session-continuation scoring, long-term value / RL, cannibalization, well-being guardrails |
| [Ad Click Prediction](ranking/ad-click-prediction.md) | Calibration and auctions (eCPM, GSP), hashing trick, LR → DCN evolution, online learning, negative downsampling correction, delayed conversions |

### Recommendations — [`recommendations/`](recommendations/)

| Topic | Key concepts |
|-------|--------------|
| [Video Recommendation System](recommendations/video-recommendation.md) | Multi-source retrieval, two-tower + log-Q correction, co-watch, content understanding for cold start, position-bias tower, watch-time objectives |

### Trust and Safety — [`trust-and-safety/`](trust-and-safety/)

| Topic | Key concepts |
|-------|--------------|
| [Harmful Content Detection](trust-and-safety/harmful-content-detection.md) | Multimodal early fusion, multi-label heads, per-severity thresholds, layered detection, review-queue prioritization, prevalence metric, adversarial adaptation |
| [Payment Fraud Detection](trust-and-safety/payment-fraud-detection.md) | Cost-sensitive decisions (approve / challenge / decline), velocity and graph features, GBDT + rules, delayed chargeback labels, selection bias, dollar-weighted recall |
| [Multimodal Perception and Authentication](trust-and-safety/multimodal-authentication.md) | 1:1 verification with face / voice / behavioural embeddings, liveness and presentation-attack detection, score-level fusion with quality and context, risk-based thresholds, TAR@FAR and APCER/BPCER, demographic fairness, on-device templates |

### Agentic Systems — [`agentic/`](agentic/)

| Topic | Key concepts |
|-------|--------------|
| [AI Customer Support Agent](agentic/customer-support-agent.md) | Workflow vs agent autonomy, ReAct tool loop, customer-scoped and policy-checked tools, guardrails and prompt injection, RAG, simulated-user evals with pass^k, LLM-as-judge, cost/latency routing |

### Simulation — [`simulation/`](simulation/)

| Topic | Key concepts |
|-------|--------------|
| [Realistic Simulation for Autonomous Driving](simulation/autonomous-driving-simulation.md) | Log replay vs closed-loop, multi-agent behaviour models (sim agents), neural sensor simulation (3DGS / NeRF), adversarial scenario generation, perception consistency, sim-to-real correlation, importance sampling for rare events |
| [Realistic Simulation for LLM Agents](simulation/llm-agent-simulation.md) | Environments as (state, task, actions, verifier), stateful mocks vs LLM-simulated tools, user simulators and their failure modes, task generation with solvability checks, reward hacking defences, curriculum, sim-to-real correlation, pass^k |

### Computer Vision — [`computer-vision/`](computer-vision/)

| Topic | Key concepts |
|-------|--------------|
| [Visual Search System](computer-vision/visual-search.md) | Embedding + ANN instead of classification, street-to-shop pairs, hard negative mining, triplet / InfoNCE losses, IVF-PQ and HNSW, exact-match re-ranking, re-indexing when the embedding model changes |
| [Face and License Plate Blurring for Street-Level Imagery](computer-vision/street-view-blurring.md) | Recall-first object detection, low operating thresholds, panorama re-projection and tiling, FPN + focal loss, test-time augmentation and ensembles, recall at low IoU and coverage, audit miss rate, batch GPU pipeline |

## Contributing

Spotted a mistake or have a better approach? Open an issue or a PR.
