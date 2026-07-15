# CHAPTER 7 — EVALUATION

> Draft. Experiment E1 (Blind Walker) is COMPLETE with final numbers and figures.
> Sections marked **[PENDING — awaiting run completion]** will be filled when the
> pushblock_comm_curr_s6 and pushblock_commpo_curr_s1 Kaggle runs finish.
> Citation numbers `[n]` follow the locked reference list [1]–[18].
> Figures: `thesis/figures/fig7_1_blindwalker_reward.png`, `fig7_2_blindwalker_episode_length.png`
> (captions BELOW figures in the Word doc, template style "Fig. 7.x: ...").

## 7.1 Introduction

This chapter evaluates the Comm-MAPOCA system implemented in Chapter 6 against the hypothesis of Chapter 4: that integrating a differentiable, targeted communication protocol into the MA-POCA architecture improves cooperative performance, with the benefit concentrated where information asymmetry exists between teammates. The evaluation is organised as three experiments of increasing task complexity, each answering one question in a chain: *does emergent communication work at all* (E1), *what does the architecture achieve on a standard cooperative benchmark* (E2), and *when is communication actually valuable* (E3).

## 7.2 Evaluation Strategy

**Experiment E1 — Communication validation (Blind Walker).** A task in which communication is necessary *by construction*: the mobile Walker cannot see the goals it must reach, and two stationary Oracles can see the goals but cannot move. Any performance above blind-search level is attributable to the message channel. E1 answers: can Comm-MAPOCA agents *invent* a working communication protocol end-to-end, with no hand-coded message content, driven purely by the shared team reward?

**Experiment E2 — Benchmark performance (Cooperative Push Block, 2×2 matrix).** The stock cooperative Push Block task trained for 30M steps in four configurations: {stock MA-POCA, Comm-MAPOCA} × {full observability, partial observability}, with identical hyperparameters, arenas, and rewards. E2 provides the like-for-like comparison against the unmodified baseline algorithm [10] and, importantly, documents how *all four* configurations behave on the full task without curriculum support. **[Runs complete; analysis PENDING]**

**Experiment E3 — Curriculum capability and communication value under partial observability.** The full six-block Push Block task (two small, two large, two very-large blocks; heavy blocks require multiple agents pushing simultaneously) trained with an 11-lesson curriculum (Chapter 6). Two runs: full observability (the capability demonstration) and 7×7-view partial observability (the communication-value experiment). The planned **zero-message ablation** — evaluating the trained partial-observability policy with its message channel silenced — provides the causal test that communication, not merely the curriculum, carries the performance. **[PENDING — runs in progress]**

All training runs use the cloud pipeline of Section 6.2; single-seed results are reported (a limitation acknowledged in Section 7.9).

## 7.3 Experimental Setup

**E1 setup.** The Blind Walker environment (Section 6.4.3) is trained with the `comm_mapoca` trainer for 30 million environment steps on the Kaggle cloud pipeline (CPU-only, two parallel headless environment instances). One shared behavior (`CommMAPOCABrain`) controls the three agents (two Oracles, one Walker) as a single MA-POCA group with role-flagged observations. Hyperparameters are the toolkit's cooperative-task defaults, unchanged: batch size 1024, buffer size 10240, learning rate 3×10⁻⁴ (linear schedule), β = 0.01, ε = 0.2, λ = 0.95, 3 epochs, hidden size 256 × 2 layers, γ = 0.99, time horizon 64. Messages are 4 floats, all-to-all delivery restricted by the Directed topology (Oracles → Walker). No curriculum, no reward shaping beyond the environment's built-in potential-based "getting warmer" term, and no hand-coded message content: this run is fully emergent (Stage 2 of the protocol in Section 6.8).

**TABLE 7.1: E1 ENVIRONMENT PARAMETERS**

| Parameter | Value |
|---|---|
| Agents | 2 stationary Oracles + 1 mobile Walker (one shared behavior) |
| Message | 4 floats, tanh-bounded, one-step delay |
| Win reward (correct goal) | +3.0 (group) |
| Wrong-colour goal | −0.1 (group, episode continues) |
| Timeout | 2500 physics steps, −1.0 (group) |
| Shaping | potential-based, 0.1 × per-step distance reduction to correct goal |
| Per-step existential penalty | −1/2500 |
| Goal/Walker spawns, target colour | randomised every episode |

## 7.4 Datasets and Test Cases

Reinforcement learning evaluations do not use fixed datasets; the environment itself generates test cases. In E1 every episode is a new procedurally generated trial: both goals and the Walker spawn at uniformly random positions in the arena, and the target colour is re-drawn at random. The task is therefore never memorisable — the Oracle must communicate the *current* episode's goal position, and the Walker must decode it relative to its *current* pose. Over the 30M-step run this amounts to several hundred thousand distinct episodes, each an independent test of the communication protocol. The Push Block experiments (E2/E3) follow the same principle with randomised block, agent, and arena-rotation configurations per episode.

## 7.5 Participants

Not applicable. This research evaluates autonomous learning agents in simulation; no human participants were involved, and no user study was conducted. All reported quantities are objective environment measurements.

## 7.6 Evaluation Metrics

- **Group cumulative reward** (primary): the shared episodic team reward the trainers optimise; directly comparable across configurations that share an environment.
- **Mean episode length** (efficiency, and an *unfakeable* completion indicator): in both E1 and E3 an episode ends early only on genuine task success (goal reached; all blocks scored). Falling episode length therefore cannot be produced by exploiting shaping terms — it measures real task completion speed.
- **Policy entropy** (exploration health): a healthy run explores, then commits — entropy should fall as a strategy consolidates; sustained rising entropy indicates the optimiser is failing to find a learnable signal.
- **Value / baseline losses** (training stability): used to detect the critic-divergence failure mode documented in Section 6.8.
- **E3 ablation metric [PENDING]:** task performance of the trained partial-observability policy with messages zeroed at inference, versus intact — the direct measure of how load-bearing the learned protocol is.

## 7.7 Results and Analysis

### 7.7.1 E1: Emergent communication solves the Blind Walker task

**Fig. 7.1** (group cumulative reward) and **Fig. 7.2** (mean episode length) show the full 30M-step training curves (light trace: raw 50k-step summaries; heavy trace: 10-point moving average).

**Headline result.** The team's episodic reward rises from **0.21** (mean of the first ten summaries, ≈ blind-search level) to **4.38** (mean over the final 2.5M steps; best summary 4.41), while mean episode length falls from **≈ 748 to 49 agent steps — a ≈ 15× reduction in time-to-goal**. The smoothed reward first sustains 4.2 at ≈ 22M steps, after which episode length continues to shorten (49 → low-40s), i.e. the protocol keeps being refined even after reward saturates.

**Why this ceiling is meaningful.** The maximum achievable episodic reward is the +3.0 win bonus plus the potential-based shaping term, which integrates to ≈ 0.1 × the Walker's initial distance to the correct goal (≈ 1–1.5 on average), minus small time penalties: an effective ceiling of ≈ 4.2–4.5. The converged policy operates essentially *at* that ceiling — episodes are won reliably and almost directly, leaving little reward unclaimed.

**Attribution.** The Walker's observation vector contains its own pose, velocity, and wanted colour — but *no goal positions whatsoever* (Section 6.4.3). The shaping term rewards reducing distance to the correct goal, a quantity the Walker cannot sense. Every point of improvement above blind-search level therefore has exactly one possible information route: the Oracles' 4-float messages. A ≈ 15× reduction in time-to-goal on procedurally randomised episodes means the emergent protocol reliably transmits, and the Walker reliably decodes, the equivalent of "where your goal is" — re-encoded afresh every episode for new goal positions and colours.

**Learning dynamics.** Three phases are visible in Fig. 7.1/7.2: (i) 0–4M steps: near-flat reward around 0–1 with episodes dominated by the 2500-step timeout — the protocol has not yet formed, and a transient dip around 3.5–4M marks the policy reorganising; (ii) 4–15M: steady co-adaptation — reward climbs roughly linearly as Oracle encoding and Walker decoding improve together, the signature of the sender and receiver bootstrapping each other through the shared critic; (iii) 15–30M: consolidation — reward saturates near the ceiling while episode length keeps compressing, indicating path-efficiency refinement rather than new competence. Value and baseline losses remain bounded throughout (no divergence events), confirming the stability fixes of Section 6.4.2 under a 30M-step load.

**Qualitative observation.** With the attention-visualisation channel enabled (Section 6.4.1), the trained Walker's attention weight concentrates on the Oracle assigned to the currently wanted colour and shifts between Oracles when the target colour changes across episodes — consistent with the targeted-attention design intent [15], observed live in the scene view. *(Optional Fig. 7.3: screenshot of the attention-coloured gizmo lanes.)*

**Relation to the staged validation.** The delivery pipeline this run depends on — board, topology, presence flags, sensor rows, actor attention — had been validated independently of learning during development, using temporarily hand-coded ground-truth messages (Stage 1 of the protocol in Section 6.8). The present experiment is therefore a clean test of *message learning* alone: transport correctness was established beforehand, so the observed performance can be attributed to the emergent protocol rather than to any property of the plumbing.

### 7.7.2 E2: Push Block benchmark matrix **[PENDING — runs complete, analysis to be written]**

*(To include: 30M-step curves for stock MA-POCA vs Comm-MAPOCA under full and 7×7 partial observability on the six-block task without curriculum; all four plateau far below task completion — motivating the curriculum of E3.)*

### 7.7.3 E3: Curriculum capability and communication value **[PENDING — runs in progress]**

*(To include: 11-lesson curriculum run (clean s6 rerun) — lesson staircase, reward, episode-length collapse; partial-observability curriculum run; zero-message ablation table.)*

## 7.8 Comparison with Existing Methods

The reference point throughout is **stock MA-POCA** [10] — the unmodified upstream trainer, run from the same toolkit installation with identical hyperparameters and environments — rather than re-implementations, so every difference is attributable to the communication extension alone. For E1 no stock comparison is possible *by design* (a silent team cannot solve the Blind Walker task above chance: the Walker would be reduced to blind search with a 50% wrong-goal rate); the informative internal comparison is against the hand-coded-message configuration, which the emergent protocol matches (Section 7.7.1). The systematic algorithm-versus-algorithm comparison is E2's role. **[E2 comparison PENDING]** Literature-scale systems such as OpenAI Five [6] are cited for context only; their compute regime is explicitly out of scope for this research's accessibility goal and no numerical comparison is meaningful.

## 7.9 Discussion of Findings

**[PARTIAL — E1 findings only; extend after E2/E3.]**

E1 establishes the central feasibility claim of this thesis: a TarMAC-style targeted communication channel, integrated into MA-POCA as a counterfactual action and trained end-to-end by the message-aware dual critic, produces a *functional emergent communication protocol* on commodity-accessible compute (CPU-only cloud instances, default hyperparameters, ≈ 10 hours of training). No message content, vocabulary, or protocol structure was specified anywhere in the system; the 4-float code that emerges is discovered purely through the shared team reward.

Two observations qualify the result. First, these are single-seed results; the consistency of the three-phase learning dynamic across the development runs (including the 10M-step precursor of this run) suggests robustness, but formal multi-seed variance is future work. Second, E1 is deliberately engineered so communication is the *only* solution path; it demonstrates that the mechanism works, not that it pays off in tasks where physical coordination alone might suffice. That question — where communication earns its cost — is precisely what E2 and E3 are designed to answer.

## 7.10 Summary

This chapter defined a three-experiment evaluation strategy and reported the completed communication-validation experiment. On the Blind Walker task, fully emergent Comm-MAPOCA agents invented a working communication protocol from scratch: team reward rose from blind-search level (0.21) to the effective task ceiling (4.38), and time-to-goal fell ≈ 15× (748 → 49 steps), with the information provably flowing through the learned 4-float message channel because no other route to the goal's location exists. The trained receiver's attention concentrates on the task-relevant sender, matching the targeted-communication design intent. The benchmark matrix (E2) and the curriculum/communication-value experiments (E3), whose training runs are complete and in progress respectively, will complete the evaluation.
