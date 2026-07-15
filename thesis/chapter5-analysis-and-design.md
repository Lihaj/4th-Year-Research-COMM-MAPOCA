# CHAPTER 5 — ANALYSIS AND DESIGN

> Rewrite of the progress-report Chapter 5, reconciled with the system as actually
> implemented (Chapter 6). Principal correction: the TarMAC message structure — the
> sender broadcasts ONLY message content; signature keys and values are derived by
> the RECEIVER (see 5.2 and 5.3.3). Citation numbers `[n]` follow the locked
> reference list [1]–[18]. Figure numbers are placeholders; the architecture
> diagram (old Figure 3) must be updated to match — notes at the end of the file.

## 5.1 Introduction

This chapter translates the approach defined in Chapter 4 into a concrete system design. It details the structural blueprint of the Comm-MAPOCA extension, analysing the modules required to bridge the research gap. It presents the top-level architecture, defines the message structure and its lifecycle across the Unity/PyTorch boundary, maps the explicit data flow between the decentralised actors and the centralised dual critic, and specifies the integration strategy with the Unity ML-Agents framework [17]. The design presented here is the one realised in Chapter 6; where the design deliberately departs from its inspirations (in particular the TarMAC message format [15]), the departure and its rationale are stated explicitly.

## 5.2 Rationale for the Design of the Extension

The architectural design of Comm-MAPOCA is driven by four principles.

**(1) Strict CTDE compliance.** The design preserves the Centralised Training and Decentralised Execution paradigm [8]: all message-aware machinery added to the critic exists only at training time, while at execution time each agent runs a single lightweight actor network whose only additional inputs are the messages it receives. The core design challenge was integrating communication without breaking MA-POCA's ability to handle dynamic, posthumous agent states [10].

**(2) Communication as a counterfactual action.** Messages are treated not as hard-coded signals but as a continuous "vocal action": the agent's outgoing message is emitted by the same policy network that emits its physical action, and the centralised critic evaluates the message's utility exactly as it evaluates physical movement. Routing the generated message mⱼ into the counterfactual baseline network alongside the physical action aⱼ forces the critic to marginalise the pair (aⱼ, mⱼ) — so the resulting advantage directly answers: *what did this agent's action and utterance, together, contribute to the team?* An agent can therefore broadcast a critical message, terminate, and still be credited if that message later helps the surviving team — the posthumous-credit property [10] extended to communication.

**(3) A minimal wire format, with all learning on the Python side.** The original TarMAC design has each sender broadcast a *signature key* alongside its message value, and receivers attend over the broadcast keys [15]. This design adapts that mechanism: **the sender broadcasts only the message content itself** (a small bounded vector), and the **receiver derives both the attention key and the value from the received content through its own learned projections**. The adaptation is motivated by three considerations:

- *Wire economy.* Every float an agent must broadcast is an extra continuous action dimension the policy must learn to control. Broadcasting content only (4 floats) rather than content plus a signature key roughly halves the communication action space, which matters for sample efficiency in exactly the low-resource settings this research targets.
- *Learnability in one place.* With receiver-side projections, every learnable communication component lives inside the PyTorch actor, where it is trained end-to-end by the standard optimiser; the Unity side remains a dumb, inspectable message board.
- *Preserved targeting capacity.* The receiver's key projection can learn to embed sender-discriminative structure that senders, under the same end-to-end gradient, learn to place in their messages. Targeting is thus learned implicitly through the content channel rather than through a dedicated signature channel.

**(4) A deliberate one-step communication delay.** Messages generated at time *t* are delivered at time *t+1*. This temporal cross-over mimics physical reality — no zero-latency telepathy — and forces agents to learn predictive, causal cooperation. In the realised design this delay is not implemented as an explicit buffer policy but arises naturally from the ordering of the environment loop: messages are posted during the action phase of step *t* and read during the observation phase of step *t+1* (Section 5.4).

## 5.3 Top-Level Architecture

The top-level architecture fuses the adapted targeted-communication protocol into the dual-critic MA-POCA system. As illustrated in Figure 5.1, the system divides between execution-time components (the environment-side communication infrastructure and the decentralised actors) and training-time components (the centralised dual critic and the optimisation loop).

**The message structure.** One message is a vector of **4 floats, bounded to [−1, 1]** by a tanh applied at the environment boundary, so that unbounded policy outputs can never destabilise the (unnormalised) critic that also consumes messages. A message *as delivered to a receiver* is a row of **5 floats: [content₀ … content₃ | presence]**, where the explicit presence flag distinguishes a real, already-broadcast message (1) from the slot of a registered teammate that has not yet spoken this episode (0). The flag exists because a silent-but-real teammate is otherwise indistinguishable from structural padding — both are all-zeros — and the receiving attention must be able to mask empty slots exactly rather than inferring emptiness from message magnitude. Delivered rows appear in a stable order (sender registration order), so the k-th attention weight always refers to the same teammate.

### 5.3.1 Preprocessing Module

The preprocessing module operates within the Unity engine and prepares all tensors the neural networks consume. It (i) gathers each agent's local observations (grid-sensor and vector observations); (ii) maintains the **message board** for the arena: every sender's latest message is stored once, and each receiver's view of the board is materialised as the 5-float rows described above, delivered through a dedicated variable-length **CommBuffer** observation sensor; (iii) enforces the communication **topology** (all-to-all for homogeneous teams, or explicit directed sender→receiver links for asymmetric tasks) and the **self-message exclusion rule** — an agent never receives its own message, which keeps mⱼ cleanly out of agent j's observations so the counterfactual baseline can marginalise it (Section 5.3.3); (iv) acts as the temporal holding point that realises the one-step delay, retaining messages posted at *t* for delivery at *t+1*; and (v) packages the shared team reward and tracks the dynamic number of active agents, as in stock MA-POCA [10].

### 5.3.2 ML Engine

The ML Engine is the core reinforcement-learning optimisation loop, built on PyTorch and inherited from the toolkit's on-policy trainer: trajectory memory buffers, the clipped-surrogate policy loss with trust-region value losses [16], TD(λ)/GAE return estimation, and backpropagation. It ingests the advantage scores produced by the centralised critic and applies gradient updates to the actor networks. The engine is configured entirely through the stock MA-POCA hyperparameter schema, so any valid MA-POCA configuration is a valid Comm-MAPOCA configuration — a deliberate design property that guarantees like-for-like benchmarking against silent baselines.

### 5.3.3 Extended Module

This module contains the novel contributions of the design.

**The Communicative Actor.** Each agent's policy network is extended in three ways:

1. *Message reception.* The CommBuffer is deliberately excluded from the actor's generic observation encoder. Instead, the received rows enter through a dedicated **targeted communication block**: the receiver predicts a query **q** from its own encoded hidden state; each received message content vector is projected into a key **kᵢ** and a value **vᵢ** by the receiver's learned projections; scaled dot-product attention α = softmax(q·kᵀ/√d_k) over the presence-masked keys yields an aggregated context vector **c = Σ αᵢvᵢ** (signature dimension d_k = 16).
2. *Message-conditioned action selection.* The action heads consume the concatenation **[own encoding | c]** — double the standard encoding width — so the physical action is conditioned on what teammates said.
3. *Message emission.* The outgoing message is emitted implicitly as the **first 4 continuous actions** of the policy's action vector, bounded at the environment boundary. No dedicated "communication head" architecture is required: the standard action model learns the message dimensions under the same PPO-style objective as movement, which is precisely what allows the critic to treat the message as an action.

**The Message-Aware Dual Critic.** Both heads of the MA-POCA critic are extended with a deliberate message channel, realised as additional entities inside the critic's Residual Self-Attention over the team [10]:

- The **State-Value network** V(s) receives, alongside every agent's observation entities, the **sent messages of all agents** — the full communication state of the team at that step.
- The **Counterfactual Baseline network** for agent j receives j's observations *without* its action, the groupmates' observation–action pairs, and — on the message channel — **j's received messages plus the groupmates' sent messages**. Agent j's own sent message mⱼ is absent (guaranteed by the self-message exclusion rule upstream). The baseline therefore marginalises **(aⱼ, mⱼ)** while conditioning on **(oⱼ, cⱼ)**: it estimates what the team would have achieved had agent j neither moved nor spoken, given everything j saw and heard.

Two structural safeguards complete the design: message entities are appended to the critic's attention set only *after* the active-agent count is computed, so communication slots never distort the agent-count signal the value heads rely on for dynamic team sizes [10]; and the message channel must be fed identically wherever the critic is evaluated — both when generating value targets and when optimising against them — since targets produced by a message-free critic cannot be chased by a message-fed one without divergence (this consistency requirement is enforced structurally in the implementation, Chapter 6).

## 5.4 Data Flow and Interaction Design

The data flow traces one observation's lifecycle through the system across two timesteps:

1. **Forward pass (execution), step t.** Agent i receives its local observation oᵢ(t) and its CommBuffer rows — the messages posted by allowed senders at step *t−1*, each with its presence flag. The actor encodes oᵢ, attends over the received rows to form the context cᵢ, and outputs a physical action aᵢ(t) together with a new message mᵢ(t) (the first four continuous action floats).
2. **Interaction routing.** The physical action drives the agent in the environment. Simultaneously the environment-side board stores mᵢ(t) (tanh-bounded); at the observation phase of step *t+1* the board materialises each receiver's rows in stable sender order — closing the one-step delay loop.
3. **Critic evaluation (training).** Trajectories of observations, actions, messages, and team rewards are batched to the centralised critic. The State-Value network processes all agents' observation entities plus all sent messages to estimate expected return; TD(λ) targets are computed from these estimates.
4. **Backward pass (optimisation).** For each agent j, the Counterfactual Baseline processes the team with (aⱼ, mⱼ) marginalised out and (oⱼ, cⱼ) retained. The difference between the return target and this baseline yields agent j's advantage — a per-agent measure of the joint contribution of *what it did and what it said* — which flows back through the shared optimiser to update both the movement and the message dimensions of the actor.

For interpretability during training, the design also routes the receiver's attention weights α back to the environment (through otherwise-unused action dimensions), where the message board renders live communication-flow lines coloured by attention weight — making "who is listening to whom" directly observable in the scene view.

## 5.5 Extension Integration into the AI Framework

Integration with Unity ML-Agents [17] is specified as three narrow contracts, chosen so that **no stock toolkit file is modified**:

1. **Sensor contract.** The communication stream is a variable-length buffer sensor with the reserved name `CommBuffer` (row size 5 = content 4 + presence). The trainer identifies the communication stream by this name and treats it specially on both actor and critic; all other sensors flow through stock code paths. A behavior without the sensor trains as plain MA-POCA — graceful degradation by design.
2. **Action-space contract.** The behavior's continuous action space is widened: slots [0..3] are the message; any additional continuous slots are reserved for the attention-weight visualisation channel. The environment consumes only its discrete branch for physics, so Unity never interprets communication floats as movement.
3. **Plugin contract.** The trainer registers as a new trainer type (`comm_mapoca`) through the toolkit's official plugin interface, reusing the stock MA-POCA settings schema. The stock `poca` trainer remains installed and untouched, guaranteeing that baseline comparisons in Chapter 7 run the genuine unmodified algorithm.

Because the entire communication pathway (attention block included) lives inside the policy network, exported models carry it intact: trained Comm-MAPOCA agents deploy through the standard in-engine inference path with no Python dependency — the same deployment story as any stock ML-Agents model, which is essential to the accessibility claim of this research.

## 5.6 Summary

This chapter presented the realised design of Comm-MAPOCA. Communication is designed as a bounded 4-float message treated as a counterfactual action: emitted by the standard policy head, delivered with a one-step delay through an environment-side message board as presence-flagged rows, received through a targeted attention block in which the receiver derives keys and values from message content (a deliberate, justified adaptation of TarMAC's sender-broadcast signature scheme [15]), and evaluated by a message-aware dual critic whose counterfactual baseline marginalises each agent's action–message pair. The three-module architecture — environment-side preprocessing, the stock optimisation engine, and the extended actor/critic module — integrates with Unity ML-Agents through three narrow contracts that leave the stock toolkit untouched. Chapter 6 details how each element of this design is implemented.

---

### Figure notes (for the Word document)

- **Fig. 5.1 (replaces old Figure 3)** — Comm-MAPOCA top-level architecture. Required corrections vs. the progress-report diagram: (i) sender side broadcasts ONLY the 4-float message — remove any broadcast Key/signature arrows; (ii) receiver-side attention block shows q from own encoding, k/v projected from received content; (iii) CommBuffer rows drawn as [m₀..m₃ | presence]; (iv) baseline network input annotated "marginalised: aⱼ, mⱼ — retained: oⱼ, cⱼ"; (v) value network input annotated "all agents' sent messages"; (vi) t → t+1 delay arrow through the message board.
- **Fig. 5.2 (optional)** — Message structure diagram: 4 tanh-bounded content floats + presence flag, stable row ordering per receiver.
