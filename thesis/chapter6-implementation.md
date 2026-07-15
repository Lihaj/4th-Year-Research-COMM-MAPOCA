# CHAPTER 6 — IMPLEMENTATION

> Draft for the final thesis (replaces the interim progress-report Chapter 6 entirely).
> Citation numbers `[n]` follow the locked reference list [1]–[18] from the progress report.
> Figure/table numbers are placeholders — renumber when pasting into the Word template
> (template rules: table captions ABOVE in caps, figure captions BELOW as "Fig. 6.x: ...").

## 6.1 Introduction

This chapter describes the complete implementation of the Comm-MAPOCA system designed in Chapter 5. The implementation comprises three modules: (i) a reusable Unity C# package, `com.lihaj.comm-mapoca`, which provides the environment-side communication infrastructure; (ii) an in-tree Python trainer plugin, `comm_mapoca`, which extends the stock MA-POCA trainer [10] of the Unity ML-Agents Toolkit [17] with a TarMAC-style targeted communication pathway [15]; and (iii) a set of evaluation environments — the Blind Walker validation task and a family of communication-enabled Cooperative Push Block variants — built on top of the first two modules.

A deliberate engineering principle runs through the whole implementation: **the stock toolkit is never modified**. All stock files under `mlagents/trainers/poca` and `mlagents/trainers/torch_entities` remain byte-identical to the release_23 distribution, and the stock C# example scripts are untouched. Every Comm-MAPOCA capability is added through subclassing, composition, and the toolkit's official plugin interface. This guarantees that the silent MA-POCA baselines used in Chapter 7 are the genuine, unmodified algorithm — a prerequisite for a fair comparison — and demonstrates that the extension is deployable by its target users (small studios) without forking the toolkit.

## 6.2 Overall Development Environment

The development environment is bifurcated into two ecosystems that communicate through the ML-Agents gRPC bridge [17]:

- **Simulation side (C#/Unity).** The environments, the communication package, and all reward/reset logic run inside the Unity Editor during development. The ML-Agents clone (release_23) is embedded in the project so that the C# `com.unity.ml-agents` package and the Python trainers are always version-locked to each other.
- **Learning side (Python/PyTorch).** Training runs in a dedicated Python 3.10.12 virtual environment, matching the version required by the toolkit's official installation guide. The `mlagents` and `mlagents-envs` packages are installed *editable* from the embedded clone, so the `comm_mapoca` trainer plugin is picked up directly from the working tree.

Two training workflows were used:

1. **Local (interactive):** `mlagents-learn` connects to the Unity Editor in Play Mode. Used for all environment debugging, curriculum development, and short experimental runs, with live Scene-view visualisation of the communication gizmos (Section 6.4.1).
2. **Cloud (headless):** for long benchmark runs, the environment is compiled as a Linux Dedicated Server build and trained on Kaggle's free CPU instances. A reproducible notebook pipeline installs the exact Python version with `uv`, clones the project repository, and resumes interrupted runs from checkpoints across the platform's ~12-hour session limit. This pipeline made 30-million-step training runs feasible without local GPU-days — directly supporting the accessibility argument of Chapter 1.

## 6.3 Hardware and Software Platforms

**TABLE 6.1: DEVELOPMENT AND TRAINING PLATFORMS**

| Component | Local development machine | Cloud training (Kaggle) |
|---|---|---|
| CPU | AMD Ryzen 7 5800H (8C/16T) | 4 vCPU (shared) |
| GPU | NVIDIA GeForce RTX 3060 (CUDA) | None (CPU-bound workload) |
| RAM | 16 GB | 30 GB |
| Role | Environment development, curriculum runs, visual verification | Long 30M-step benchmark runs, resume-chained sessions |

**TABLE 6.2: SOFTWARE STACK**

| Layer | Version |
|---|---|
| Unity Editor | 6000.4.0f1 |
| Unity ML-Agents Toolkit | release_23 clone (`mlagents` 1.2.0.dev0, embedded) |
| Python | 3.10.12 |
| PyTorch | 2.5.1 (+cu121 locally; CPU wheels on Kaggle) |
| Communication package | `com.lihaj.comm-mapoca` (this work) |
| Trainer plugin | `mlagents.trainers.comm_mapoca` (this work) |

## 6.4 Module-wise Implementation

### 6.4.1 Module 1: The `com.lihaj.comm-mapoca` Unity Package

The environment-side communication infrastructure is packaged as a self-contained Unity package with two runtime classes.

**`CommChannel` — the message board.** One `CommChannel` component per arena acts as the team's shared message board. Senders post a fixed-size float message every action step; receivers read the board through a buffer sensor. Its key mechanics:

- **Message format.** A message is `messageSize` floats (4 in all experiments). Each *delivered* row is `messageSize + 1` floats: the content plus an explicit **presence flag** (1 = the sender has spoken this episode; 0 = a registered-but-silent slot). The presence flag lets the trainer's attention mask empty slots explicitly instead of guessing from message magnitude — a real agent that has not yet spoken is otherwise indistinguishable from structural padding (both are all-zeros).
- **Stable row order.** Rows are delivered in sender-registration order, so attention weight *k* always refers to the same teammate across steps.
- **Topology.** `AllToAll` (used by Push Block) or `Directed` with explicit sender→receiver links (used by the Blind Walker task to give the Oracles a one-way line to the Walker).
- **Self-message exclusion.** By default an agent never receives its own message. This is not merely hygiene: the MA-POCA counterfactual baseline must be able to marginalise an agent's own sent message out (Section 6.4.2), which requires that message to be cleanly absent from the agent's observations.
- **Visualisation.** In Play Mode the channel draws communication-flow gizmo lines between agents, coloured red→green by the receiver's attention weight over that sender (Section 6.5 explains how the weights reach C#). Opposite directions of a pair are offset into parallel lanes so that A-listens-to-B and B-listens-to-A remain individually visible.

**`CommAgent` — the communicating agent base class.** `CommAgent` derives from the toolkit's `Agent` and *seals* the four standard lifecycle overrides (`Initialize`, `CollectObservations`, `OnActionReceived`, `OnEpisodeBegin`), exposing `OnComm*` virtual equivalents instead. Sealing guarantees the communication plumbing always runs, in the right order, no matter what a derived environment agent does. Automatically handled:

- configuring the required `BufferSensorComponent` as the `"CommBuffer"` sensor the trainer looks for (row size and capacity taken from the channel);
- appending the incoming rows to that sensor every observation step;
- broadcasting the **first `messageSize` continuous actions, tanh-bounded to [−1, 1]**, as the outgoing message every action step (the bound prevents unbounded policy outputs from destabilising the unnormalised critic that also consumes these messages);
- clearing the agent's posted message when its episode begins.

`CommAgent` also exposes an `OverrideOutgoingMessage` hook: an environment may substitute a scripted message for the learned one. This single hook is what enabled the staged hand-coded-message validation protocol of Section 6.8.

### 6.4.2 Module 2: The `comm_mapoca` Trainer Plugin

The Python side lives entirely in `mlagents/trainers/comm_mapoca/` (four files) and is registered through the toolkit's official trainer-plugin interface, making it selectable in any run configuration with `trainer_type: comm_mapoca`. It reuses the stock `POCASettings` hyperparameter schema, so any valid MA-POCA configuration is a valid Comm-MAPOCA configuration — an intentional property for like-for-like benchmarking.

**`TargetedCommunicationBlock` — receiver-side targeted attention.** The actor's attention over incoming messages follows TarMAC's query/key/value formulation [15]: the receiver predicts a query from its hidden state; each incoming message is projected into a key (signature) and a value; scaled dot-product attention selects whose value to aggregate:

α = softmax(q·kᵀ / √d_k),  c = Σᵢ αᵢ vᵢ  (signature dimension d_k = 16)

One deliberate deviation from the original TarMAC design must be noted: in TarMAC the *sender* broadcasts a separate signature key alongside its value. In this implementation the sender broadcasts only the 4-float message itself, and the **receiver** derives both key and value from the message content through its own learned projections. This keeps the Unity-side wire format minimal (4 floats + presence flag), keeps every learnable component inside PyTorch, and halves the action-space overhead each agent pays for communication; targeting capacity is preserved because the key projection can still learn to embed sender-discriminative structure that senders learn to place in their messages. The block masks padded/silent rows using the explicit presence flag, and guards the softmax against all-masked rows with an ONNX-opset-9-compatible NaN replacement so exported models remain valid.

**`CommSimpleActor` — the communicating policy.** A subclass of the stock `SimpleActor` that:

1. splits the `"CommBuffer"` observation away from the normal observations, so the base network body never sees raw messages — the message pathway stays deliberate (through the attention block), never a generic input;
2. encodes the normal observations with the stock network body, then attends over the incoming messages with the `TargetedCommunicationBlock`;
3. conditions the action heads on the concatenation **[own encoding | aggregated message]** (the action model is rebuilt at exactly double the encoding width);
4. emits the outgoing message implicitly as the first 4 continuous actions (bounded on the C# side by `CommAgent`).

The actor also implements the attention-visualisation channel: at inference time during training, the receiver's attention weights are written into the *spare* continuous action slots after the message and read back by `CommAgent` for the gizmo colouring. A subtle correctness detail: the action tuple sent to Unity shares memory with the tensor stored in the on-policy training buffer, so the weights are written into a **fresh copy** — an in-place write would silently corrupt the training data.

**`CommMultiAgentNetworkBody` — the message-conditioned centralised critic.** A subclass of the stock `MultiAgentNetworkBody` whose Residual Self-Attention entity set can be *extended with message entities*: each 4-float message is projected by a dedicated encoder into the same embedding space as the observation and observation–action entities and participates in the critic's self-attention. Two ordering details matter: messages are appended **after** the agent count is computed (so message slots never inflate the normalised agent-count scalar the value heads rely on), and message slots carry their own attention masks.

**`TorchCommMAPOCAOptimizer` — the dual critic with a deliberate message channel.** The optimizer builds the stock MA-POCA dual-critic structure (state-value network + counterfactual baseline network [10]) on top of `CommMultiAgentNetworkBody`, with the following Comm-MAPOCA-specific wiring:

- **CommBuffer stripping.** The `"CommBuffer"` sensor is removed from the observation specs the critic is built from, and from every observation list fed to it. Without this, the critic's generic observation encoder would pool raw message buffers through its own uncoordinated attention — an accidental, *asymmetric* leak (only agents carrying the sensor would be affected) instead of the deliberate channel described next.
- **The message channels.** A single helper constructs the critic's message tensors for both critic heads: the **value network** receives the *sent* messages of every agent (tanh-bounded to match what C# actually broadcasts); the **counterfactual baseline for agent j** receives j's *received* messages (presence-masked; j's own sent message is excluded on the C# side by the self-message rule) plus the other agents' sent messages. The baseline therefore marginalises the pair **(aⱼ, mⱼ)** — agent j's physical action *and* its message — while keeping **(oⱼ, cⱼ)**, exactly the "message as a counterfactual action" semantics designed in Chapter 5: the advantage signal directly measures what agent j's message contributed to the team's expected return.
- **Train/target consistency contract.** The message channel is gated by a single master switch honoured by *both* the training update and the TD(λ)/GAE target-generation pass. This is a hard invariant: if targets were produced by a message-free critic while the update optimised a message-fed critic (or vice versa), the value network would chase targets from a different function and diverge. The recurrent (LSTM) evaluation path is not threaded with messages and deliberately raises an error rather than silently falling back to a message-free pass.
- **Gradient clipping.** The stock toolkit applies no gradient clipping. The extended critic — additional message entities inside attention — was observed to occasionally produce a single enormous gradient that blew the value/baseline heads up to millions (a stable-then-explode signature after several million steps). A global gradient-norm clip (max-norm 0.5) was added to the shared optimiser; it is inert during normal training and caps exactly these pathological updates.
- **Bootstrap approximation.** For truncated (non-terminal) trajectory tails, the next step's sent messages do not exist yet; they are approximated with the final step's messages. Terminal episodes zero the bootstrap anyway, so the approximation only touches truncated tails.

**`CommMAPOCATrainer` — trainer subclass and a curriculum fix.** The trainer subclass selects the comm actor and optimizer, and contributes one fix of independent interest: the stock trainer fills the `reward_buffer` — the statistic that curriculum-lesson completion criteria compare against — with *individual* episode rewards, which remain ≈0 in a purely group-rewarded cooperative task. Curricula therefore never advance for group-reward trainers, regardless of team performance: a genuine limitation of the upstream toolkit uncovered by this work. `CommMAPOCATrainer` mirrors the episodic *group* reward into the buffer (replacing the stock entry), which makes ML-Agents curricula usable with group-reward training and enabled the curriculum experiments of Chapter 7.

### 6.4.3 Module 3: Evaluation Environments

**Blind Walker (communication validation task).** A purpose-built task in which communication is *necessary by construction*: two stationary **Oracle** agents each see the position of one colour-coded goal and the Walker, while the mobile **Walker** knows which colour it must reach but cannot see any goal. The only route from goal positions to the Walker's movement is the message channel (Directed topology, Oracles → Walker). The environment controller provides a win reward, timeout penalty, existential time pressure, and potential-based "getting warmer" shaping measured to the correct goal — a quantity the blind Walker can only improve by exploiting the Oracles' messages. One implementation subtlety: ML-Agents requires all members of a `SimpleMultiAgentGroup` to share one behavior name, so the heterogeneous Oracle/Walker roles are unified into a single role-flagged behavior — a shared 9-float observation layout in which element 0 is a role flag and the remaining slots are interpreted per role. This constraint and its solution are documented because any practitioner building heterogeneous communicating teams in ML-Agents will encounter it.

**Communication-enabled Cooperative Push Block (benchmark family).** The stock cooperative Push Block example is recreated in communication-enabled form: `CommPushAgentCollab` derives from `CommAgent` and reproduces the stock movement mapping exactly, adding 6 continuous actions (4 message + 2 attention-visualisation spares) alongside the stock discrete branch. A shared environment controller (`CommPushBlockEnvController`) mirrors stock reward/reset semantics and adds strictly **opt-in** extensions, each defaulting to stock-identical behaviour: an 11-lesson curriculum table driven by the `pb_lesson` environment parameter (block counts, spawn spread, and temporary "light-mass" introductions of heavy blocks), near-goal spawn bands with collision re-checks, potential-based heavy-block shaping, and a record-distance ratchet reward. Scene variants instantiate the family: full observability (20×20 grid sensor), partial observability (the same 20×20 grid at 0.35 cell scale — a 7×7-world-unit local view), and their curriculum versions. The opt-in principle means one controller serves benchmark, curriculum, and ablation scenes without any risk of configuration leaking into the baselines.

## 6.5 Core Algorithms and Pseudocode

The complete per-timestep message lifecycle, spanning both ecosystems:

```
# --- Unity, action step t (CommAgent.OnActionReceived) ---
m_i(t) = tanh(continuous_actions[0:4])          # learned message, bounded
channel.PostMessage(agent_i, m_i(t))            # onto the board

# --- Unity, observation step t+1 (CommAgent.CollectObservations) ---
for sender in channel.GetAllowedSenders(agent_i):        # stable row order
    row = [m_sender | presence]                          # presence: spoken yet?
    CommBuffer_i.append(row)

# --- Python, policy forward pass (CommSimpleActor) ---
h_i        = NetworkBody(normal_obs_i)                   # CommBuffer excluded
q          = W_q h_i                                     # receiver query
K, V       = W_k M_in, W_v M_in                          # keys/values from content
alpha      = softmax(q K^T / sqrt(d_k))  [presence-masked]
c_i        = alpha V                                     # aggregated message
a_i, m_i   = ActionModel([h_i | c_i])                    # heads on double width

# --- Python, training update (TorchCommMAPOCAOptimizer) ---
V(s)       = ValueNet(all obs entities, all obs-action entities,
                      sent messages of ALL agents)        # message entities in RSA
b_j        = BaselineNet(o_j, groupmates' (o,a),
                      j's RECEIVED messages + groupmates' sent messages)
             # (a_j, m_j) marginalised out; (o_j, c_j) kept
A_j        = TD(lambda) return - b_j                      # message-aware advantage
```

The messages posted at step *t* are observed at step *t+1* — the one-step communication delay designed in Chapter 5, which arises naturally from the post-then-collect ordering rather than from any explicit buffering logic.

## 6.6 Workflow Diagrams / Flowcharts

*(Figures to produce for the Word document:)*

- **Fig. 6.1** — Two-ecosystem architecture: Unity (CommChannel, CommAgent, environments) ↔ gRPC ↔ Python (CommSimpleActor, TargetedCommunicationBlock, dual critic), with the CommBuffer sensor and the action-slot message highlighted as the two halves of the wire protocol.
- **Fig. 6.2** — Per-timestep message lifecycle (the pseudocode of Section 6.5 as a sequence diagram across t and t+1).
- **Fig. 6.3** — Critic message channels: value network (all sent messages) vs. counterfactual baseline for agent j (received messages + others' sent), annotated with the marginalised set (aⱼ, mⱼ).
- **Fig. 6.4** — Blind Walker arena screenshot with attention-coloured gizmo lanes (Oracle→Walker), demonstrating the built-in interpretability tooling.

## 6.7 Integration of Components

Integration rests on three narrow, explicit contracts:

1. **The sensor contract.** The C# package names its buffer sensor `"CommBuffer"`; the Python actor and optimizer locate the communication stream by that name, treat it specially (attention pathway on the actor; stripped and replaced by the deliberate channel on the critic), and pass every other sensor through stock code paths. A behavior without the sensor trains as plain MA-POCA — the plugin degrades gracefully.
2. **The action-space contract.** The first 4 continuous actions are the message; any further continuous slots are the attention-visualisation spares. Unity never interprets these as movement because the environment agents consume only their discrete branch for physics.
3. **The plugin contract.** The trainer registers through `mlagents.plugins.trainer_type`, mapping `comm_mapoca` to `CommMAPOCATrainer` with stock `POCASettings`. No toolkit file is edited; the stock `poca` trainer remains available untouched for baselines.

Exported ONNX models embed the full communication pathway (attention block included), so trained Comm-MAPOCA agents run in-engine via the standard Barracuda/Sentis inference path with no Python dependency — the same deployment story as any stock ML-Agents model.

## 6.8 Testing During Development

Testing followed a staged protocol in which each layer was validated before the next was allowed to learn.

**Stage 1 — Hand-coded messages (pipeline validation).** Using the `OverrideOutgoingMessage` hook, the Oracles' outgoing messages were scripted to carry ground-truth goal information while the critic's message channel stayed disabled. This validated the entire delivery pipeline — board, topology, presence flags, sensor rows, actor attention — independently of learning: if the blind Walker could exploit *known-good* messages, the plumbing was correct. In this sanity configuration the Blind Walker task converged decisively in a short local run — development logs record the episodic team reward rising from ≈0 to ≈+3.4 while mean episode length fell roughly sevenfold, with flat, stable value and baseline losses — establishing that any later failure had to lie in the learning of messages, not their transport. (As a development smoke test, this stage was not retained as an archived experiment; its role in the thesis is methodological, and the quantitative results reported in Chapter 7 come exclusively from the archived emergent-communication runs.)

**Stage 2 — Emergent messages.** Hand-coding was then removed and the critic's message channel enabled, requiring the Oracles to *learn* what to say. The Chapter 7 Blind Walker results are from this configuration.

**Debugging findings that shaped the implementation** (each captured earlier in this chapter):

- *Heterogeneous group failure:* ML-Agents rejects mixed behavior names inside one agent group; resolved with the unified role-flagged behavior (Section 6.4.3).
- *Value-function divergence:* stable-then-explode value/baseline losses after millions of steps, traced to unclipped gradients through the message-extended attention critic; fixed with global gradient-norm clipping (Section 6.4.2).
- *Train/target inconsistency:* an early wiring in which only the update path fed messages to the critic reproduced the same divergence signature; fixed by the single-construction-path contract and a deliberate hard failure on the unthreaded recurrent path.
- *Silent-agent masking:* presence flags added after observing that registered-but-silent senders are indistinguishable from padding by magnitude alone.
- *Training-buffer aliasing:* the attention-weight visualisation initially wrote into an action array that aliased the training buffer; caught and fixed with the safe-copy (Section 6.4.2).
- *Curriculum stall:* lessons never advanced under group rewards, traced to the individual-reward `reward_buffer` upstream behaviour; fixed in `CommMAPOCATrainer` (Section 6.4.2).
- *Partial-observability sensor floor:* the toolkit's `simple` visual encoder requires a minimum 20×20 input, so partial observability was implemented by shrinking the grid's cell scale (0.35 ⇒ a 7×7-world-unit view) rather than the cell count.

**Continuous verification.** Every training run was monitored in TensorBoard (group cumulative reward, episode length, entropy, value/baseline losses); episode-length collapse was used as the unfakeable indicator of genuine task completion (an episode only ends early when every block is scored), and the attention gizmos provided live visual confirmation that receivers attended to the intended senders.

## 6.9 Summary

This chapter presented the realised Comm-MAPOCA system: a reusable Unity communication package (`CommChannel`/`CommAgent`), an in-tree trainer plugin implementing receiver-side targeted attention on the actor and a deliberately-wired message channel through both heads of the MA-POCA dual critic, and the evaluation environments built on them. The implementation preserves the stock toolkit untouched, degrades gracefully to plain MA-POCA in the absence of a communication sensor, and packages its interpretability tooling (attention-coloured communication gizmos) into the runtime. Development surfaced and resolved several substantive issues — critic gradient explosions, train/target consistency, heterogeneous-group unification, and an upstream curriculum limitation for group-reward trainers — each of which is documented as part of the contribution. With the system implemented and validated through the staged hand-coded-to-emergent protocol, Chapter 7 evaluates it: emergent communication on the Blind Walker task, the curriculum-driven full Push Block task, and the communication-value experiment under partial observability.
