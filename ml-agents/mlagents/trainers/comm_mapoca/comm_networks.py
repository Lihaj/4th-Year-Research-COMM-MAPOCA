"""
Comm-MAPOCA network components (Lihaj Wickramasinghe, University of Moratuwa).

Extends stock ml-agents (release_23) networks with TarMAC-style targeted
communication (Das et al., 2020) for the MA-POCA trainer (Cohen et al., 2022):

- TargetedCommunicationBlock: query/key/value soft attention over received
  messages (actor side).
- CommSimpleActor: SimpleActor that splits off the "CommBuffer" sensor, attends
  over incoming messages, and conditions its action heads on
  [own encoding | aggregated message].
- CommMultiAgentNetworkBody: MultiAgentNetworkBody whose RSA entity set can be
  extended with message entities, giving the centralized critic (value +
  counterfactual baseline) visibility into what was communicated.

The stock files under mlagents/trainers/poca and torch_entities remain
unmodified; everything Comm-MAPOCA lives in this package.
"""
from typing import List, Dict, Tuple, Optional, Union, Any

import torch.nn.functional as F
from mlagents.torch_utils import torch, nn

from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.attention import EntityEmbedding
from mlagents.trainers.torch_entities.networks import (
    MultiAgentNetworkBody,
    NetworkBody,
    SimpleActor,
)
from mlagents.trainers.settings import NetworkSettings

# Message content floats per message (excluding the presence flag). Matches the
# CommChannel.messageSize on the Unity side; CommBuffer rows are COMM_MSG_SIZE + 1.
COMM_MSG_SIZE = 4
COMM_SENSOR_NAME = "CommBuffer"


class TargetedCommunicationBlock(nn.Module):
    """TarMAC-style targeted communication (receiver side).

    The receiver predicts a query from its hidden state; each incoming message is
    projected into a key (signature) and a value. Scaled dot-product attention over
    the keys selects whose value to aggregate: alpha = softmax(q . k / sqrt(d_k)),
    c = sum(alpha_i * v_i).
    """

    def __init__(self, hidden_dim, msg_size=COMM_MSG_SIZE, signature_dim=16):
        super().__init__()
        self.signature_dim = signature_dim

        self.query_layer = nn.Linear(hidden_dim, signature_dim)
        self.key_layer = nn.Linear(msg_size, signature_dim)
        self.value_layer = nn.Linear(msg_size, hidden_dim)

    def forward(self, agent_hidden_state, raw_messages_buffer):
        # The last column of raw_messages_buffer is an explicit presence flag
        # (1.0 = real, already-broadcast message; 0.0 = structural padding OR a
        # registered agent that hasn't spoken yet this episode), set on the C# side
        # (CommChannel.GetIncomingRows). The remaining columns are message content.
        # Slicing them apart keeps the presence flag out of the Q/K/V math while
        # still using it for masking.
        content = raw_messages_buffer[..., :-1]
        presence = raw_messages_buffer[..., -1]

        Q = self.query_layer(agent_hidden_state).unsqueeze(1)
        K = self.key_layer(content)
        V = self.value_layer(content)

        K_T = K.transpose(1, 2)
        scaled_scores = torch.bmm(Q, K_T) / (self.signature_dim**0.5)

        # Mask using the explicit presence flag instead of inferring "ghost slot"
        # from message magnitude -- a real-but-not-yet-spoken agent is otherwise
        # also all zeros and would be wrongly treated as real content.
        is_padded = (presence < 0.5).unsqueeze(1)
        scaled_scores = scaled_scores.masked_fill(is_padded, -1e9)

        attention_weights = F.softmax(scaled_scores, dim=-1)

        # Safety catch: ONNX Opset 9 compatible NaN replacement
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights,
        )

        aggregated_message = torch.bmm(attention_weights, V).squeeze(1)
        return aggregated_message, attention_weights


class CommMultiAgentNetworkBody(MultiAgentNetworkBody):
    """MultiAgentNetworkBody whose RSA entity set can include message entities.

    Used by the Comm-MAPOCA centralized critic: the value network receives every
    agent's SENT message; the counterfactual baseline receives the target agent's
    RECEIVED messages plus the other agents' sent messages. Messages are appended
    as additional attention entities AFTER the agent count is computed, so message
    slots never inflate the agent-count scalar.
    """

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
    ):
        super().__init__(observation_specs, network_settings, action_spec)
        # Dedicated encoder projecting COMM_MSG_SIZE-float message vectors into the
        # same attention embedding space as the obs / obs-action entities.
        attention_embeding_size = self.h_size
        self.message_encoder = EntityEmbedding(
            COMM_MSG_SIZE, None, attention_embeding_size
        )

    def forward(
        self,
        obs_only: List[List[torch.Tensor]],
        obs: List[List[torch.Tensor]],
        actions: List[AgentAction],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        messages: Optional[torch.Tensor] = None,
        msg_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Same contract as MultiAgentNetworkBody.forward, plus:
        :param messages: optional (batch, N, COMM_MSG_SIZE) tensor of message
            content vectors to add as extra RSA entities alongside obs entities.
        :param msg_mask: optional bool tensor (batch, N); True = padding slot to
            be masked out of attention.
        """
        self_attn_masks = []
        self_attn_inputs = []
        concat_f_inp = []
        if obs:
            obs_attn_mask = self._get_masks_from_nans(obs)
            obs = self._copy_and_remove_nans_from_obs(obs, obs_attn_mask)
            for inputs, action in zip(obs, actions):
                encoded = self.observation_encoder(inputs)
                cat_encodes = [
                    encoded,
                    action.to_flat(self.action_spec.discrete_branches),
                ]
                concat_f_inp.append(torch.cat(cat_encodes, dim=1))
            f_inp = torch.stack(concat_f_inp, dim=1)
            self_attn_masks.append(obs_attn_mask)
            self_attn_inputs.append(self.obs_action_encoder(None, f_inp))

        concat_encoded_obs = []
        if obs_only:
            obs_only_attn_mask = self._get_masks_from_nans(obs_only)
            obs_only = self._copy_and_remove_nans_from_obs(
                obs_only, obs_only_attn_mask
            )
            for inputs in obs_only:
                encoded = self.observation_encoder(inputs)
                concat_encoded_obs.append(encoded)
            g_inp = torch.stack(concat_encoded_obs, dim=1)
            self_attn_masks.append(obs_only_attn_mask)
            self_attn_inputs.append(self.obs_encoder(None, g_inp))

        # Count real agents from obs/obs_only masks BEFORE appending message
        # entities, so message slots don't inflate the agent-count scalar.
        flipped_masks = 1 - torch.cat(self_attn_masks, dim=1)
        num_agents = torch.sum(flipped_masks, dim=1, keepdim=True)
        if torch.max(num_agents).item() > self._current_max_agents:
            self._current_max_agents = torch.nn.Parameter(
                torch.as_tensor(torch.max(num_agents).item()), requires_grad=False
            )
        # num_agents will be -1 for a single agent and +1 at the current maximum
        num_agents = num_agents * 2.0 / self._current_max_agents - 1

        # Add message entities into the RSA entity set AFTER the agent count is
        # locked. Each message is projected via message_encoder, then attends /
        # is attended to by the obs and obs-action entities, giving the critic
        # visibility into what was communicated this step.
        if messages is not None:
            msg_encoded = self.message_encoder(None, messages)
            self_attn_inputs.append(msg_encoded)
            if msg_mask is not None:
                self_attn_masks.append(msg_mask.float())
            else:
                self_attn_masks.append(
                    torch.zeros(
                        messages.shape[0], messages.shape[1], device=messages.device
                    )
                )

        encoded_entity = torch.cat(self_attn_inputs, dim=1)
        encoded_state = self.self_attn(encoded_entity, self_attn_masks)

        encoding = self.linear_encoder(encoded_state)
        if self.use_lstm:
            # Resize to (batch, sequence length, encoding size)
            encoding = encoding.reshape([-1, sequence_length, self.h_size])
            encoding, memories = self.lstm(encoding, memories)
            encoding = encoding.reshape([-1, self.m_size // 2])
        encoding = torch.cat([encoding, num_agents], dim=1)
        return encoding, memories


class CommSimpleActor(SimpleActor):
    """SimpleActor + TarMAC communication.

    Splits the "CommBuffer" observation off from the normal observations, encodes
    the normal observations with the stock NetworkBody, attends over the incoming
    messages with a TargetedCommunicationBlock, and conditions the action heads on
    [encoding | aggregated message]. The outgoing message is simply the first
    COMM_MSG_SIZE continuous actions (tanh-bounded on the Unity side by CommAgent).
    """

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        # Isolate the CommBuffer: the base actor body is built WITHOUT it, so the
        # message path stays deliberate (attention block), never a generic input.
        comm_idx = -1
        normal_specs = []
        for i, spec in enumerate(observation_specs):
            if spec.name == COMM_SENSOR_NAME:
                comm_idx = i
            else:
                normal_specs.append(spec)

        super().__init__(
            normal_specs,
            network_settings,
            action_spec,
            conditional_sigma,
            tanh_squash,
        )
        self.comm_idx = comm_idx

        self.tarmac_block = TargetedCommunicationBlock(
            hidden_dim=self.encoding_size,
            msg_size=COMM_MSG_SIZE,
            signature_dim=16,
        )

        # The action heads consume [encoding | aggregated message], so the
        # ActionModel input is exactly double the encoding size (replaces the
        # single-width ActionModel the base constructor built).
        self.action_model = ActionModel(
            self.encoding_size * 2,
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
            deterministic=network_settings.deterministic,
        )

    def _split_inputs(
        self, inputs: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        if self.comm_idx == -1:
            return inputs, None
        normal_inputs = [inp for i, inp in enumerate(inputs) if i != self.comm_idx]
        comm_buffer = inputs[self.comm_idx]
        return normal_inputs, comm_buffer

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:

        normal_inputs, comm_buffer = self._split_inputs(inputs)

        # 1. The agent's internal state (query source)
        encoding, memories = self.network_body(
            normal_inputs, memories=memories, sequence_length=sequence_length
        )

        # 2. TarMAC attention over incoming messages
        if comm_buffer is not None:
            aggregated_msg, attention_weights = self.tarmac_block(
                encoding, comm_buffer
            )
        else:
            aggregated_msg = torch.zeros_like(encoding)
            attention_weights = None

        # 3. Condition the action heads on [encoding | aggregated message]
        final_encoding = torch.cat([encoding, aggregated_msg], dim=-1)
        action, log_probs, entropies = self.action_model(final_encoding, masks)

        run_out = {}
        run_out["env_action"] = action.to_action_tuple(
            clip=self.action_model.clip_action
        )
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies

        # --- VISUALIZATION (SAFE-COPY) ---
        # Smuggle the attention weights to Unity in the continuous slots after the
        # message ([COMM_MSG_SIZE:]) so the C# gizmos can color the comm lines.
        # CRITICAL: never write into the env_action numpy array in place --
        # to_action_tuple(clip=False) returns a VIEW sharing memory with
        # action.continuous_tensor (stored in the training buffer), so an in-place
        # write would corrupt the on-policy training data whenever clip is off.
        # We build a fresh copy for Unity; the stored tensor keeps the true samples.
        if (
            attention_weights is not None
            and run_out["env_action"].continuous is not None
        ):
            available_slots = (
                run_out["env_action"].continuous.shape[1] - COMM_MSG_SIZE
            )
            if available_slots > 0:
                numpy_weights = attention_weights.squeeze(1).detach().cpu().numpy()
                safe_continuous = run_out["env_action"].continuous.copy()
                safe_continuous[
                    :, COMM_MSG_SIZE : COMM_MSG_SIZE + available_slots
                ] = numpy_weights[:, :available_slots]
                run_out["env_action"].add_continuous(safe_continuous)
        # ----------------------------------

        return action, run_out, memories

    def get_stats(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:

        normal_inputs, comm_buffer = self._split_inputs(inputs)

        encoding, actor_mem_outs = self.network_body(
            normal_inputs, memories=memories, sequence_length=sequence_length
        )

        if comm_buffer is not None:
            aggregated_msg, _ = self.tarmac_block(encoding, comm_buffer)
        else:
            aggregated_msg = torch.zeros_like(encoding)

        final_encoding = torch.cat([encoding, aggregated_msg], dim=-1)
        log_probs, entropies = self.action_model.evaluate(
            final_encoding, masks, actions
        )
        run_out = {}
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies
        return run_out

    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        """ONNX export path -- same output contract as SimpleActor.forward."""

        normal_inputs, comm_buffer = self._split_inputs(inputs)

        encoding, memories_out = self.network_body(
            normal_inputs, memories=memories, sequence_length=1
        )

        if comm_buffer is not None:
            aggregated_msg, _ = self.tarmac_block(encoding, comm_buffer)
        else:
            aggregated_msg = torch.zeros_like(encoding)

        final_encoding = torch.cat([encoding, aggregated_msg], dim=-1)

        (
            cont_action_out,
            disc_action_out,
            action_out_deprecated,
            deterministic_cont_action_out,
            deterministic_disc_action_out,
        ) = self.action_model.get_action_out(final_encoding, masks)

        export_out = [self.version_number, self.memory_size_vector]
        if self.action_spec.continuous_size > 0:
            export_out += [
                cont_action_out,
                self.continuous_act_size_vector,
                deterministic_cont_action_out,
            ]
        if self.action_spec.discrete_size > 0:
            export_out += [
                disc_action_out,
                self.discrete_act_size_vector,
                deterministic_disc_action_out,
            ]
        if self.network_body.memory_size > 0:
            export_out += [memories_out]
        return tuple(export_out)
