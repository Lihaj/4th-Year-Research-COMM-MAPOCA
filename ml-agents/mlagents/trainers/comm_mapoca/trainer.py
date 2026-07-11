"""
Comm-MAPOCA trainer (Lihaj Wickramasinghe, University of Moratuwa).

MA-POCA (Cohen et al., 2022) extended with TarMAC-style targeted communication
(Das et al., 2020): actors exchange learned messages through a CommBuffer sensor
and attend over them; the centralized critic (value + counterfactual baseline)
is conditioned on the communicated messages.

Select with `trainer_type: comm_mapoca` in the behavior config. Hyperparameters
are identical to poca (POCASettings). The stock poca trainer remains untouched
and available for baselines.
"""
from collections import defaultdict
from typing import Any, Dict, Type, Union

from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.poca.trainer import POCATrainer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.torch_entities.networks import SharedActorCritic
from mlagents.trainers.comm_mapoca.comm_networks import CommSimpleActor
from mlagents.trainers.comm_mapoca.optimizer_torch import TorchCommMAPOCAOptimizer
from mlagents.trainers.trajectory import Trajectory
from mlagents_envs.base_env import BehaviorSpec

TRAINER_NAME = "comm_mapoca"


class CommMAPOCATrainer(POCATrainer):
    """POCATrainer with the Comm-MAPOCA actor and message-conditioned critic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Episodic GROUP reward accumulator for the curriculum fix below.
        self._group_reward_acc: Dict[str, float] = defaultdict(float)

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Curriculum fix: stock rl_trainer fills reward_buffer (the signal that
        lesson completion_criteria with measure 'reward' compare against) with
        INDIVIDUAL environment rewards, which stay ~0 in pure group-reward
        cooperative tasks -- so lessons never advance no matter how well the
        team performs. Mirror POCA's episodic GROUP reward into reward_buffer
        so curricula key on what the trainer actually optimizes.
        """
        agent_id = trajectory.agent_id
        self._group_reward_acc[agent_id] += float(
            sum(step.group_reward for step in trajectory.steps)
        )
        super()._process_trajectory(trajectory)
        if trajectory.done_reached and trajectory.all_group_dones_reached:
            self.reward_buffer.appendleft(self._group_reward_acc.pop(agent_id, 0.0))
        elif trajectory.done_reached:
            # Mirrors POCA: agent finished but the group episode did not -- the
            # partial sum is discarded, matching collected_group_rewards handling.
            self._group_reward_acc.pop(agent_id, None)

    def end_episode(self) -> None:
        super().end_episode()
        self._group_reward_acc.clear()

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        actor_cls: Union[
            Type[CommSimpleActor], Type[SharedActorCritic]
        ] = CommSimpleActor
        actor_kwargs: Dict[str, Any] = {
            "conditional_sigma": False,
            "tanh_squash": False,
        }

        policy = TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
        )
        return policy

    def create_optimizer(self) -> TorchCommMAPOCAOptimizer:
        return TorchCommMAPOCAOptimizer(self.policy, self.trainer_settings)

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
