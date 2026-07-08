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
from typing import Any, Dict, Type, Union

from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.poca.trainer import POCATrainer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.torch_entities.networks import SharedActorCritic
from mlagents.trainers.comm_mapoca.comm_networks import CommSimpleActor
from mlagents.trainers.comm_mapoca.optimizer_torch import TorchCommMAPOCAOptimizer
from mlagents_envs.base_env import BehaviorSpec

TRAINER_NAME = "comm_mapoca"


class CommMAPOCATrainer(POCATrainer):
    """POCATrainer with the Comm-MAPOCA actor and message-conditioned critic."""

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
