"""M9 guardrail ablation conditions with a fixed three-agent topology."""

from __future__ import annotations

from enum import StrEnum


class ExperimentCondition(StrEnum):
    """Supported combinations of the three independently controlled layers."""

    BASELINE = "BASELINE"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L1_L2 = "L1_L2"
    L1_L3 = "L1_L3"
    L2_L3 = "L2_L3"
    FULL = "FULL"

    @property
    def active_layers(self) -> frozenset[int]:
        """Return the enabled role-consistency layers for this condition."""

        if self is ExperimentCondition.BASELINE:
            return frozenset()
        if self is ExperimentCondition.FULL:
            return frozenset({1, 2, 3})
        return frozenset(
            int(part.removeprefix("L")) for part in self.value.split("_")
        )

    def layer_enabled(self, layer: int) -> bool:
        """Report whether one of Layer 1, 2, or 3 is active."""

        if layer not in {1, 2, 3}:
            raise ValueError(f"Unsupported guardrail layer: {layer}")
        return layer in self.active_layers

    @property
    def layer_flags(self) -> dict[str, bool]:
        """Return configuration-shaped flags without importing config models."""

        return {
            "layer1": self.layer_enabled(1),
            "layer2": self.layer_enabled(2),
            "layer3": self.layer_enabled(3),
        }
