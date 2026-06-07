"""Observation preprocessing for low-dimensional control environments."""

from __future__ import annotations

import math

import numpy as np


_CARTPOLE_SCALE = np.array([4.8, 5.0, 0.418, 5.0], dtype=np.float32)


def _cartpole_base_normalized(observation: np.ndarray) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    scaled = obs / _CARTPOLE_SCALE
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def _cartpole_fuzzy_compact(observation: np.ndarray) -> np.ndarray:
    """Compact heuristic/fuzzy feature map for CartPole.

    The last two features encode domain knowledge: a stability membership score
    and a signed recommendation score similar to a fuzzy rule controller.
    """
    position, velocity, angle, angular_velocity = _cartpole_base_normalized(observation)
    stability = 1.0 - abs(angle)
    stability_signed = 2.0 * stability - 1.0
    recommendation = np.tanh(2.0 * angle + 0.5 * angular_velocity + 0.05 * position + 0.05 * velocity)
    features = np.asarray(
        [angle, angular_velocity, stability_signed, recommendation],
        dtype=np.float32,
    )
    return np.clip(features, -1.0, 1.0) * math.pi


def _cartpole_fuzzy_membership_4(observation: np.ndarray) -> np.ndarray:
    """Four descriptive fuzzy memberships derived from CartPole limits.

    These features describe state risk without encoding a recommended action:
    pole falling left/right and cart escaping left/right.
    """
    x, velocity, theta, angular_velocity = np.asarray(observation, dtype=np.float32)
    pole_limit = 0.2095
    cart_limit = 2.4
    velocity_scale = 2.0
    angular_velocity_scale = 2.0

    pole_left = np.clip((-theta / pole_limit) + (-angular_velocity / angular_velocity_scale), 0.0, 1.0)
    pole_right = np.clip((theta / pole_limit) + (angular_velocity / angular_velocity_scale), 0.0, 1.0)
    cart_left = np.clip((-x / cart_limit) + (-velocity / velocity_scale), 0.0, 1.0)
    cart_right = np.clip((x / cart_limit) + (velocity / velocity_scale), 0.0, 1.0)

    features = np.asarray([pole_left, pole_right, cart_left, cart_right], dtype=np.float32)
    return features * math.pi


def preprocess_observation(
    observation: np.ndarray,
    env_id: str,
    feature_mode: str = "raw_normalized",
) -> np.ndarray:
    """Normalize observations to a compact numeric range.

    CartPole has infinite velocity bounds, so fixed practical scales are used
    for velocity components before clipping to [-pi, pi].
    """
    obs = np.asarray(observation, dtype=np.float32)

    if env_id == "CartPole-v1":
        if feature_mode == "raw_normalized":
            return _cartpole_base_normalized(obs) * math.pi
        if feature_mode == "fuzzy_compact":
            return _cartpole_fuzzy_compact(obs)
        if feature_mode == "fuzzy_membership_4":
            return _cartpole_fuzzy_membership_4(obs)
        raise ValueError(f"Unknown feature_mode for CartPole-v1: {feature_mode}")

    return obs.astype(np.float32)
