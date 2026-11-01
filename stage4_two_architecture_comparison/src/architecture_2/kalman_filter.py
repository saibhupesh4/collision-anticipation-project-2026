"""Timestamp-aware constant-velocity Kalman filter in normalized image space."""

from __future__ import annotations

import numpy as np


class ConstantVelocityKalman:
    def __init__(self, measurement_noise: float = 0.01, acceleration_noise: float = 0.15):
        self.x = np.zeros(4, dtype=float)  # x, y, vx, vy; normalized image units
        self.P = np.diag([0.05, 0.05, 0.25, 0.25]).astype(float)
        self.R = np.eye(2, dtype=float) * measurement_noise**2
        self.acceleration_noise = float(acceleration_noise)
        self.initialized = False

    def initialize(self, x: float, y: float) -> None:
        self.x[:] = [float(x), float(y), 0.0, 0.0]
        self.P = np.diag([0.02, 0.02, 0.5, 0.5]).astype(float)
        self.initialized = True

    def predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if not self.initialized:
            raise RuntimeError("Filter must be initialized before prediction")
        dt = max(float(dt), 1e-6)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        q = self.acceleration_noise**2
        Q = q * np.array([
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.copy(), self.P.copy()

    def update(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        if not self.initialized:
            self.initialize(x, y)
            return self.x.copy(), self.P.copy()
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        z = np.array([float(x), float(y)])
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        identity = np.eye(4)
        self.P = (identity - K @ H) @ self.P @ (identity - K @ H).T + K @ self.R @ K.T
        return self.x.copy(), self.P.copy()
