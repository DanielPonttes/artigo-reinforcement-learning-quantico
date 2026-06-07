"""Variational quantum circuit Q-network."""

from __future__ import annotations

import torch
from torch import nn


class VQCQNetwork(nn.Module):
    """Hybrid VQC Q-network with Pauli-Z readout, batching, and optional dueling."""

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        n_qubits: int,
        n_layers: int,
        backend: str = "default.qubit",
        data_reuploading: bool = True,
        dueling: bool = False,
    ):
        super().__init__()
        if input_dim > n_qubits:
            raise ValueError(
                f"input_dim={input_dim} cannot exceed n_qubits={n_qubits} for angle encoding"
            )

        self.input_dim = int(input_dim)
        self.n_actions = int(n_actions)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.backend = backend
        self.data_reuploading = bool(data_reuploading)
        self.dueling = bool(dueling)

        weight_shape = (self.n_layers, self.n_qubits, 3)
        self.weights = nn.Parameter(0.01 * torch.randn(weight_shape, dtype=torch.float32))

        if self.dueling:
            self.value_head = nn.Linear(self.n_qubits, 1)
            self.advantage_head = nn.Linear(self.n_qubits, self.n_actions)
        else:
            self.readout = nn.Linear(self.n_qubits, self.n_actions)

        self._qnode = self._build_qnode()

    def _build_qnode(self):
        import pennylane as qml

        dev = qml.device(self.backend, wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            for layer_idx in range(self.n_layers):
                if layer_idx == 0 or self.data_reuploading:
                    for wire in range(self.n_qubits):
                        qml.RY(inputs[:, wire], wires=wire)

                for wire in range(self.n_qubits):
                    qml.Rot(
                        weights[layer_idx, wire, 0],
                        weights[layer_idx, wire, 1],
                        weights[layer_idx, wire, 2],
                        wires=wire,
                    )

                for wire in range(self.n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])

            return [qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits)]

        return circuit

    def _quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        padded = torch.zeros(x.shape[0], self.n_qubits, dtype=x.dtype, device=x.device)
        padded[:, : self.input_dim] = x
        outputs = self._qnode(padded, self.weights)
        if isinstance(outputs, (tuple, list)):
            return torch.stack([o.float() for o in outputs], dim=-1)
        return outputs.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        quantum_features = self._quantum_features(x.float())

        if self.dueling:
            value = self.value_head(quantum_features)
            advantages = self.advantage_head(quantum_features)
            return value + advantages - advantages.mean(dim=1, keepdim=True)

        return self.readout(quantum_features)

