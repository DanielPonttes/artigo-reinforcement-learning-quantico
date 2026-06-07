#!/usr/bin/env python3
"""Draw the VQC ansatz used in the experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pennylane as qml
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cartpole_vqc.yaml")
    parser.add_argument("--output", default="outputs/figures/vqc_circuit.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_config = dict(config.get("model", {}))
    n_qubits = int(model_config.get("n_qubits", 4))
    n_layers = int(model_config.get("n_layers", 1))
    input_dim = 4
    data_reuploading = bool(model_config.get("data_reuploading", True))

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        padded_inputs = torch.zeros(n_qubits)
        padded_inputs[:input_dim] = inputs[:input_dim]

        for layer_idx in range(n_layers):
            if layer_idx == 0 or data_reuploading:
                for wire in range(n_qubits):
                    qml.RY(padded_inputs[wire], wires=wire)
            for wire in range(n_qubits):
                qml.Rot(
                    weights[layer_idx, wire, 0],
                    weights[layer_idx, wire, 1],
                    weights[layer_idx, wire, 2],
                    wires=wire,
                )
            for wire in range(n_qubits):
                qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    inputs = torch.zeros(input_dim)
    weights = torch.zeros(n_layers, n_qubits, 3)
    fig, _ = qml.draw_mpl(circuit, decimals=2)(inputs, weights)
    fig.set_size_inches(9.0, 2.8)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"circuit_figure={output_path}")


if __name__ == "__main__":
    main()

