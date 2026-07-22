"""Export the verified production bundle—not a mock architecture—to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.core.artifacts import load_bundle


def export_bundle_to_onnx(
    bundle_path: str | Path,
    output_path: str | Path,
    opset_version: int = 17,
) -> Path:
    processor, model, _manifest = load_bundle(bundle_path)
    model.eval()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    example = torch.ones((1, processor.max_length), dtype=torch.long)
    torch.onnx.export(
        model,
        example,
        destination,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
        input_names=["item_sequence"],
        output_names=["logits"],
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="models/current")
    parser.add_argument("--output", default="models/deep_sequence_rec.onnx")
    args = parser.parse_args()
    print(export_bundle_to_onnx(args.bundle, args.output))


if __name__ == "__main__":
    main()
