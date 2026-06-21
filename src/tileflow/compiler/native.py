"""Native MLIR pass pipeline integration.

The Python frontend is intentionally thin. It emits MLIR text, then delegates
verification and optimization to native MLIR.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


DEFAULT_PIPELINE = "canonicalize,cse"


@dataclass(frozen=True)
class NativePipelineResult:
    mlir: str
    backend: str
    pipeline: str
    message: str = ""


def run_native_pipeline(
    mlir: str,
    *,
    pipeline: str | None = None,
) -> NativePipelineResult:
    """Run native MLIR passes.

    Resolution order:

    1. `tileflow._mlir.tileflow_mlir.PassPipeline` Python extension.
    2. `tileflow-opt` executable.

    Raises when neither native path is available.
    """

    pipeline = pipeline or os.environ.get("TILEFLOW_MLIR_PIPELINE") or DEFAULT_PIPELINE

    extension_result = _run_extension(mlir, pipeline)
    if extension_result is not None:
        return extension_result

    tool_result = _run_tool(mlir, pipeline)
    if tool_result is not None:
        return tool_result

    raise RuntimeError(
        "native MLIR pipeline unavailable: install/build tileflow_mlir or put tileflow-opt on PATH"
    )


def _run_extension(mlir: str, pipeline: str) -> NativePipelineResult | None:
    try:
        from tileflow._mlir.tileflow_mlir import PassPipeline  # type: ignore[import-not-found]
    except Exception:
        return None

    pass_pipeline = PassPipeline()
    pass_pipeline.add(pipeline)
    return NativePipelineResult(
        mlir=pass_pipeline.run(mlir),
        backend="tileflow_mlir",
        pipeline=pipeline,
    )


def _run_tool(mlir: str, pipeline: str) -> NativePipelineResult | None:
    tool = os.environ.get("TILEFLOW_OPT") or shutil.which("tileflow-opt")
    if not tool:
        return None
    command = [tool, f"--pass-pipeline=builtin.module({pipeline})"]
    result = subprocess.run(command, input=mlir, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "tileflow-opt failed with exit code "
            f"{result.returncode}\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )
    return NativePipelineResult(
        mlir=result.stdout,
        backend="tileflow-opt",
        pipeline=pipeline,
        message=result.stderr,
    )
