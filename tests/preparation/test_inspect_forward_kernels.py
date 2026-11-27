# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Forward kernel inspection reads saved six-field StorageBackend artifacts on CPU."""

import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from ignition.preparation import aot


class _CpuForward:
    def __init__(self):
        self.value = torch.ones(4, dtype=torch.float32)

    def __call__(self):
        return self.value + 1.0


def test_inspect_forward_kernels_reads_saved_aot_artifact(tmp_path):
    compiled = aot.aot_forward(_CpuForward(), options={"compile_threads": 1})
    compiled._artifacts.compiled_fn._cuda_allocation_sizes = (4096, 1024)
    artifact_path = tmp_path / "forward.bin"
    compiled.save_compiled_function(str(artifact_path))

    config_path = tmp_path / "config.toml"
    config_path.write_text("")
    script = Path(__file__).resolve().parents[2] / "scripts" / "inspect_forward_kernels.py"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path), str(artifact_path)],
        capture_output=True, text=True, env=env, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip())
    assert report["artifact"] == str(artifact_path)
    assert isinstance(report["graph_key"], str) and report["graph_key"]
    assert report["frozen"] is False
    assert report["missing_static"] == []
    assert report["unused_static"] == []
