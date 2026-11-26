import os
from pathlib import Path

import pytest

from faster_whisper_local import runtime
from faster_whisper_local.runtime import validate_cuda_runtime


def test_cuda_runtime_validation_reports_all_missing_library_groups(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError) as error:
        validate_cuda_runtime(tmp_path)

    message = str(error.value)
    assert "cublas64_12.dll" in message
    assert "cublasLt64_12.dll" in message
    assert "cudnn64_9.dll" in message
    assert "cudnn_ops64_9.dll" in message


def test_register_cuda_runtime_prepends_process_path_once_and_keeps_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_directory = tmp_path / "cuda12"
    runtime_directory.mkdir()
    for filename in runtime._REQUIRED_CUDA_DLLS:
        (runtime_directory / filename).touch()

    existing_directory = tmp_path / "existing"
    runtime_path = str(runtime_directory.resolve())
    monkeypatch.setenv("PATH", f"{existing_directory}{os.pathsep}{runtime_path}")
    handles: list[object] = []
    add_dll_directory_calls: list[str] = []
    monkeypatch.setattr(runtime, "_DLL_DIRECTORY_HANDLES", handles)
    monkeypatch.setattr(runtime, "_REGISTERED_DLL_DIRECTORIES", set())
    monkeypatch.setattr(
        runtime.os,
        "add_dll_directory",
        lambda directory: add_dll_directory_calls.append(directory) or object(),
    )

    runtime.register_cuda_runtime(runtime_directory)
    runtime.register_cuda_runtime(runtime_directory)

    path_entries = os.environ["PATH"].split(os.pathsep)
    assert path_entries[0] == runtime_path
    assert path_entries.count(runtime_path) == 1
    assert path_entries[1:] == [str(existing_directory)]
    assert add_dll_directory_calls == [runtime_path]
    assert len(handles) == 1
