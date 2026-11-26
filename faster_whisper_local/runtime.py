from __future__ import annotations

import os
from pathlib import Path


_REQUIRED_CUDA_DLLS = (
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
    "cudnn_ops64_9.dll",
)
_DLL_DIRECTORY_HANDLES: list[object] = []
_REGISTERED_DLL_DIRECTORIES: set[Path] = set()


def validate_cuda_runtime(runtime_directory: Path) -> tuple[Path, ...]:
    found: list[Path] = []
    missing: list[str] = []
    for filename in _REQUIRED_CUDA_DLLS:
        matches = tuple(runtime_directory.rglob(filename)) if runtime_directory.exists() else ()
        if matches:
            found.extend(matches)
        else:
            missing.append(filename)
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"CUDA runtime 不完整，目錄 {runtime_directory} 缺少：{names}"
        )
    return tuple(found)


def register_cuda_runtime(runtime_directory: Path) -> tuple[Path, ...]:
    libraries = validate_cuda_runtime(runtime_directory)
    dll_directories = {library.parent.resolve() for library in libraries}
    if not hasattr(os, "add_dll_directory"):
        raise RuntimeError("此作業系統不支援私有 Windows DLL 搜尋路徑")
    ordered_directories = tuple(sorted(dll_directories, key=str))
    directory_keys = {
        os.path.normcase(os.path.normpath(str(directory)))
        for directory in ordered_directories
    }
    existing_path_entries = os.environ.get("PATH", "").split(os.pathsep)
    remaining_path_entries = [
        entry
        for entry in existing_path_entries
        if os.path.normcase(os.path.normpath(entry)) not in directory_keys
    ]
    os.environ["PATH"] = os.pathsep.join(
        [*(str(directory) for directory in ordered_directories), *remaining_path_entries]
    )
    for directory in ordered_directories:
        if directory not in _REGISTERED_DLL_DIRECTORIES:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
            _REGISTERED_DLL_DIRECTORIES.add(directory)
    return libraries
