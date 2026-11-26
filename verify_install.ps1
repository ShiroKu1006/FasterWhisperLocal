$ErrorActionPreference = 'Stop'
if (-not $env:CONDA_PREFIX -or $env:CONDA_DEFAULT_ENV -ne 'faster-whisper') {
    throw 'Activate the project Conda environment: conda activate faster-whisper'
}
$pythonExe = Join-Path $env:CONDA_PREFIX 'python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$runtimeDirectory = Join-Path $PSScriptRoot 'runtime\cuda12'
$modelDirectory = Join-Path $PSScriptRoot 'models\faster-whisper-large-v3'

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw 'Python is missing from the faster-whisper Conda environment. Check environment.yml.'
}

$diagnostic = @'
import sys
from importlib.metadata import version
from pathlib import Path
from faster_whisper_local.runtime import register_cuda_runtime

runtime = Path(sys.argv[1])
libraries = register_cuda_runtime(runtime)
import ctranslate2

print('Python packages:')
for package in ('faster-whisper', 'ctranslate2', 'OpenCC'):
    print(f'  {package}={version(package)}')
print(f'CUDA devices: {ctranslate2.get_cuda_device_count()}')
print(f'Private CUDA DLLs: {len(libraries)} required libraries found')
model = Path(sys.argv[2])
model_status = 'present' if model.is_dir() else 'not installed'
print(f'large-v3 model: {model_status}')
'@
& $pythonExe -c $diagnostic $runtimeDirectory $modelDirectory
exit $LASTEXITCODE
