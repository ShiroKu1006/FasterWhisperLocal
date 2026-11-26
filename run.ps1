param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $CliArguments
)

$ErrorActionPreference = 'Stop'
if (-not $env:CONDA_PREFIX -or $env:CONDA_DEFAULT_ENV -ne 'faster-whisper') {
    throw 'Activate the project Conda environment: conda activate faster-whisper'
}
$pythonExe = Join-Path $env:CONDA_PREFIX 'python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw 'Python is missing from the faster-whisper Conda environment. Check environment.yml.'
}

& $pythonExe -m faster_whisper_local.cli @CliArguments
exit $LASTEXITCODE
