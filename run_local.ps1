param(
    [ValidateSet("check", "trace", "smoke", "train", "build-v2", "train-v2", "eval-v2-validation", "eval-v2", "eval-v2-challenge", "pipeline-v2", "eval-base", "eval-qlora", "eval-challenge", "analyze", "benchmark", "demo")]
    [string]$Mode = "check"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\anaconda\envs\gemma-workorder\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "gemma-workorder environment not found: $Python"
}

Set-Location -LiteralPath $ProjectRoot

function Invoke-ProjectPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code: $LASTEXITCODE"
    }
}

switch ($Mode) {
    "check" {
        Invoke-ProjectPython -Arguments @(
            "-c",
            "import torch; print('Python environment ready'); print('torch =', torch.__version__); print('CUDA =', torch.cuda.is_available()); print('GPU =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
        )
        Invoke-ProjectPython -Arguments @("-m", "pytest", "-q")
    }
    "trace" {
        Invoke-ProjectPython -Arguments @(
            "scripts/export_agent_traces.py",
            "--input", "data/tool_use/train.jsonl",
            "--output", "artifacts/traces/train.jsonl"
        )
    }
    "smoke" {
        Invoke-ProjectPython -Arguments @(
            "scripts/train_tool_use_qlora.py",
            "--max-train-samples", "32",
            "--max-validation-samples", "32",
            "--epochs", "1",
            "--warmup-steps", "1",
            "--output-dir", "artifacts/tool_use_smoke_adapter"
        )
    }
    "train" {
        Invoke-ProjectPython -Arguments @(
            "scripts/train_tool_use_qlora.py",
            "--output-dir", "artifacts/tool_use_qlora_adapter"
        )
    }
    "build-v2" {
        Invoke-ProjectPython -Arguments @("scripts/build_v2_curriculum.py")
    }
    "train-v2" {
        if (-not (Test-Path -LiteralPath "data/tool_use/train_v2.jsonl")) {
            Invoke-ProjectPython -Arguments @("scripts/build_v2_curriculum.py")
        }
        Invoke-ProjectPython -Arguments @(
            "scripts/train_tool_use_qlora.py",
            "--train", "data/tool_use/train_v2.jsonl",
            "--output-dir", "artifacts/tool_use_qlora_v2_adapter",
            "--run-name", "tool-use-qlora-v2",
            "--epochs", "1.5",
            "--learning-rate", "0.0001",
            "--eval-steps", "50",
            "--save-steps", "50",
            "--save-total-limit", "3"
        )
    }
    "eval-v2-validation" {
        if (-not (Test-Path -LiteralPath "artifacts/tool_use_qlora_v2_adapter")) {
            throw "V2 adapter not found. Run: .\run_local.ps1 -Mode train-v2"
        }
        Invoke-ProjectPython -Arguments @(
            "scripts/evaluate_tool_use.py",
            "--adapter", "artifacts/tool_use_qlora_v2_adapter",
            "--precision", "4bit",
            "--dataset", "data/tool_use/validation.jsonl",
            "--output", "reports/tool_use_qlora_v2_validation_4bit.json"
        )
    }
    "eval-v2" {
        if (-not (Test-Path -LiteralPath "artifacts/tool_use_qlora_v2_adapter")) {
            throw "V2 adapter not found. Run: .\run_local.ps1 -Mode train-v2"
        }
        Invoke-ProjectPython -Arguments @(
            "scripts/evaluate_tool_use.py",
            "--adapter", "artifacts/tool_use_qlora_v2_adapter",
            "--precision", "4bit",
            "--output", "reports/tool_use_qlora_v2_4bit.json"
        )
    }
    "eval-v2-challenge" {
        if (-not (Test-Path -LiteralPath "artifacts/tool_use_qlora_v2_adapter")) {
            throw "V2 adapter not found. Run: .\run_local.ps1 -Mode train-v2"
        }
        Invoke-ProjectPython -Arguments @(
            "scripts/evaluate_tool_use.py",
            "--adapter", "artifacts/tool_use_qlora_v2_adapter",
            "--precision", "4bit",
            "--dataset", "data/tool_use/challenge.jsonl",
            "--output", "reports/tool_use_qlora_v2_challenge_4bit.json"
        )
    }
    "pipeline-v2" {
        $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $LogDirectory = Join-Path $ProjectRoot "logs"
        $LogPath = Join-Path $LogDirectory "v2_pipeline_$Timestamp.log"
        $AdapterDirectory = Join-Path $ProjectRoot "artifacts\tool_use_qlora_v2_adapter"
        New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
        Start-Transcript -Path $LogPath
        try {
            Write-Host "[1/6] Building V2 training curriculum"
            Invoke-ProjectPython -Arguments @("scripts/build_v2_curriculum.py")

            $FinalAdapter = Join-Path $AdapterDirectory "adapter_config.json"
            if (Test-Path -LiteralPath $FinalAdapter) {
                Write-Host "[2/6] Completed V2 adapter found; skipping training"
            }
            else {
                $TrainingArguments = @(
                    "scripts/train_tool_use_qlora.py",
                    "--train", "data/tool_use/train_v2.jsonl",
                    "--output-dir", "artifacts/tool_use_qlora_v2_adapter",
                    "--run-name", "tool-use-qlora-v2",
                    "--epochs", "1.5",
                    "--learning-rate", "0.0001",
                    "--eval-steps", "50",
                    "--save-steps", "50",
                    "--save-total-limit", "3"
                )
                $LatestCheckpoint = Get-ChildItem -LiteralPath $AdapterDirectory -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue |
                    Sort-Object { [int]($_.Name -replace "checkpoint-", "") } |
                    Select-Object -Last 1
                if ($null -ne $LatestCheckpoint) {
                    Write-Host "[2/6] Resuming from checkpoint: $($LatestCheckpoint.FullName)"
                    $TrainingArguments += @("--resume-from-checkpoint", $LatestCheckpoint.FullName)
                }
                else {
                    Write-Host "[2/6] Starting V2 training from the base model"
                }
                Invoke-ProjectPython -Arguments $TrainingArguments
            }

            Write-Host "[3/6] Evaluating validation split"
            Invoke-ProjectPython -Arguments @(
                "scripts/evaluate_tool_use.py",
                "--adapter", "artifacts/tool_use_qlora_v2_adapter",
                "--precision", "4bit",
                "--dataset", "data/tool_use/validation.jsonl",
                "--output", "reports/tool_use_qlora_v2_validation_4bit.json"
            )

            Write-Host "[4/6] Evaluating fixed test split"
            Invoke-ProjectPython -Arguments @(
                "scripts/evaluate_tool_use.py",
                "--adapter", "artifacts/tool_use_qlora_v2_adapter",
                "--precision", "4bit",
                "--output", "reports/tool_use_qlora_v2_4bit.json"
            )

            Write-Host "[5/6] Evaluating challenge split"
            Invoke-ProjectPython -Arguments @(
                "scripts/evaluate_tool_use.py",
                "--adapter", "artifacts/tool_use_qlora_v2_adapter",
                "--precision", "4bit",
                "--dataset", "data/tool_use/challenge.jsonl",
                "--output", "reports/tool_use_qlora_v2_challenge_4bit.json"
            )

            Write-Host "[6/6] Saving run summary"
            Invoke-ProjectPython -Arguments @("scripts/summarize_v2_run.py")
            Write-Host "V2 pipeline completed. Log: $LogPath"
        }
        finally {
            Stop-Transcript
        }
    }
    "eval-base" {
        Invoke-ProjectPython -Arguments @(
            "scripts/evaluate_tool_use.py",
            "--precision", "4bit",
            "--output", "reports/tool_use_base_4bit.json"
        )
    }
    "eval-qlora" {
        if (-not (Test-Path -LiteralPath "artifacts/tool_use_qlora_adapter")) {
            throw "Adapter not found. Run: .\run_local.ps1 -Mode train"
        }
        Invoke-ProjectPython -Arguments @(
            "scripts/evaluate_tool_use.py",
            "--adapter", "artifacts/tool_use_qlora_adapter",
            "--precision", "4bit",
            "--output", "reports/tool_use_qlora_4bit.json"
        )
    }
    "eval-challenge" {
        if (-not (Test-Path -LiteralPath "artifacts/tool_use_qlora_adapter")) {
            throw "Adapter not found. Run: .\run_local.ps1 -Mode train"
        }
        Invoke-ProjectPython -Arguments @(
            "scripts/evaluate_tool_use.py",
            "--adapter", "artifacts/tool_use_qlora_adapter",
            "--precision", "4bit",
            "--dataset", "data/tool_use/challenge.jsonl",
            "--output", "reports/tool_use_qlora_challenge_4bit.json"
        )
    }
    "analyze" {
        Invoke-ProjectPython -Arguments @("scripts/analyze_tool_use_results.py")
    }
    "benchmark" {
        Invoke-ProjectPython -Arguments @("scripts/run_agent_benchmark.py")
    }
    "demo" {
        if (-not (Test-Path -LiteralPath "artifacts/tool_use_qlora_adapter")) {
            throw "Adapter not found. Run: .\run_local.ps1 -Mode train"
        }
        Invoke-ProjectPython -Arguments @(
            "scripts/run_tool_use_demo.py",
            "--adapter", "artifacts/tool_use_qlora_adapter",
            "--index", "0"
        )
    }
}
