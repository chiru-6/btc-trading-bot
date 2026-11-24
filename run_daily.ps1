# Local (Windows) version of the daily bot.
# Fetches new candles, retrains weekly, paper-trades the newest candle.
#
# Run once:   powershell -ExecutionPolicy Bypass -File .\run_daily.ps1
# Schedule:   schtasks /Create /SC DAILY /ST 18:10 /TN "btcbot" /TR "powershell -ExecutionPolicy Bypass -File <full path>\run_daily.ps1"
#             (18:10 Central = just after the 00:00 UTC daily candle close)

Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path "paper" | Out-Null
Start-Transcript -Path "paper\run_log.txt" -Append | Out-Null

& .\.venv\Scripts\Activate.ps1

btcbot fetch --days 3650 --out data/btc_1d.csv

$model = "models/model.joblib"
$stale = (-not (Test-Path $model)) -or (((Get-Date) - (Get-Item $model).LastWriteTime).TotalDays -ge 7)
if ($stale) {
    Write-Output "Retraining model..."
    btcbot train --data data/btc_1d.csv --model gbm --name ("auto-" + (Get-Date -Format yyyyMMdd))
}

btcbot paper --data data/btc_1d.csv
btcbot live-report --data data/btc_1d.csv

Stop-Transcript | Out-Null
