$noOutputLimit = 5000  # seconds

while ($true) {
    $outputFile = "temp_output.txt"
    if (Test-Path $outputFile) { Remove-Item $outputFile }

    $process = Start-Process python `
        -ArgumentList "-u", "continuous_adaptive_metaheuristics.py --run-root runs/de_evolution --seed-provider anthropic --variant-provider llamacpp --adaptive-provider anthropic --component-selector llm --selector-provider anthropic --n-seeds 2" `
        -RedirectStandardOutput $outputFile `
        -PassThru

    $lastChange = Get-Date

    while (-not $process.HasExited) {
        if (Test-Path $outputFile) {
            $currentWrite = (Get-Item $outputFile).LastWriteTime
            if ($currentWrite -ne $lastChange) {
                $lastChange = $currentWrite
            }
        }

        if (((Get-Date) - $lastChange).TotalSeconds -ge $noOutputLimit) {
            taskkill /F /T /PID $process.Id | Out-Null
            break
        }

        Start-Sleep -Milliseconds 500
    }

    Start-Sleep -Seconds 2
}