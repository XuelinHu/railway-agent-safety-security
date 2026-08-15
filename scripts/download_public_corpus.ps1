param(
    [int]$MaxReports = 0,
    [int]$DelayMilliseconds = 250
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$repoRoot = Split-Path -Parent $PSScriptRoot
$rawDir = Join-Path $repoRoot "data\raw\raib"
$catalogDir = Join-Path $repoRoot "data\catalog"
$null = New-Item -ItemType Directory -Force -Path $rawDir, $catalogDir

$headers = @{ "User-Agent" = "railway-agent-safety-security-research/0.1 (public research corpus; contact via repository)" }
$listUrl = "https://www.gov.uk/raib-reports"
$reportUrls = [System.Collections.Generic.HashSet[string]]::new()

function Get-RemoteText([string]$Url) {
    (Invoke-WebRequest -Uri $Url -Headers $headers -UseBasicParsing).Content
}

Write-Host "Collecting public RAIB report pages..."
for ($page = 1; $page -le 20; $page++) {
    $url = if ($page -eq 1) { $listUrl } else { "${listUrl}?page=$page" }
    try {
        $html = Get-RemoteText $url
        $matches = [regex]::Matches($html, 'href="(/raib-reports/[^"]+)"')
        foreach ($match in $matches) {
            $path = $match.Groups[1].Value
            if ($path -notmatch '/raib-reports\?$' -and $path -notmatch '/raib-reports/page' -and $path -notmatch '/email-signup') {
                $null = $reportUrls.Add("https://www.gov.uk$path")
            }
        }
    } catch {
        Write-Warning "Unable to read list page $page`: $($_.Exception.Message)"
    }
    Start-Sleep -Milliseconds $DelayMilliseconds
}

$orderedUrls = @($reportUrls | Sort-Object)
if ($MaxReports -gt 0) {
    $orderedUrls = @($orderedUrls | Select-Object -First $MaxReports)
}

$manifestPath = Join-Path $catalogDir "raib_manifest.csv"
if (-not (Test-Path $manifestPath)) {
    "report_url,pdf_url,file,status,error" | Set-Content -Encoding UTF8 $manifestPath
}

$counter = 0
foreach ($reportUrl in $orderedUrls) {
    $counter++
    try {
        $html = Get-RemoteText $reportUrl
        $pdfMatch = [regex]::Match($html, 'href="(https://assets\.publishing\.service\.gov\.uk/[^"]+\.pdf)"')
        if (-not $pdfMatch.Success) {
            "$reportUrl,,,no-pdf," | Add-Content -Encoding UTF8 $manifestPath
            continue
        }

        $pdfUrl = [System.Net.WebUtility]::HtmlDecode($pdfMatch.Groups[1].Value)
        $fileName = [System.IO.Path]::GetFileName(([System.Uri]$pdfUrl).AbsolutePath)
        $fileName = $fileName -replace '[^a-zA-Z0-9._-]', '_'
        $target = Join-Path $rawDir $fileName
        if (-not (Test-Path $target)) {
            Invoke-WebRequest -Uri $pdfUrl -Headers $headers -OutFile $target -UseBasicParsing
        }
        "$reportUrl,$pdfUrl,$fileName,downloaded," | Add-Content -Encoding UTF8 $manifestPath
        Write-Host "[$counter/$($orderedUrls.Count)] $fileName"
    } catch {
        $message = $_.Exception.Message -replace '[\r\n,]', ' '
        "$reportUrl,,,error,$message" | Add-Content -Encoding UTF8 $manifestPath
        Write-Warning "[$counter/$($orderedUrls.Count)] $reportUrl`: $message"
    }
    Start-Sleep -Milliseconds $DelayMilliseconds
}

Write-Host "Finished. Report pages: $($orderedUrls.Count). Files: $rawDir. Manifest: $manifestPath"
