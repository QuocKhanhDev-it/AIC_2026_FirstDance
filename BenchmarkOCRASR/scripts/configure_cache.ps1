param([string]$CacheRoot = "D:\Library\ai_cache")

$ErrorActionPreference = "Stop"
$resolvedRoot = [System.IO.Path]::GetFullPath($CacheRoot)
if (-not $resolvedRoot.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "CacheRoot must be on drive D: $resolvedRoot"
}

$variables = [ordered]@{
    AIC_AI_CACHE_ROOT     = $resolvedRoot
    HF_HOME               = Join-Path $resolvedRoot "huggingface"
    HF_HUB_CACHE          = Join-Path $resolvedRoot "huggingface\hub"
    HUGGINGFACE_HUB_CACHE = Join-Path $resolvedRoot "huggingface\hub"
    HF_XET_CACHE          = Join-Path $resolvedRoot "huggingface\xet"
    HF_ASSETS_CACHE       = Join-Path $resolvedRoot "huggingface\assets"
    HF_DATASETS_CACHE     = Join-Path $resolvedRoot "huggingface\datasets"
    TORCH_HOME            = Join-Path $resolvedRoot "torch"
    XDG_CACHE_HOME        = Join-Path $resolvedRoot "xdg"
    EASYOCR_MODULE_PATH   = Join-Path $resolvedRoot "easyocr"
    PADDLE_PDX_CACHE_HOME = Join-Path $resolvedRoot "paddle"
    AIC_VIETOCR_CACHE     = Join-Path $resolvedRoot "vietocr"
    AIC_WHISPER_CACHE     = Join-Path $resolvedRoot "whisper"
}

foreach ($entry in $variables.GetEnumerator()) {
    New-Item -ItemType Directory -Force -Path $entry.Value | Out-Null
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "User")
    Set-Item -Path "Env:$($entry.Key)" -Value $entry.Value
    Write-Output "$($entry.Key)=$($entry.Value)"
}

Write-Output "Persistent user cache variables configured. Open a new terminal to inherit them."
