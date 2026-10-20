# Build a 10,000-MID selection with complete stacks, starting from the 6,000 sample.
# Same logic as select_10000_mids.py, without Python.
#
# Inputs (in the folder of this script):
#     A00_G07_MAD_6000_sample_MID.txt   MIDs already downloaded (one per line)
#     output\output.csv                 all MIDs: StackID, DocID, SubDocId, SST, MasterindexID
#
# Steps:
#     1. Every stack touched by the sample is completed with all its MIDs.
#     2. Remaining class deficits (A00=4000, G07=3000, MAD=3000) are filled with whole,
#        randomly drawn stacks; a stack is taken only when it helps a class below target
#        more than it overshoots classes already full.
#     3. Writes (no header, one MID per line, LF line endings):
#          output\selected_mids.txt      full selection (sample included)
#          output\to_download_mids.txt   only MIDs not in the sample -> AWS export
#
# Run:  powershell -ExecutionPolicy Bypass -File .\select_10000_mids.ps1

param(
    [int]$TargetA00 = 4000,
    [int]$TargetG07 = 3000,
    [int]$TargetMAD = 3000,
    [int]$Seed = 13
)
$ErrorActionPreference = 'Stop'

$root       = $PSScriptRoot
$sampleFile = Join-Path $root 'A00_G07_MAD_6000_sample_MID.txt'
$allFile    = Join-Path $root 'output\output.csv'
$outDir     = Join-Path $root 'output'
$classes    = @('A00', 'G07', 'MAD')
$targets    = @{ A00 = $TargetA00; G07 = $TargetG07; MAD = $TargetMAD }

# Case-sensitive containers (PowerShell @{} hashtables ignore case).
function New-Set { return ,[System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal) }
function New-Map { return ,[System.Collections.Generic.Dictionary[string, System.Collections.Generic.HashSet[string]]]::new([System.StringComparer]::Ordinal) }

function Add-ToMap($map, [string]$key, [string]$value) {
    if (-not $map.ContainsKey($key)) { $map[$key] = New-Set }
    [void]$map[$key].Add($value)
}

# ---- 1. read the sample ------------------------------------------------------
$sample = New-Set
foreach ($line in [System.IO.File]::ReadLines($sampleFile)) {
    $mid = $line.Trim().TrimStart([char]0xFEFF)
    if ($mid -and $mid -notin @('masterindex_id', 'masterindexid', 'mid')) { [void]$sample.Add($mid) }
}

# ---- 2. read all MIDs (columns found by header name) ----------------------------
$midClasses = New-Map   # mid   -> classes
$stackMids  = New-Map   # stack -> mids
$midStacks  = New-Map   # mid   -> stacks
$header = $null
foreach ($line in [System.IO.File]::ReadLines($allFile)) {
    if (-not $line.Trim()) { continue }
    $parts = $line.Split(',')
    if ($null -eq $header) {
        $header = $parts | ForEach-Object { $_.Trim().TrimStart([char]0xFEFF).ToLower() }
        $iStack = [array]::IndexOf($header, 'stackid')
        $iSst   = [array]::IndexOf($header, 'sst')
        $iMid   = [array]::IndexOf($header, 'masterindexid')
        if ($iStack -lt 0 -or $iSst -lt 0 -or $iMid -lt 0) {
            throw "output.csv needs the columns StackID, SST and MasterindexID (found: $($header -join ', '))"
        }
        continue
    }
    if ($parts.Count -le [Math]::Max($iStack, [Math]::Max($iSst, $iMid))) { continue }
    $stack = $parts[$iStack].Trim()
    $sst   = $parts[$iSst].Trim().ToUpper()
    $mid   = $parts[$iMid].Trim()
    if (-not $stack -or -not $mid) { continue }
    Add-ToMap $stackMids $stack $mid
    Add-ToMap $midStacks $mid $stack
    if ($classes -contains $sst) { Add-ToMap $midClasses $mid $sst }
}

function Get-ClassCounts($mids) {
    $counts = @{ A00 = 0; G07 = 0; MAD = 0 }
    foreach ($m in $mids) {
        if ($midClasses.ContainsKey($m)) { foreach ($c in $midClasses[$m]) { $counts[$c]++ } }
    }
    return $counts
}
function Format-Counts($counts) { return "A00=$($counts.A00) G07=$($counts.G07) MAD=$($counts.MAD)" }

$unknown = @($sample | Where-Object { -not $midStacks.ContainsKey($_) }).Count
if ($unknown) { Write-Host "WARNING: $unknown sample MIDs are not in output.csv; kept, but cannot be completed." }

# ---- 3. complete the stacks of the sample ----------------------------------------
$selectedStacks = New-Set
foreach ($mid in $sample) {
    if ($midStacks.ContainsKey($mid)) { foreach ($s in $midStacks[$mid]) { [void]$selectedStacks.Add($s) } }
}
$selected = New-Set
foreach ($mid in $sample) { [void]$selected.Add($mid) }
foreach ($s in $selectedStacks) { foreach ($m in $stackMids[$s]) { [void]$selected.Add($m) } }

Write-Host "Sample: $($sample.Count) MIDs, $(Format-Counts (Get-ClassCounts $sample))"
Write-Host "After completing $($selectedStacks.Count) stacks: $($selected.Count) MIDs, $(Format-Counts (Get-ClassCounts $selected))"

# ---- 4. fill the deficits with whole random stacks ---------------------------------
$candidates = [System.Collections.Generic.List[string]]::new()
foreach ($s in $stackMids.Keys) { if (-not $selectedStacks.Contains($s)) { $candidates.Add($s) } }
$candidates.Sort([System.StringComparer]::Ordinal)
$rng = [System.Random]::new($Seed)
for ($i = $candidates.Count - 1; $i -gt 0; $i--) {
    $j = $rng.Next($i + 1)
    $tmp = $candidates[$i]; $candidates[$i] = $candidates[$j]; $candidates[$j] = $tmp
}

$counts = Get-ClassCounts $selected
foreach ($s in $candidates) {
    $deficit = @{}; $open = $false
    foreach ($c in $classes) {
        $deficit[$c] = [Math]::Max($targets[$c] - $counts[$c], 0)
        if ($deficit[$c] -gt 0) { $open = $true }
    }
    if (-not $open) { break }

    $new = @($stackMids[$s] | Where-Object { -not $selected.Contains($_) })
    if ($new.Count -eq 0) { continue }
    $added = Get-ClassCounts $new
    $gain = 0; $over = 0
    foreach ($c in $classes) {
        $gain += [Math]::Min($added[$c], $deficit[$c])
        if ($added[$c] -gt $deficit[$c]) { $over += $added[$c] - $deficit[$c] }
    }
    if ($gain -gt 0 -and $gain -ge $over) {
        foreach ($m in $new) { [void]$selected.Add($m) }
        [void]$selectedStacks.Add($s)
        foreach ($c in $classes) { $counts[$c] += $added[$c] }
    }
}

# ---- 5. write outputs ----------------------------------------------------------------
function Write-MidList([string]$path, $mids) {
    $list = [System.Collections.Generic.List[string]]::new()
    foreach ($m in $mids) { $list.Add($m) }
    $list.Sort([System.StringComparer]::Ordinal)
    [System.IO.File]::WriteAllText($path, (($list -join "`n") + "`n"))   # UTF-8 without BOM, LF
}

$toDownload = @($selected | Where-Object { -not $sample.Contains($_) })
[void][System.IO.Directory]::CreateDirectory($outDir)
Write-MidList (Join-Path $outDir 'selected_mids.txt') $selected
Write-MidList (Join-Path $outDir 'to_download_mids.txt') $toDownload

Write-Host ""
Write-Host "Final: $($selected.Count) MIDs in $($selectedStacks.Count) complete stacks"
foreach ($c in $classes) { Write-Host "  ${c}: $($counts[$c]) (target $($targets[$c]))" }
Write-Host "To download: $($toDownload.Count) MIDs -> $(Join-Path $outDir 'to_download_mids.txt')"
Write-Host "Full selection: $(Join-Path $outDir 'selected_mids.txt')"
