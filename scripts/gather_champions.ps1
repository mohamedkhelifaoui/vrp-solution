$champ = "data\solutions_champions\all"
New-Item -ItemType Directory -Force -Path $champ | Out-Null

Import-Csv data\reports\ml_champions.csv | ForEach-Object {
  $inst = $_.instance
  $meth = $_.method
  $srcPattern = "data\ml_runs\tmp\{0}\{1}\*.json" -f $inst, $meth
  $src = Get-ChildItem $srcPattern -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -eq $src) {
    Write-Host "[WARN] No JSON found for $inst / $meth at $srcPattern"
  } else {
    Copy-Item $src.FullName (Join-Path $champ ($src.Name)) -Force
  }
}
