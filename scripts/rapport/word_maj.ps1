param([string]$Docx, [string]$Pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $false)
    # deux passes : numéros SEQ, puis sommaire / listes avec les bons numéros de page
    for ($i = 0; $i -lt 2; $i++) {
        $doc.Fields.Update() | Out-Null
        foreach ($t in $doc.TablesOfContents) { $t.Update() | Out-Null }
        foreach ($t in $doc.TablesOfFigures) { $t.Update() | Out-Null }
    }
    $doc.Save()
    $doc.ExportAsFixedFormat($Pdf, 17)
    $pages = $doc.ComputeStatistics(2)
    Write-Output "pages=$pages"
    $doc.Close($false)
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
