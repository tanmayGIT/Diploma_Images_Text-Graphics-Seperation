 cd C:\Users\Tanmoy\Documents\Scan_Fujitsu_Hidden_Seperated\300_DPI\Gray\Ground_Truth
 for /R %%f in (*.bmp) do (
 "C:\Users\Tanmoy\Documents\Scan_Fujitsu_Hidden_Seperated\BinEvalWeights\BinEvalWeights.exe" "%%f"
 )