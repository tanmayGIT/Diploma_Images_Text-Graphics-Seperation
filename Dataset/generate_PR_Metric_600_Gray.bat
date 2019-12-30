 cd C:\Users\mondal\Documents\Dataset\Fujitsu_Seperated\600_DPI\Gray\Ground_Truth
 for /R %%f in (*.bmp) do (
 "C:\Users\mondal\Documents\Dataset\BinEvalWeights\BinEvalWeights.exe" "%%f"
 )