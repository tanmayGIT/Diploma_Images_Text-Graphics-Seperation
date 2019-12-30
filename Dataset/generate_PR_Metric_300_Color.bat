 cd C:\Users\mondal\Documents\Dataset\Fujitsu_Seperated\300_DPI\Color\Ground_Truth
 for /R %%f in (*.bmp) do (
 "C:\Users\mondal\Documents\Dataset\BinEvalWeights\BinEvalWeights.exe" "%%f"
 )