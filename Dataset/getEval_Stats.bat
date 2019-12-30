 cd C:\Users\mondal\Documents\Dataset\Fujitsu_Seperated\300_DPI\Gray\Ground_Truth\
 for /R %%f in (*.bmp) do (
	"C:\Users\mondal\Documents\Dataset\DIBCO_metrics\DIBCO_metrics.exe" "%%f" C:\Users\mondal\Documents\Dataset\Fujitsu_Seperated\300_DPI\Gray\BG_Seperated\Results-9\%%~nfOutput.png C:\Users\mondal\Documents\Dataset\Fujitsu_Seperated\300_DPI\Gray\Ground_Truth\%%~nf_RWeights.dat C:\Users\mondal\Documents\Dataset\Fujitsu_Seperated\300_DPI\Gray\Ground_Truth\%%~nf_PWeights.dat
 	)
cmd /k