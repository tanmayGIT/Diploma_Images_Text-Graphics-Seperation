cd C:\Users\mondal\Documents\DIBCO\Dataset\Seperated_All_Results\DIBCO_11\GT\
 for /R %%f in (*.bmp) do (
	"C:\Users\mondal\Documents\DIBCO\Dataset\Seperated_All_Results\BinEvalWeights\BinEvalWeights.exe" "%%f" 
 	)
@pause