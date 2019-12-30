 cd C:\Users\mondal\Documents\DIBCO\Dataset\Seperated_All_Results\DIBCO_17\GT\
 for /R %%f in (*.bmp) do (
	"C:\Users\mondal\Documents\DIBCO\Dataset\Seperated_All_Results\DIBCO_metrics\DIBCO_metrics.exe" "%%f" C:\Users\mondal\Documents\DIBCO\Dataset\Seperated_All_Results\DIBCO_17\AlgoResult\%%~nf.png "%%~nf_RWeights.dat" "%%~nf_PWeights.dat"
 	)
cmd /k

