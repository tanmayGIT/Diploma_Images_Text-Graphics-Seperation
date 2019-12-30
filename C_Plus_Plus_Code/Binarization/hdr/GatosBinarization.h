//
// Created by tmondal on 19/12/2018.
//

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "../../util/hdr/BasicAlgo.h"
#ifndef CLION_PROJECT_GATOSBINARIZATION_H
#define CLION_PROJECT_GATOSBINARIZATION_H

using namespace std;
using namespace cv;
class GatosBinarization {

public :

    GatosBinarization();
    static GatosBinarization* getInstance();
    virtual ~GatosBinarization();

    void qr_binarize_gatos(unsigned char *,int, int);
    void qr_gatos_mask(unsigned char *, const unsigned char *, const unsigned char *,
                              int, int, unsigned, int, int, int);
    void qr_interpolate_background(unsigned char *, int *, int *, const unsigned char *,
                                          const unsigned char *, int, int,unsigned, int);
    void qr_sauvola_mask(unsigned char *, unsigned *, int *, const unsigned char *, int, int);
    void qr_wiener_filter_5_cross_5(unsigned char *, int, int);
    void qr_wiener_filter_3_cross_3(unsigned char *,  int,   int);
    unsigned char *qr_binarize_special(const unsigned char *,int, int);

    void runGaborBinarization(Mat&, Mat& );

private:
    static GatosBinarization* instance;

};


#endif //CLION_PROJECT_GATOSBINARIZATION_H
