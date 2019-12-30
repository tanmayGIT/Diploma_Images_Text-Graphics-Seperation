//
// Created by tmondal on 19/12/2018.
//

#include "hdr/Feature_Space_Partition_Binarization.h"
#include "../../util/hdr/BasicAlgo.h"
/*
 * Implements binarization of documents
 * Check this paper: DOI 10.1007/s10032-010-0142-4
 * "Binarization of degraded document image based on feature space partitioning and classification"
 */


// Definitions
#define uget(x,y)    at<unsigned char>(y,x)
#define uset(x,y,v)  at<unsigned char>(y,x)=v;
#define fget(x,y)    at<float>(y,x)
#define fset(x,y,v)  at<float>(y,x)=v;

// Parameters
#define DEBUG 0

// Globals
std::string auxMethod = "n";


Feature_Space_Partition_Binarization* Feature_Space_Partition_Binarization::instance = 0;

Feature_Space_Partition_Binarization* Feature_Space_Partition_Binarization::getInstance() {
    if (!instance)
        instance = new Feature_Space_Partition_Binarization();

    return instance;
}

Feature_Space_Partition_Binarization::Feature_Space_Partition_Binarization() {
}

Feature_Space_Partition_Binarization::~Feature_Space_Partition_Binarization() {
}


// Method that computes the stroke width
int Feature_Space_Partition_Binarization::computeSW(IplImage *image) {

    IplImage *binImage = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1);
    int *distanceHistogram = (int*)malloc(image->height * sizeof(int));
    int distance, maxBin = -1, maxDistance;
    int beginEdge, endEdge, pixel;

    // Compute the edge pixels
    cvSobel(image, binImage, 1, 1);
    // Binarize the image
    cvThreshold(binImage, binImage, 32, 255, CV_THRESH_BINARY);
    // Initialize the histogram
    for(int bin = 0; bin < binImage->height; bin++)
        distanceHistogram[bin] = 0;
    // Compute the distance histogram (histogram of the distances between two successive edge pixels)
    for(int row = 0; row < binImage->height; row++) {
        beginEdge = -1;
        endEdge = -1;
        for(int col = 0; col < binImage->width; col++) {
            pixel = row * binImage->width + col;
            if((unsigned char)binImage->imageData[pixel] == 255 && beginEdge == -1) {
                beginEdge = pixel;
            } else if((unsigned char)binImage->imageData[pixel] == 255 && beginEdge != -1) {
                endEdge = pixel;
                distance = endEdge - beginEdge;
                distanceHistogram[distance]++;
                beginEdge = endEdge;
            }
        }
    }

    // Recover the distance with highest value in the distance histogram
    for(int bin = 2; bin < binImage->height; bin++)
        if(distanceHistogram[bin] > maxBin) {
            maxBin = distanceHistogram[bin];
            maxDistance = bin;
        }

    cvReleaseImage(&binImage);
    delete [] distanceHistogram;
    return maxDistance;
}

// Method that computes the MSW
float Feature_Space_Partition_Binarization::computeMSW(IplImage *grayImage, int strokeWidth, int pk) {

    int pkx, pky, pixel, x, y;
    pkx = pk % grayImage->width;
    pky = pk / grayImage->width;
    float sum = 0, msw = 0;

    for(int i = -strokeWidth; i <= strokeWidth; i++) {
        for(int j = -strokeWidth; j <= strokeWidth; j++) {
            y = pky - j;
            x = pkx - i;
            pixel = y * grayImage->width + x;
            if(pixel >= 0 && pixel < grayImage->width * grayImage->height)
                sum += (unsigned char)grayImage->imageData[pixel];
        }
    }

    msw = sum / powf(2 * strokeWidth + 1, 2);
    return msw;

}

// Method that convert a neighbour index to a pixel (based on the structural contrast neighbourhood)
int Feature_Space_Partition_Binarization::convertSCNeighbourhoodToPixel(int x, int y, int width, int k) {

    if(k == 0)
        return y * width + x - 1;
    else if(k == 1)
        return y * width + x - width - 1;
    else if(k == 2)
        return y * width + x - width;
    else if(k == 3)
        return y * width + x - width + 1;
    else if(k == 4)
        return y * width + x + 1;
    else if(k == 5)
        return y * width + x + width + 1;
    else if(k == 6)
        return y * width + x + width;
    else if(k == 7)
        return y * width + x + width - 1;

}

// Method that computes the structural contrast feature image
void Feature_Space_Partition_Binarization::computeSC(IplImage *grayImage, IplImage *structuralContrastImage, int strokeWidth) {

    float max = 0;
    float minMSW[4];
    float maxMSW[4];
    float msw[4];
    int pixel = 0;

    // For each pixel
    for(int x = 0; x < grayImage->width; x++) {
        for(int y = 0; y < grayImage->height; y++) {
            pixel = y * grayImage->width + x;
            for(int k = 0; k < 4; k++) {
                // Compute the MSW (Equation 4 from the paper)
                msw[0] = computeMSW(grayImage, strokeWidth, convertSCNeighbourhoodToPixel(x, y, grayImage->width, k));
                msw[1] = computeMSW(grayImage, strokeWidth, convertSCNeighbourhoodToPixel(x, y, grayImage->width, k + 1));
                msw[2] = computeMSW(grayImage, strokeWidth, convertSCNeighbourhoodToPixel(x, y, grayImage->width, ((k + 4) % 8)));
                msw[3] = computeMSW(grayImage, strokeWidth, convertSCNeighbourhoodToPixel(x, y, grayImage->width, ((k + 5) % 8)));
                // Get the min MSW
                minMSW[k] = msw[0];
                for(int m = 1; m < 3; m++)
                    if(minMSW[k] > msw[m])  minMSW[k] = msw[m];
                maxMSW[k] = std::max(minMSW[k], msw[3]);
            }
            // Compute the max from the min MSW
            max = maxMSW[0];
            for(int m = 1; m < 4; m++)
                if(max < maxMSW[m])  max = maxMSW[m];
            // Compute the structural contrast
            if(max < (unsigned char)grayImage->imageData[pixel])
                structuralContrastImage->imageData[pixel] = 0;
            else
                structuralContrastImage->imageData[pixel] = max - (unsigned char)grayImage->imageData[pixel];
        }
    }

}


// Return and remove a random point from a list
Point Feature_Space_Partition_Binarization::getRandomPoint(std::vector<Point> &available) {
    srand((unsigned)time(0));
    int i = rand() % available.size();
    Point p = available[i];
    if (DEBUG) printf("Got a random element (%d, %d) (from position %d), now there are %d elements available\n", p.x, p.y, i, available.size());
    return p;
}


// Method that executes clustering over feature space using auxiliary binary image
void Feature_Space_Partition_Binarization::executeClusteringAndBinarization(IplImage *gr, IplImage *im, cv::Mat niblack) {

    // Build the feature space
    int w = im->width;
    int h = im->height;
    int dim = 256;
    cv::Mat contrast = cv::cvarrToMat(im);
    cv::Mat gray = cv::cvarrToMat(gr);
    int fs[dim][dim];

    // It's easier if we represent the non-labeled points as an array: i = y + w * x
    std::vector<Point> available;

    // x = structural contrast
    // y = gray level
    for (int x = 0; x < dim; x++) {
        for (int y = 0; y < dim; y++) {
            fs[x][y] = 0;
            Point p;
            p.x = x;
            p.y = y;
            available.push_back(p);
        }
    }
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int c = contrast.uget(x,y);
            int g = gray.uget(x,y);
            fs[c][g]++;
        }
    }

    // Print feature space
    if (DEBUG) {
        for(int i = 0; i < dim; i++) {
            for(int j = 0; j < dim; j++) {
                std::cout << std::setw(3) << std::setfill('0') << fs[i][j];
                printf(" ");
            }
            printf("\n");
        }
    }

    // Initialize labeled matrix
    // Start label as 0, and increment as new groups are formed
    // -2  means not labeled
    // -1  is the current path
    //  0+ means labeled as n (n >= 0)
    int labeled[dim][dim];
    for(int i=0; i < dim; i++)
        for(int j=0; j < dim; j++)
            labeled[i][j] = -2;
    int label = 0;

    int max = 0;
    Point point;
    Point start = getRandomPoint(available);
    point.x = start.x;
    point.y = start.y;
    int n = 0;
    while(available.size() > 0) {

        if (DEBUG) printf("%d. We are at point (%d,%d) (%d remaining)\n", n, point.x, point.y, available.size());
        n++;

        // Find maximum inside 5x5 window
        Point previous;
        previous.x = point.x;
        previous.y = point.y;
        for (int x = point.x - 2; x <= point.x + 2; x++) {
            for (int y = point.y - 2; y <= point.y + 2; y++) {
                if (x >= 0 && x < dim && y >= 0 && y < dim) {

                    if (DEBUG) printf("We are at point (%d,%d) of the 5x5 window\n", x, y);

                    if (labeled[x][y] == -2) {
                        labeled[x][y] = -1;
                        if (available.size() == 1) available.clear();
                        for (int c = 0; c < available.size(); c++)
                            if (available[c].x == x && available[c].y == y) available.erase(available.begin() + c);
                    }

                    if (fs[x][y] > max) {
                        if (DEBUG) printf("Point (%d,%d) is max\n", x, y);
                        max = fs[x][y];
                        point.x = x;
                        point.y = y;
                    }
                }
            }
        }
        // Already labeled
        if (labeled[point.x][point.y] > -1) {
            if (DEBUG) printf("Point (%d,%d) is labeled as %d\n", point.x, point.y, label);
            for(int i=0; i < dim; i++)
                for(int j=0; j < dim; j++)
                    if (labeled[i][j] == -1) {
                        labeled[i][j] = labeled[point.x][point.y];
                    }
            if (available.size() > 0) {
                Point next = getRandomPoint(available);
                point.x = next.x;
                point.y = next.y;
            }
        }
        else if (previous.x == point.x && previous.y == point.y) {
            if (DEBUG) printf("Local max reached at (%d,%d)\n", point.x, point.y);
            // Local maximum was reached
            for(int i=0; i < dim; i++)
                for(int j=0; j < dim; j++)
                    if (labeled[i][j] == -1) {
                        labeled[i][j] = label;
                    }
            label++;
            if (available.size() > 0) {
                Point next = getRandomPoint(available);
                point.x = next.x;
                point.y = next.y;
            }
        }

    }

    /* Or, a simpler approach: each pixel is a partition
    for(int i=0; i < w; i++) {
      for(int j=0; j < h; j++) {
        labeled[i][j] = label;
        label++;
      }
    }
    */

    // Print the partitions (debugging only)
    if (DEBUG) {
        printf("We have %d labels (from %d pixels)\n", label, w * h);
        for(int i = 0; i < dim; i++) {
            for(int j = 0; j < dim; j++) {
                std::cout << std::setw(3) << std::setfill('0') << labeled[i][j];
                printf(" ");
            }
            printf("\n");
        }
    }

    // Classify the partitions
    char partitions[label]; // 't' : text, 'b' : background
    int ntext[label]; // Number of text partitions per label
    int nbg[label]; // Number of background partitions per label
    for(int i = 0; i < label; i++) {
        partitions[i] = 'x';
        ntext[i] = 0;
        nbg[i] = 0;
    }

    // Count number of text pixels and background pixels on the auxiliary image
    for(int x = 0; x < w; x++) {
        for(int y = 0; y < h; y++) {
            int c = contrast.uget(x,y);
            int g = gray.uget(x,y);
            int l = labeled[c][g];
            if (niblack.uget(x,y) == 0) {
                ntext[l]++;
            }
            else if (niblack.uget(x,y) == 255) {
                nbg[l]++;
            }
        }
    }

    // Classify the partitions as either text or background
    for(int i = 0; i < label; i++) {
        if (ntext[i] > nbg[i])
            partitions[i] = 't';
        else
            partitions[i] = 'b';
    }

    // Final binarization
    cv::Mat final (h, w, CV_8U);
    for(int x = 0; x < w; x++) {
        for(int y = 0; y < h; y++) {
            int c = contrast.uget(x,y);
            int g = gray.uget(x,y);
            int l = labeled[c][g];
            if (partitions[l] == 't') {
                final.uset(x,y,0);
            }
            else if (partitions[l] == 'b') {
                final.uset(x,y,255);
            }
        }
    }
    showImage(final);
    // imwrite(std::string(inputImgIndex) + ".this." + std::string(auxMethod) + ".png", final);
}


float Feature_Space_Partition_Binarization::calcLocalStats(cv::Mat &im, cv::Mat &map_m, cv::Mat &map_s, int winx, int winy) {
    float m,s,max_s, sum, sum_sq, foo;
    int wxh  = winx/2;
    int wyh  = winy/2;
    int x_firstth= wxh;
    int y_lastth = im.rows-wyh-1;
    int y_firstth= wyh;
    float winarea = winx*winy;
    max_s = 0;
    for(int j=y_firstth; j <= y_lastth; j++) {
        // Calculate the initial window at the beginning of the line
        sum = sum_sq = 0;
        for(int wy=0 ; wy<winy; wy++)
            for(int wx=0 ; wx<winx; wx++) {
                foo = im.uget(wx,j-wyh+wy);
                sum += foo;
                sum_sq += foo*foo;
            }
        m = sum / winarea;
        s = sqrt ((sum_sq - (sum*sum)/winarea)/winarea);
        if (s > max_s) max_s = s;
        map_m.fset(x_firstth, j, m);
        map_s.fset(x_firstth, j, s);
        // Shift the window, add and remove  new/old values to the histogram
        for(int i=1; i <= im.cols-winx; i++) {
            // Remove the left old column and add the right new column
            for(int wy=0; wy<winy; ++wy) {
                foo = im.uget(i-1,j-wyh+wy);
                sum -= foo;
                sum_sq -= foo*foo;
                foo = im.uget(i+winx-1,j-wyh+wy);
                sum += foo;
                sum_sq += foo*foo;
            }
            m = sum / winarea;
            s = sqrt ((sum_sq - (sum*sum)/winarea)/winarea);
            if (s > max_s) max_s = s;
            map_m.fset(i+wxh, j, m);
            map_s.fset(i+wxh, j, s);
        }
    }
    return max_s;
}


void Feature_Space_Partition_Binarization::auxBinarization(cv::Mat & im, cv::Mat & output, int winx, int winy, float k, float dR, std::string auxm) {

  //  showImage(output);
    float m, s, max_s;
    float th=0;
    double min_I, max_I;
    int wxh  = winx/2;
    int wyh  = winy/2;
    int x_firstth= wxh;
    int x_lastth = im.cols-wxh-1;
    int y_lastth = im.rows-wyh-1;
    int y_firstth= wyh;
    int mx, my;
    // Create local statistics and store them in a double matrices
    cv::Mat map_m = cv::Mat::zeros (im.rows, im.cols, CV_32F);
    cv::Mat map_s = cv::Mat::zeros (im.rows, im.cols, CV_32F);
    max_s = calcLocalStats (im, map_m, map_s, winx, winy);
    cv::minMaxLoc(im, &min_I, &max_I);
    cv::Mat thsurf(im.rows, im.cols, CV_32F);
    // Create the threshold surface, including border processing
    for(int j=y_firstth; j <= y_lastth; j++) {
        // Non-border area
        for(int i=0 ; i <= im.cols-winx; i++) {
            m  = map_m.fget(i + wxh, j);
            s  = map_s.fget(i + wxh, j);

            // Calculate the threshold
            if (auxm == "n") { // Niblack
                th = m + k*s;
            }
            else if (auxm == "s") { // Sauvola
                th = m * (1 + k*(s/dR-1));
            }
            else if (auxm == "w") { // Wolfjolion
                th = m + k * (s/max_s-1) * (m-min_I);
            }

            thsurf.fset(i+wxh,j,th);

            if (i == 0) {
                // Left border
                for(int i=0; i <= x_firstth; ++i)
                    thsurf.fset(i,j,th);
                // Left upper corner
                if (j == y_firstth)
                    for(int u=0; u < y_firstth; ++u)
                        for(int i=0; i <= x_firstth; ++i)
                            thsurf.fset(i,u,th);

                // Left lower corner
                if (j == y_lastth)
                    for(int u=y_lastth+1; u<im.rows; ++u)
                        for(int i=0; i<=x_firstth; ++i)
                            thsurf.fset(i,u,th);
            }

            // Upper border
            if (j == y_firstth)
                for(int u=0; u < y_firstth; ++u)
                    thsurf.fset(i+wxh,u,th);

            // Lower border
            if (j == y_lastth)
                for(int u=y_lastth+1; u < im.rows; ++u)
                    thsurf.fset(i+wxh,u,th);
        }

        // Right border
        for(int i=x_lastth; i < im.cols; ++i)
            thsurf.fset(i,j,th);

        // Right upper corner
        if (j == y_firstth)
            for(int u=0; u < y_firstth; ++u)
                for(int i=x_lastth; i < im.cols; ++i)
                    thsurf.fset(i,u,th);

        // Right lower corner
        if (j == y_lastth)
            for(int u=y_lastth+1; u<im.rows; ++u)
                for(int i=x_lastth; i<im.cols; ++i)
                    thsurf.fset(i,u,th);
    }
/*    printMatrix(thsurf);
    imwrite("/Users/tmondal/Documents/Imag_Output/thSurf.png",thsurf);*/

    for (int iRow = 0; iRow < thsurf.rows; iRow++){
        for (int jCol = 0; jCol < thsurf.cols; jCol++){
            if (im.at<uchar>(iRow, jCol) >= thsurf.at<uchar>(iRow, jCol)){
                output.at<uchar>(iRow, jCol) = 255;
            } else {
                output.at<uchar>(iRow, jCol) = 0;
            }
        }
    }
/*
    for(int y=0; y < im.rows; ++y)
        for(int x=0; x < im.cols; ++x) {
            if ( (im.uget(x,y) > thsurf.fget(x,y)) || (im.uget(x,y) == thsurf.fget(x,y)) ) {
                output.uset(x,y,255);
            }
            else {
                output.uset(x,y,0);
            }
        }*/
   // showImage(output);
/*    String imagenam = std::string(inputImgIndex) + "." + std::string(auxMethod) + ".png" ;
    imwrite(imagenam, output);*/
}

void Feature_Space_Partition_Binarization::runBinarization(IplImage & origImage, cv::Mat & binarizedImage){
    // send the color image in this function

    IplImage image = origImage;

    // Gray-scale image
    IplImage *grayImage = cvCreateImage(cvSize((&image)->width, (&image)->height), IPL_DEPTH_8U, 1);

    // Structural contrast feature image (this is our feature space)
    IplImage* structuralContrastImage = cvCreateImage(cvSize((&image)->width, (&image)->height), IPL_DEPTH_8U, 1);

    int strokeWidth;

    // RGB to Gray
    if ((&image)->nChannels > 2) // that means it is a color image
        cvCvtColor((&image), grayImage, CV_RGB2GRAY);

    // Compute the stroke width automatically over the image
    strokeWidth = computeSW(grayImage);

    // Once with the stroke width, compute the structural contrast feature image
    computeSC(grayImage, structuralContrastImage, strokeWidth);
    cvSaveImage("/Users/tmondal/Documents/Imag_Output/structuralContrastImage.png",structuralContrastImage);


    // Auxiliary binarization
    cv::Mat input = cv::cvarrToMat(&origImage) ; // this is the greyscale image here
    float optK = 0.5;
    int winy = (int) (2.0 * input.rows-1)/3;
    int winx = (int) input.cols-1 < winy ? input.cols-1 : winy;
    if (winx > 100) winx = winy = 40;

    cv::Mat niblackGame = Mat::zeros (input.rows, input.cols, CV_8UC1);
   // showImage(niblackGame);


    auxBinarization(input, niblackGame, winx, winy, optK, 128, auxMethod);
/*    imwrite("/Users/tmondal/Documents/Imag_Output/niBlack.png",niblack);
    showImage(niblack);*/

    // Clustering & final binarization
    cvSaveImage("/Users/tmondal/Documents/Imag_Output/grayImage.png",grayImage);
    cvSaveImage("/Users/tmondal/Documents/Imag_Output/structuralContrastImage.png",structuralContrastImage);
    executeClusteringAndBinarization(grayImage, structuralContrastImage, niblackGame);
   // showImage(niblack);
}