#include "dehaze.h"
#include <stdio.h>
//#include "/usr/local/include/opencv2/opencv.hpp"
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
IplImage*  img_src;
IplImage* dehazeImage;
int main()
{
               img_src = cvLoadImage("./input/photo/haze.bmp",CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);


//Allocate the 2D dark channel
		uchar **darkChannel = new uchar *[img_src->height];
		for (int i = 0; i < img_src->height; ++i)
		{
			darkChannel[i] = new uchar [img_src->width];
		}

// 		//Allocate the dark channel histogram

		int hiWidth, hiHeight;

		CalcDarkChannel(img_src, darkChannel, &hiWidth, &hiHeight);

		//AtmosphericLight
		uchar asLight;

//		AtmosphericLight(img_src, hiWidth, hiHeight, asLight);
		asLight = 255;

		//Allocate the transmission
		float **transmission = new float *[img_src->height];
		#pragma acc kernels loop
		for (int i = 0; i < img_src->height; ++i)
		{
			transmission[i] = new float [img_src->width];
		}

		Transmission(img_src, darkChannel, asLight, transmission);

		SoftMatting(img_src, transmission);

		//dehaze image
		CvMat *transMat = cvCreateMat(img_src->height, img_src->width, CV_8UC1);
		dehazeImage = cvCreateImage(cvGetSize(img_src),img_src->depth,img_src->nChannels);

		RecoverSceneRadiance(img_src, transmission, asLight, dehazeImage, transMat);

                cvSaveImage("Dehaze.bmp",dehazeImage);

	
return 0;
}

