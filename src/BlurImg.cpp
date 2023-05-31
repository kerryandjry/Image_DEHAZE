#include "stdio.h"
#include "stdlib.h"
#include "memory.h"
#include "omp.h"
#include "mytime.h"
#include "/usr/local/include/opencv2/opencv.hpp"
#include "/usr/local/include/opencv2/core/core.hpp"
#include "/usr/local/include/opencv2/highgui/highgui.hpp"
#include "/usr/local/include/opencv2/imgcodecs/legacy/constants_c.h"
#include <string>
#include <iostream>

//Block Size : Range = 10 - 100
#define Blocksize 15

using namespace std;
using namespace cv;

float* BoxBlurGray(unsigned char* Data, unsigned char* Dest, int Width, int Height, int Radius)
{
	const float Inv255 = 1.0f / 255;
	int X, Y, Z ,XX ,YY;
	float* P = (float*)malloc( Width * Height * sizeof(float)); 
	float Sum = 0, InvAmount;
	int LastAddress, NextAddress;
	float* Pointer;
	float* Result = (float*)malloc( Width * Height * sizeof(float));
	float* ColSum = (float*)malloc( Width * sizeof(float));
	int* RowOffset = (int*)malloc( (Width + Radius + Radius) * sizeof(int));
	int* ColOffSet = (int*)malloc( (Height + Radius + Radius) * sizeof(int));

	RowOffset += Radius;
	ColOffSet += Radius;
	InvAmount = 1.0f / ((2 * Radius + 1) * (2 * Radius + 1));

	for (Y = 0; Y < Width * Height; Y++) P[Y] = Data[Y] * Inv255;

	//#pragma omp parallel for
	for (X = -Radius; X < Width + Radius; X++)
    	{
        	if (X < 0)
		{
			XX = -X;
			while (XX >= Width) XX -= Width;
            		RowOffset[X] = XX;
		}
        	else if (X >= Width)
		{
            		XX = Width - (X - Width + 2);
			while (XX < 0) XX += Width;
			RowOffset[X] = XX;
		}
        	else
		{
            		RowOffset[X] = X;
		}
    	}
	//#pragma omp parallel for
	for (Y = -Radius; Y < Height + Radius; Y++)
    	{
        	if (Y < 0)
		{
            		YY = -Y;
			while (YY >= Height) YY -= Height;
			ColOffSet[Y] = (int) YY * Width;
		}
        	else if (Y >= Height)
		{
            		YY = Height - (Y - Height + 2);
			while (YY < 0) YY += Height;
			ColOffSet[Y] = (int) YY * Width;
		}
        	else
		{
            		ColOffSet[Y] = (int) Y * Width;
		}
    	}

	for (Y = 0; Y < Height; Y++)
	{
		if (Y == 0)
		{
			//#pragma omp parallel for default(none) private(X, Sum, Z) shared(Width, Radius, Data, ColOffSet, ColSum)
			for (X = 0; X < Width; X++)
			{
				Sum = 0;
				for (Z = -Radius; Z <= Radius; Z++) Sum += P[ColOffSet[Z] + X];
				ColSum[X] = Sum;
			}
		}
		else
		{
			#pragma omp parallel for 
			for (X = 0; X < Width; X++) ColSum[X] = ColSum[X] - P[ColOffSet[Y - Radius - 1] + X] + P[ColOffSet[Y + Radius] + X];
		}		
		for (Z = -Radius, Sum = 0; Z <= Radius; Z++) Sum += ColSum[RowOffset[Z]];
		Result[Width * Y] = Sum * InvAmount;

		for (X = 1; X < Width; X++)
		{
			Sum = Sum - ColSum[RowOffset[X - Radius - 1]] + ColSum[RowOffset[X + Radius]];
			Result[Width * Y + X] = Sum * InvAmount;
		}
	}

	for (Y = 0; Y < Width * Height; Y++) Dest[Y] = Result[Y];

	cout << Sum << endl; 

	RowOffset -= Radius;
	ColOffSet -= Radius;

	free(ColOffSet);
	free(RowOffset);
	free(ColSum);

}

int main(int argc, char ** argv)
{
	if (argc != 2)
	{
		printf("Usage: ./ImageFile\n");
		exit(1);
	}

	double t1, t2;

	printf("loadImg\n");

	t1 = get_seconds();

	Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if(! src.data)
	{
		cout << "Could not open or find the image!!" << std::endl;
		exit(1);
	}

	Mat dest(src.rows, src.cols, CV_8UC3);

	cout << "Height = " << src.rows << endl;
	cout << "Width = " << src.cols << endl;

	unsigned char *srcData;
	unsigned char *destData;

	srcData = src.data;
	destData = dest.data;


	BoxBlurGray(srcData, destData, src.cols, src.rows, Blocksize);

	t2 = get_seconds();

	dest.data = destData;

	namedWindow("Source Image", WINDOW_AUTOSIZE);
	imshow("Source Image", src);
	namedWindow("Blur Image", WINDOW_AUTOSIZE);
	imshow("Blur Image", dest); 
	imwrite("Blur_Image.jpg", dest);	

	waitKey(0);

	printf("\nElapsed Time = %18.6e\n", t2 - t1);

	return 0;
}
