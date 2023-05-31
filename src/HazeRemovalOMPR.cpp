#include "stdio.h"
#include "stdlib.h"
#include "memory.h"
#include "omp.h"
#include "mytime.h"
#include "/usr/local/include/opencv2/opencv.hpp"
#include "/usr/local/include/opencv2/core/core.hpp"
#include "/usr/local/include/opencv2/highgui/highgui.hpp"
//#include <highgui.h>
#include <string>
#include <iostream>

//#define STRIDE 4808
//Block Size : Range = 10 - 100
#define Blocksize 15
//Patch Size : Range = 20 - 400
#define GuideBlocksize 50
//Airlight   : Range = 100 - 255
#define Airlight 220
//Fog Degree : Range = 50% - 100%
#define OMEGA 95
//Epsilon    : Range = 1 - 50
#define EPSILON 10
//TAU        : Range = 1 - 50
#define TAU 10

using namespace std;
using namespace cv;

void MinFilter(unsigned char *Scan0,int Width,int Height, int Radius )
{
	int X, Y, I, Stride, Index, Speed=0;
	int StrideC,HeightC;
	int Temp1,Temp2,Fast1,Fast2;
	unsigned char * Pointer,*Expand,*ColMinValue;
	register int MinValue,SpeedC;

	Stride =Width;
	StrideC = Stride + Radius * 2;
	HeightC = Height + Radius * 2;

	Expand = (unsigned char *)malloc( StrideC * HeightC);
	ColMinValue = (unsigned char *)malloc( Radius+Radius + Width);
	memset( Expand, 255,StrideC * HeightC);
	#pragma omp parallel for
	for (Y = 0;Y<Height;Y++) memcpy(Expand + (Radius + Y) * StrideC + Radius,  Scan0 + Y * Stride, Stride);
	memset( ColMinValue, 255,2 * Radius + Width );
	ColMinValue+=Radius;

	#pragma omp parallel for private(X, I, MinValue, SpeedC)// reduction(+: SpeedC)
	for (X = 0;X<Width;X++)
	{
		SpeedC = Radius * StrideC + Radius + X;    
		MinValue = ColMinValue[X];
		for (I = 0;I<=Radius;I++)
		{
			if (Expand[SpeedC] < MinValue) MinValue = Expand[SpeedC] ;
			SpeedC = SpeedC + StrideC;
		}
		ColMinValue[X] = MinValue;
	}

	#pragma omp parallel for private(X, I, MinValue) //reduction(+: Speed)
	for (X = 0;X<Width;X++)
	{
		MinValue = ColMinValue[X - Radius];
		for (I =  X - Radius + 1;I<= X + Radius;I++)
			if ( MinValue > ColMinValue[I])  MinValue = ColMinValue[I];
		/*
		Scan0[Speed] = MinValue;
		Speed = Speed + 1;
		*/
		Scan0[X] = MinValue;
	}       

	Temp1 = (Radius + Radius) * StrideC + Radius;
	Temp2 = -StrideC + Radius;

	//Speed = Stride;
	//#pragma omp parallel for
	for (Y = 1;Y<Height;Y++)
	{
		Speed = Stride * Y;
		#pragma omp parallel for private(X, I, SpeedC, Fast1, Fast2, MinValue)// shared(Temp1, Temp2, Y)
		for (X = 0;X<Width;X++)
		{
			SpeedC = Y * StrideC + X;
			Fast1 = SpeedC + Temp1;
			Fast2 = SpeedC + Temp2;
			if (ColMinValue[X] != Expand[Fast2] )
			{
				if (ColMinValue[X] > Expand[Fast1])  ColMinValue[X] = Expand[Fast1];
			}
			else
			{
				MinValue = 255;
				SpeedC = Fast2;
				for (I = -Radius;I<=Radius;I++)
				{
					SpeedC = SpeedC + StrideC;
					if ( MinValue > Expand[SpeedC] )    MinValue = Expand[SpeedC];
				}
				ColMinValue[X] = MinValue;
			}
		}
		MinValue = ColMinValue[-Radius];
		for (I = -Radius+1;I<=Radius;I++)
			if (MinValue > ColMinValue[I])  MinValue = ColMinValue[I];
		Scan0[Speed] = MinValue;

		//#pragma omp parallel for private(X, I) reduction(min: MinValue)
		for (X = 1;X<Width;X++)
		{
			if (MinValue != ColMinValue[X - Radius - 1] )
			{
				if (MinValue > ColMinValue[X + Radius] )  MinValue = ColMinValue[X + Radius];
			}
			else
			{
				MinValue = ColMinValue[X - Radius];
				for (I =  X - Radius + 1 ;I<=Radius + X;I++)
				{
					if (MinValue > ColMinValue[I])	MinValue = ColMinValue[I];
				}
			}
			/*
			Scan0[Speed1] = MinValue;
			Speed = Speed + 1;
			*/

			Scan0[X + Speed -1] = MinValue;
		}
	}
	ColMinValue-=Radius;
	free( Expand);
	free( ColMinValue);
}

float* BoxBlurGray(float* Data, int Width, int Height, int Radius)
{
	int X, Y, Z ,XX ,YY;
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

	#pragma omp parallel for
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
	#pragma omp parallel for
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

	//#pragma omp parallel for private(Sum, X)
	for (Y = 0; Y < Height; Y++)
	{
		if (Y == 0)
		{
			#pragma omp parallel for default(none) private(X, Sum, Z) shared(Width, Radius, Data, ColOffSet, ColSum)
			for (X = 0; X < Width; X++)
			{
				Sum = 0;
				for (Z = -Radius; Z <= Radius; Z++) Sum += Data[ColOffSet[Z] + X];
				ColSum[X] = Sum;
			}
		}
		else
		{
			//LastAddress = ColOffSet[Y - Radius - 1];
			//NextAddress = ColOffSet[Y + Radius];
			#pragma omp parallel for 
			for (X = 0; X < Width; X++) ColSum[X] = ColSum[X] - Data[ColOffSet[Y - Radius - 1] + X] + Data[ColOffSet[Y + Radius] + X];
			//for (X = 0; X < Width; X++) ColSum[X] = ColSum[X] - Data[LastAddress++] + Data[NextAddress++];
		}
		//Pointer = Result + Width * Y;
		
		for (Z = -Radius, Sum = 0; Z <= Radius; Z++) Sum += ColSum[RowOffset[Z]];
		Result[Width * Y] = Sum * InvAmount;

		//#pragma omp parallel for default(none) /*reduction(-: Sum)*/ private(X, Sum) shared(Y, Radius, Width, ColSum, RowOffset, Result, InvAmount)
		for (X = 1; X < Width; X++)
		{
			/*if(X == 0)
			{
				Sum = 0;
				for (Z = -Radius; Z <= Radius; Z++) Sum += ColSum[RowOffset[Z]];
			}
			else*/
				Sum = Sum - ColSum[RowOffset[X - Radius - 1]] + ColSum[RowOffset[X + Radius]];
			//Sum = Sum + (ColSum[RowOffset[X + Radius]] - ColSum[RowOffset[X - Radius - 1]]);

			Result[Width * Y + X] = Sum * InvAmount;
		}
	}

	//cout << Sum << endl; 

	RowOffset -= Radius;
	ColOffSet -= Radius;

	free(ColOffSet);
	free(RowOffset);
	free(ColSum);

	return Result;
}

float* GuidedFilterGray(float* I, float* P, int Radius, float eps, int Width, int Height)
{
	int Y;
	float CovIP, VarI,T,TT;
	float* MeanI, *MeanP, *MeanIP, *MeanII, *MeanA, *MeanB;
	float* A = (float*)malloc( Width * Height * (sizeof(float)));      //  Equation (5) in the paper;
	float* B = (float*)malloc( Width * Height * (sizeof(float)));      //  Equation (6) in the paper;
	float* Q = (float*)malloc( Width * Height * (sizeof(float)));
	float* IPI = (float*)malloc( Width * Height * (sizeof(float)));   
	float* II = (float*)malloc( Width * Height * (sizeof(float)));

	MeanP = BoxBlurGray(P, Width, Height, Radius);
	#pragma omp parallel for
	for (Y = 0; Y < Width * Height; Y++) II[Y] = I[Y] * I[Y];
	MeanII = BoxBlurGray(II, Width, Height, Radius);
	MeanI = BoxBlurGray(I, Width, Height, Radius);
	#pragma omp parallel for
	for (Y = 0; Y < Width * Height; Y++) IPI[Y] = I[Y] * P[Y];
	MeanIP = BoxBlurGray(IPI, Width, Height, Radius);
	#pragma omp parallel for private(T, TT, CovIP, VarI)
	for (Y = 0; Y < Width * Height; Y++)
	{
		T = MeanI[Y];
		TT = MeanP[Y];
		CovIP = MeanIP[Y] - T * TT;
		VarI = MeanII[Y] - T * T;
		A[Y] = CovIP / (VarI + eps);
		B[Y] = TT - (A[Y] * T);
	}

	MeanA = BoxBlurGray(A, Width, Height, Radius);
	MeanB = BoxBlurGray(B, Width, Height, Radius);

	#pragma omp parallel for
	for (Y = 0; Y < Width * Height; Y++) Q[Y] = MeanA[Y] * I[Y] + MeanB[Y];                             //  Equation (8) in the paper;      

	free(MeanI);
	free(MeanP);
	free(MeanIP);
	free(MeanII);
	free(MeanA);
	free(MeanB);
	free(IPI);
	free(II);
	free(A);
	free(B);

	return Q;
}

void HazeRemovalUseDarkChannelPrior(unsigned char * Src,unsigned char * Dest,int Width,int Height,int Stride, int Radius ,int GuideRadius, int MaxAtom, float Omega,float Epsilon,float T0 )
{
	const float Inv255 = 1.0f / 255;
	int X, Y, Min;
	int Sum, Value,Threshold = 0;
	int SumR = 0, SumG = 0, SumB = 0, AtomR, AtomB, AtomG, Amount = 0;
	unsigned char *ImgPt, *DarkPt,*Pt;
	float* SP,*Q;
	float Transmission;

	unsigned char* DarkChannel = (unsigned char*)malloc(Width * Height*sizeof(unsigned char));				
	int* Histgram = (int*)calloc(256 , sizeof(int));														
	float* I = (float*)malloc( Width * Height * sizeof(float));
	float* P = (float*)malloc( Width * Height * sizeof(float));

	DarkPt = DarkChannel;
	ImgPt = Src;
	#pragma omp parallel for default(none)  private(Min,Y,X) shared(DarkPt, Height, Width, ImgPt, Stride)
	for (Y = 0; Y < Height; Y++)
	{
		for (X = 0; X < Width; X++) {
			Min = ImgPt[Y * Stride + X * 3];
			if (Min > ImgPt[Y * Stride + X * 3 + 1]) Min = ImgPt[Y * Stride + X * 3 + 1];
			if (Min > ImgPt[Y * Stride + X * 3 + 2]) Min = ImgPt[Y * Stride + X * 3 + 2];
			DarkPt[Y * Width + X] = Min;										
		}
	}
	MinFilter(DarkChannel, Width, Height, Radius);				

	
	#pragma omp parallel for// reduction(+: Histgram[DarkChannel])
	for (Y = 0; Y < Width * Height; Y++) Histgram[DarkChannel[Y]]++;
	for (Y = 255, Sum = 0; Y >= 0; Y--)
	{
		Sum += Histgram[Y];
		if (Sum > Height * Width * 0.01)
		{
			Threshold = Y;										
			break;
		}
	}
	AtomB = 0; AtomG = 0; AtomR = 0;

	DarkPt = DarkChannel;
	ImgPt = Src;
	#pragma omp parallel for default(none) private(X, Y) shared(DarkPt, ImgPt, Height, Width, Stride, Threshold) reduction(+: SumB, SumG, SumR, Amount)
	for (Y = 0; Y < Height; Y++)
	{
		for (X = 0; X < Width; X++)
		{
			if (DarkPt[Y * Width + X] >= Threshold)													
			{
				SumB += ImgPt[Y * Stride + X * 3];
				SumG += ImgPt[Y * Stride + X * 3 + 1];
				SumR += ImgPt[Y * Stride + X * 3 + 2];
				Amount++;
			}
		}
	}

	//cout << Amount << endl;
	AtomB = SumB / Amount;
	AtomG = SumG / Amount;
	AtomR = SumR / Amount;
	if (AtomB > MaxAtom) AtomB = MaxAtom;						
	if (AtomG > MaxAtom) AtomG = MaxAtom;
	if (AtomR > MaxAtom) AtomR = MaxAtom;

	

	Omega = Omega * 255 / ((AtomB + AtomG * 2 + AtomR) / 4);       
	#pragma omp parallel for private(Value)
	for (Y = 0; Y < Width * Height; Y++)
	{
		Value = 255 - DarkChannel[Y] * Omega;
		if (Value > 255)
			Value = 255;
		else if (Value < 0)
			Value = 0;
		DarkChannel[Y] = Value;
	}

	
	#pragma omp parallel for
	for (Y = 0; Y < Width * Height; Y++) P[Y] = DarkChannel[Y] * Inv255;				

	SP = I;
	ImgPt = Src;
	#pragma omp parallel for default(none) private(X, Y) shared(Height, Width, SP, ImgPt, Stride)
	for (Y = 0; Y < Height; Y++)
	{
		for (X = 0; X < Width; X++)
		{

			SP[Y * Width + X] = (ImgPt[Y * Stride + X * 3] + (ImgPt[Y * Stride + X * 3 + 1] << 1) + ImgPt[Y * Stride + X * 3 + 2]) * (Inv255 * 0.25F);		
		}
	}

	Q = GuidedFilterGray(I, P,GuideRadius, Epsilon, Width, Height);

	SP = Q;
	ImgPt = Dest;
	Pt = Src; 
	#pragma omp parallel for default(none) private(Transmission, Value, X, Y) shared(Height, Width, SP, ImgPt, Pt, Stride, T0, AtomB, AtomG, AtomR)
	for (Y = 0; Y < Height; Y++)
	{
		for (X = 0; X < Width; X++)
		{
			Transmission = SP[Y * Width + X];
			if (Transmission < T0) Transmission = T0;
			Transmission = 1.0F / Transmission;
			Value = (Pt[Y * Stride + X * 3] - AtomB) * Transmission + AtomB;
			if (Value > 255)
				Value = 255;
			else if (Value < 0)
				Value = 0;
			ImgPt[Y * Stride + X * 3] = Value;
			Value = (Pt[Y * Stride + X * 3 + 1] - AtomG) * Transmission + AtomG;
			if (Value > 255)
				Value = 255;
			else if (Value < 0)
				Value = 0;
			ImgPt[Y * Stride + X * 3 + 1] = Value;
			Value = (Pt[Y * Stride + X * 3 + 2] - AtomR) * Transmission + AtomR;
			if (Value > 255)
				Value = 255;
			else if (Value < 0)
				Value = 0;
			ImgPt[Y * Stride + X * 3 + 2] = Value;
		}
	}

	free(Q);
	free(P); 
	free(I);
	free(Histgram);
	free(DarkChannel);

}

int main(int argc, char ** argv)
{
	if (argc != 2)
	{
		printf("Usage: ./HazeRemoval ImageFile\n");
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
	//Mat darkchannel(src.rows, src.cols, CV_8UC3);
	//Mat transmission(src.rows, src.cols, CV_8UC3);

	//t1 = get_seconds();

	//printf("%d %d", src.rows, src.cols);
	cout << "Height = " << src.rows << endl;
	cout << "Width = " << src.cols << endl;

	//BPP(Bits Per Pixel) = 24, 3 bytes per pixel.
	//Stride = width * bpp  ==>  Bits per row.
	//Stride += 31  ==>  round up to next 32-bit boundary.
	//Stride /= 32  ==>  DWORDs(1 DWORDs = 4 Bytes) per row.
	//Stride *= 4   ==>  bytes per row.  
	int STRIDE = 4 * ((src.cols * 24 + 31) / 32);

	cout << "Stride = " << STRIDE << endl;

	unsigned char *srcData;
	unsigned char *destData;
	//unsigned char *darkchannelData;
	//unsigned char *transmissionData;

	srcData = src.data;
	destData = dest.data;

	//t1 = get_seconds();

	HazeRemovalUseDarkChannelPrior(srcData, destData, src.cols, src.rows, STRIDE, Blocksize, GuideBlocksize, Airlight, OMEGA * 0.01f, EPSILON * 0.001f, TAU * 0.01f);

	t2 = get_seconds();

	dest.data = destData;

	//t2 = get_seconds();

	namedWindow("Source Image", WINDOW_AUTOSIZE);
	imshow("Source Image", src);
	//namedWindow("DarkChannel Image", WINDOW_AUTOSIZE);
	//imshow("DarkChannel Image", darkchannel);
	//namedWindow("Transmission Image", WINDOW_AUTOSIZE);
	//imshow("Transmission Image", transmission);
	namedWindow("HazeRemoval Image", WINDOW_AUTOSIZE);
	imshow("HazeRemoval Image", dest); 
	imwrite("HazeRemoval_Image2.jpg", dest);	

	waitKey(0);

	printf("\nElapsed Time = %18.6e\n", t2 - t1);

	return 0;
}
