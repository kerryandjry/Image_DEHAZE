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
	int X, Y,I,  Stride, Index,Speed=0;
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
	for (Y = 0;Y<Height;Y++) memcpy(Expand + (Radius + Y) * StrideC + Radius,  Scan0 + Y * Stride, Stride);
	memset( ColMinValue, 255,2 * Radius + Width );
	ColMinValue+=Radius;

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

	for (X = 0;X<Width;X++)
	{
		MinValue = ColMinValue[X - Radius];
		for (I =  X - Radius + 1;I<= X + Radius;I++)
			if ( MinValue > ColMinValue[I])  MinValue = ColMinValue[I];
		Scan0[Speed] = MinValue;
		Speed = Speed + 1;
	}       

	Temp1 = (Radius + Radius) * StrideC + Radius;
	Temp2 = -StrideC + Radius;

	for (Y = 1;Y<Height;Y++)
	{
		Speed = Stride * Y;
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
			Scan0[Speed] = MinValue;
			Speed = Speed + 1;
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
			for (X = 0; X < Width; X++)
			{
				for (Z = -Radius, Sum = 0; Z <= Radius; Z++) Sum += Data[ColOffSet[Z] + X];
				ColSum[X] = Sum;
			}
		}
		else
		{
			LastAddress = ColOffSet[Y - Radius - 1];
			NextAddress = ColOffSet[Y + Radius];
			for (X = 0; X < Width; X++) ColSum[X] = ColSum[X] - Data[LastAddress++] + Data[NextAddress++];
		}
		Pointer = Result + Width * Y;
		for (X = 0; X < Width; X++)
		{
			if (X == 0)
				for (Z = -Radius, Sum = 0; Z <= Radius; Z++) Sum += ColSum[RowOffset[Z]];
			else
				Sum = Sum - ColSum[RowOffset[X - Radius - 1]] + ColSum[RowOffset[X + Radius]];
			*Pointer = Sum * InvAmount;
			Pointer++;
		}
	}
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
	for (Y = 0; Y < Width * Height; Y++) II[Y] = I[Y] * I[Y];
	MeanII = BoxBlurGray(II, Width, Height, Radius);
	MeanI = BoxBlurGray(I, Width, Height, Radius);
	for (Y = 0; Y < Width * Height; Y++) IPI[Y] = I[Y] * P[Y];
	MeanIP = BoxBlurGray(IPI, Width, Height, Radius);
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

	Mat darkchannel(Height, Width, CV_8UC3);

	unsigned char* DarkChannel = (unsigned char*)malloc(Width * Height*sizeof(unsigned char));				
	int* Histgram = (int*)calloc(256 , sizeof(int));														
	float* I = (float*)malloc( Width * Height * sizeof(float));
	float* P = (float*)malloc( Width * Height * sizeof(float));

	

	for (Y = 0, DarkPt = DarkChannel; Y < Height; Y++)
	{
		ImgPt = Src + Y * Stride;
		for (X = 0; X < Width; X++)
		{
			Min = *ImgPt;
			if (Min > *(ImgPt + 1)) Min = *(ImgPt + 1);
			if (Min > *(ImgPt + 2)) Min = *(ImgPt + 2);
			*DarkPt = Min;										
			ImgPt += 3;
			DarkPt++;
		}
	}
	MinFilter(DarkChannel, Width, Height, Radius);				

	

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
	for (Y = 0, DarkPt = DarkChannel; Y < Height; Y++)
	{
		ImgPt = Src + Y * Stride;
		for (X = 0; X < Width; X++)
		{
			if (*DarkPt >= Threshold)													
			{
				SumB += *ImgPt;
				SumG += *(ImgPt + 1);
				SumR += *(ImgPt + 2);
				Amount++;
			}
			ImgPt += 3;
			DarkPt++;
		}
	}
	AtomB = SumB / Amount;
	AtomG = SumG / Amount;
	AtomR = SumR / Amount;
	if (AtomB > MaxAtom) AtomB = MaxAtom;						
	if (AtomG > MaxAtom) AtomG = MaxAtom;
	if (AtomR > MaxAtom) AtomR = MaxAtom;

	

	Omega = Omega * 255 / ((AtomB + AtomG * 2 + AtomR) / 4);       
	for (Y = 0; Y < Width * Height; Y++)
	{
		Value = 255 - DarkChannel[Y] * Omega;
		if (Value > 255)
			Value = 255;
		else if (Value < 0)
			Value = 0;
		DarkChannel[Y] = Value;
	}

	

	for (Y = 0; Y < Width * Height; Y++) P[Y] = DarkChannel[Y] * Inv255;				
	for (Y = 0, SP = I; Y < Height; Y++)
	{
		ImgPt = Src + Y * Stride;
		for (X = 0; X < Width; X++)
		{
			*SP = (*ImgPt + (*(ImgPt + 1) << 1) + *(ImgPt + 2)) * (Inv255 * 0.25F);		
			ImgPt += 3;
			SP++;
		}
	}

	Q = GuidedFilterGray(I, P,GuideRadius, Epsilon, Width, Height);

	 

	for (Y = 0, SP = Q; Y < Height; Y++)
	{
		ImgPt = Dest + Y * Stride;
		Pt = Src + Y * Stride;
		for (X = 0; X < Width; X++)
		{
			Transmission = *SP;
			if (Transmission < T0) Transmission = T0;
			Transmission = 1.0F / Transmission;
			Value = (*Pt - AtomB) * Transmission + AtomB;
			if (Value > 255)
				Value = 255;
			else if (Value < 0)
				Value = 0;
			*ImgPt = Value;
			Value = (*(Pt + 1) - AtomG) * Transmission + AtomG;
			if (Value > 255)
				Value = 255;
			else if (Value < 0)
				Value = 0;
			*(ImgPt + 1) = Value;
			Value = (*(Pt + 2) - AtomR) * Transmission + AtomR;
			if (Value > 255)
				Value = 255;
			else if (Value < 0)
				Value = 0;
			*(ImgPt + 2) = Value;
			ImgPt += 3;
			Pt+=3;
			SP++;
		}
	}

	darkchannel.data = DarkChannel;
	namedWindow("DarkChannel Image", WINDOW_AUTOSIZE);
	imshow("DarkChannel Image", darkchannel);
	//waitKey(0);

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

	HazeRemovalUseDarkChannelPrior(srcData, destData, src.cols, src.rows, STRIDE, Blocksize, GuideBlocksize, Airlight, OMEGA * 0.01f, EPSILON * 0.001f, TAU * 0.01f);

	dest.data = destData;

	t2 = get_seconds();

	namedWindow("Source Image", WINDOW_AUTOSIZE);
	imshow("Source Image", src);
	//namedWindow("DarkChannel Image", WINDOW_AUTOSIZE);
	//imshow("DarkChannel Image", darkchannel);
	//namedWindow("Transmission Image", WINDOW_AUTOSIZE);
	//imshow("Transmission Image", transmission);
	namedWindow("HazeRemoval Image", WINDOW_AUTOSIZE);
	imshow("HazeRemoval Image", dest); 
	
	waitKey(0);

	printf("\nElapsed Time = %18.6e\n", t2 - t1);

	return 0;
}
