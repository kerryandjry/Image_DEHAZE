/*agnarok @ ZJU CAD&CG
va
*  2011.3.1
*  Implementation of <Single Image Haze Removal Using Dark Channel Prior> @ CVPR2009
*  Using OpenCV 2.1 with VC 9.0
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "/usr/local/include/opencv2/opencv.hpp"
//#include <opencv.hpp>
#include <highgui.h>
#include <string>
#include <iostream>
#include "mytime.h"
//regularizing parameter for Laplacian matrix
#define EPSILON 0.01f
//better be an odd number
#define PATCH 15
//top 0.1% brightest pixels in the dark channel.
#define ATOMSPHERERATIO 0.001f
//omega to relieve aerial perspective
#define OMEGA 0.95f
//lower bound t0 reciprocal
#define T0RECI 10
//windows size for Laplacian Matting
#define WNDSIZE 3
//lamda for Laplacian matrix
#define LAMDA 0.0001f
//PCG algorithm max iteration
#define ITERMAX 1000
//PCG algorithm error tolerance square
#define TOLERANCE 0.001f

using namespace std;

//Return minimal RGB Color Channel
uchar GetMiniRGB(uchar *ptr)
{
	if (ptr[0] < ptr[1])
	{
		if (ptr[0] < ptr[2])
		{
			return ptr[0];
		}
		else
		{
			return ptr[2];
		}
	}
	else
	{
		if (ptr[1] < ptr[2])
		{
			return ptr[1];
		}
		else
		{
			return ptr[2];
		}
	}

	return ptr[0];
}

//Marcel van Herk's fast algorithm to calculate Dark Channel
void CalcDarkChannel(IplImage *src, uchar **darkChannel, int *hiWidth, int *hiHeight)
{
	//horizon patch & reverse
	uchar **horizonPatch = new uchar *[src->height];
	uchar **horizonPatchReverse = new uchar *[src->height];
        //double  t1 = get_seconds();
	#pragma omp parallel for
	for (int i = 0; i < src->height; ++i)
	{
		horizonPatch[i] = new uchar[src->width];
		horizonPatchReverse[i] = new uchar[src->width];
	}
	int patchInterval = PATCH < src->width ? PATCH : src->width;

	//for row
	for (int height = 0; height < src->height; ++height)
	{
		uchar *ptr = (uchar *)(src->imageData + height * src->widthStep);
		//Next Time
		for (int width = 0; width < src->width; width += patchInterval)
		{
			int patchStep = width;
			//point to the End of Patch
			uchar *ptrRev;
			if (width + patchInterval >= src->width)
			{
				ptrRev = ptr + 3 * (src->width - 1 - patchStep);
				horizonPatchReverse[height][src->width - 1] = GetMiniRGB(ptrRev);
			}
			else
			{
				ptrRev = ptr + 3 * (patchInterval - 1);
				horizonPatchReverse[height][width * 2 + patchInterval - patchStep - 1] = GetMiniRGB(ptrRev);
			}
			//get first RGB
			horizonPatch[height][patchStep] = GetMiniRGB(ptr);
			ptr += 3;
			//get last RGB
			ptrRev -= 3;
			//patch step add
			++patchStep;

			//get patch others RGB, & compare them with Previous
			while (patchStep < width + patchInterval && patchStep < src->width)
			{
				horizonPatch[height][patchStep] = GetMiniRGB(ptr);
				ptr += 3;

				//whether less than previous one
				if (horizonPatch[height][patchStep] > horizonPatch[height][patchStep - 1])
				{
					horizonPatch[height][patchStep] = horizonPatch[height][patchStep - 1];
				}

				if (width + patchInterval >= src->width)
				{
					horizonPatchReverse[height][src->width - 1 - patchStep + width] = GetMiniRGB(ptrRev);
					if (horizonPatchReverse[height][src->width - 1 - patchStep + width] >
						horizonPatchReverse[height][src->width - patchStep + width])
					{
						horizonPatchReverse[height][src->width - 1 - patchStep + width] =
							horizonPatchReverse[height][src->width - patchStep + width];
					}
				}
				else
				{
					horizonPatchReverse[height][width * 2 + patchInterval - patchStep - 1] = GetMiniRGB(ptrRev);
					if (horizonPatchReverse[height][width * 2 + patchInterval - patchStep - 1] >
						horizonPatchReverse[height][width * 2 + patchInterval - patchStep])
					{
						horizonPatchReverse[height][width * 2 + patchInterval - patchStep - 1] =
							horizonPatchReverse[height][width * 2 + patchInterval - patchStep];
					}
				}

				ptrRev -= 3;
				++patchStep;
			}
		}
	}
	//calculate the row minimal
	uchar **rowMin = new uchar *[src->height];
	#pragma omp parallel for
	for (int i = 0; i < src->height; ++i)
	{
		rowMin[i] = new uchar[src->width];
	}
        #pragma omp parallel for	
	for (int height = 0; height < src->height; ++height)
	{
		//beginning of the row
		for (int width = 0; width < PATCH / 2; ++width)
		{
			rowMin[height][width] = horizonPatch[height][width + PATCH / 2];
		}
		//middle
		for (int width = PATCH / 2; width < src->width - PATCH / 2; ++width)
		{
			rowMin[height][width] = horizonPatchReverse[height][width - PATCH / 2] < horizonPatch[height][width + PATCH / 2] ?
				horizonPatchReverse[height][width - PATCH / 2] : horizonPatch[height][width + PATCH / 2];
		}
		//end
		for (int width = src->width - PATCH / 2; width < src->width; ++width)
		{
			rowMin[height][width] = horizonPatchReverse[height][width - PATCH / 2];
		}
	}

	//vertical patch & reverse
	uchar **verticalPatch = new uchar *[src->height];
	uchar **verticalPatchReverse = new uchar *[src->height];
	#pragma omp parallel for
	for (int i = 0; i < src->height; ++i)
	{
		verticalPatch[i] = new uchar[src->width];
		verticalPatchReverse[i] = new uchar[src->width];
	}
	patchInterval = PATCH < src->height ? PATCH : src->height;

	//for column
	//Next Time
	for (int width = 0; width < src->width; ++width)
	{
		for (int height = 0; height < src->height; height += patchInterval)
		{
			int patchStep = height;
			//get order
			verticalPatch[patchStep][width] = rowMin[patchStep][width];
			//get reverse order
			if (height + patchInterval >= src->height)
			{
				verticalPatchReverse[src->height - 1][width] = rowMin[src->height - 1][width];
			}
			else
			{
				verticalPatchReverse[height * 2 + patchInterval - patchStep - 1][width] = rowMin[height * 2 + patchInterval - patchStep - 1][width];
			}
			//patch step add
			++patchStep;

			//get others patch min, & compare them with Previous
			while (patchStep < height + patchInterval && patchStep < src->height)
			{
				verticalPatch[patchStep][width] = rowMin[patchStep][width];

				//whether less than previous one
				if (verticalPatch[patchStep][width] > verticalPatch[patchStep - 1][width])
				{
					verticalPatch[patchStep][width] = verticalPatch[patchStep - 1][width];
				}

				if (height + patchInterval >= src->height)
				{
					verticalPatchReverse[src->height - 1 - patchStep + height][width] = rowMin[src->height - 1 - patchStep + height][width];
					if (verticalPatchReverse[src->height - 1 - patchStep + height][width] >
						verticalPatchReverse[src->height - patchStep + height][width])
					{
						verticalPatchReverse[src->height - 1 - patchStep + height][width] =
							verticalPatchReverse[src->height - patchStep + height][width];
					}
				}
				else
				{
					verticalPatchReverse[height * 2 + patchInterval - patchStep - 1][width] = rowMin[height * 2 + patchInterval - patchStep - 1][width];
					if (verticalPatchReverse[height * 2 + patchInterval - patchStep - 1][width] >
						verticalPatchReverse[height * 2 + patchInterval - patchStep][width])
					{
						verticalPatchReverse[height * 2 + patchInterval - patchStep - 1][width] =
							verticalPatchReverse[height * 2 + patchInterval - patchStep][width];
					}
				}

				++patchStep;
			}
		}
	}

	//calculate the final minimal patch
	uchar hiDC = 0;
	//Next Time
	for (int width = 0; width < src->width; ++width)
	{
		//beginning of the row
		for (int height = 0; height < PATCH / 2; ++height)
		{
			darkChannel[height][width] = verticalPatch[height + PATCH / 2][width];
			if (hiDC < darkChannel[height][width])
			{
				*hiWidth = width;
				*hiHeight = height;
				hiDC = darkChannel[height][width];
			}
		}
		//middle
		for (int height = PATCH / 2; height < src->height - PATCH / 2; ++height)
		{
			darkChannel[height][width] = verticalPatchReverse[height - PATCH / 2][width] < verticalPatch[height + PATCH / 2][width] ?
				verticalPatchReverse[height - PATCH / 2][width] : verticalPatch[height + PATCH / 2][width];
			if (hiDC < darkChannel[height][width])
			{
				*hiWidth = width;
				*hiHeight = height;
				hiDC = darkChannel[height][width];
			}
		}
		//end
		for (int height = src->height - PATCH / 2; height < src->height; ++height)
		{
			darkChannel[height][width] = verticalPatchReverse[height - PATCH / 2][width];
			if (hiDC < darkChannel[height][width])
			{
				*hiWidth = width;
				*hiHeight = height;
				hiDC = darkChannel[height][width];
			}
		}
	}
        //double t2 = get_seconds();
        //printf("\ntotal Darkchannel time = %f \n", t2-t1);
}

//Section 4.4 Estimating the Atmospheric Light
void AtmosphericLight(IplImage *src, int width, int height, uchar& asLight)
{
	// 	int ratioPercent = (int)(src->width * src->height * ATOMSPHERERATIO);
	// 	int index = 255;
	// 	while (ratioPercent > 0)
	// 	{
	// 		ratioPercent -= histogram[index--];
	// 	}
	// 	asLight = 255;
	// 	int tmpHist = histogram[255];
	// 	for (int idx = 254; idx > index + 1; --idx)
	// 	{
	// 		if (histogram[idx] > tmpHist)
	// 		{
	// 			asLight = idx;
	// 			tmpHist = histogram[idx];
	// 		}
	// 	}
	// 	if (histogram[index + 1] + ratioPercent > tmpHist)
	// 	{
	// 		asLight = index + 1;
	// 	}
	uchar *ptr = (uchar *)(src->imageData + height * src->widthStep + width * 3);
	asLight = ptr[0];
	if (asLight < ptr[1])
		asLight = ptr[1];
	if (asLight < ptr[2])
		asLight = ptr[2];
}

//Section 4.1 Estimating the Transmission
void Transmission(IplImage *src, uchar **darkChannel, uchar asLight, float **transmission)
{
	float light = 1.0f / asLight;
	
	#pragma omp parallel for
	for (int height = 0; height < src->height; ++height)
	{
		for (int width = 0; width < src->width; ++width)
		{
			transmission[height][width] = 1 - darkChannel[height][width] * light * OMEGA;
		}
	}
}

//determine whether i and j are in one window
bool iWNDj(int i, int j, int width)
{
	if (j + WNDSIZE * width <= i || j - WNDSIZE * width >= i)
	{
		return false;
	}
	for (int row = -WNDSIZE + 1; row < WNDSIZE; ++row)
	{
		int tmp = j + row * width;
		if (tmp < 0)
		{
			continue;
		}
		for (int col = -WNDSIZE + 1; col < WNDSIZE; ++col)
		{
			if (i == tmp + col)
			{
				//test points beside j are not in other row
				if (tmp / width == (tmp + col) / width)
				{
					return true;
				}
			}
		}
	}
	return false;
}

//To get Matrix from IplImage window
void GetMatFromImgWND(IplImage *src, int height, int width, int windowSize, CvMat *windowMat)
{
	//Next Time
	for (int row = 0; row < windowSize; ++row)
	{
		uchar *ptr = (uchar *)(src->imageData + (height + row) * src->widthStep + width * 3);
		for (int col = 0; col < windowSize; ++col)
		{
			CV_MAT_ELEM(*windowMat, float, row * windowSize + col, 0) = *ptr;
			++ptr;
			CV_MAT_ELEM(*windowMat, float, row * windowSize + col, 1) = *ptr;
			++ptr;
			CV_MAT_ELEM(*windowMat, float, row * windowSize + col, 2) = *ptr;
			++ptr;
		}
	}
}

//To get Matting Laplacian Matrix
//Refer to A. Levin <A Closed Form Solution to Natural Image Matting>
void GetLMatrix(CvSparseMat *LMat, IplImage *src)
{
	//U3 Identity matrix
	CvMat *U3 = cvCreateMat(3, 3, CV_32FC1);
	float param = EPSILON / (WNDSIZE * WNDSIZE);
	int a, b;

	//try copy U3 to B
	

	//#pragma omp parallel for private( b )
	for (a = 0; a < 3; ++a)
	{
		for (b = 0; b < 3; ++b)
		{
			if (a == b)
			{
				CV_MAT_ELEM(*U3, float, a, b) = param;
			}
			else
			{
				CV_MAT_ELEM(*U3, float, a, b) = 0;
			}
		}
	}

	printf("%d , %d\n", src->height, src->width);
	//#pragma omp parallel for
	for (int height = 0; height <= src->height - WNDSIZE; ++height)
	{
		//#pragma omp parallel for
		for (int width = 0; width <= src->width - WNDSIZE; ++width)
		{
			//printf("i=%d,j=%d\n",height,width);
			//for window's 9(window * window) * 3(RGB) matrix
			CvMat *windowsMat = cvCreateMat(WNDSIZE * WNDSIZE, 3, CV_32FC1);
			GetMatFromImgWND(src, height, width, WNDSIZE, windowsMat);
			//get covariance & mean matrix
			CvMat *preCal[1];
			preCal[0] = windowsMat;
			CvMat *meanTmp = cvCreateMat(1, WNDSIZE, CV_32FC1);
			CvMat *covTmp = cvCreateMat(WNDSIZE, WNDSIZE, CV_32FC1);
			cvCalcCovarMatrix((const CvArr **)preCal, WNDSIZE * WNDSIZE, covTmp,
				meanTmp, CV_COVAR_NORMAL + CV_COVAR_ROWS);
			CvMat *meanMatrix1 = cvCreateMat(WNDSIZE * WNDSIZE, 3, CV_32FC1), *meanMatrix2 = cvCreateMat(3, WNDSIZE * WNDSIZE, CV_32FC1);
			CvMat *covMatrix = cvCreateMat(WNDSIZE, WNDSIZE, CV_32FC1);
			//#pragma omp parallel for  firstprivate (windowsMat, preCal, meanTmp, covTmp, meanMatrix1, covMatrix ) lastprivate ( covMatrix, covTmp )
			for (int i = 0; i < 3; ++i)
			{
				for (int j = 0; j < 3; ++j)
				{
					CV_MAT_ELEM(*covMatrix, float, i, j) = CV_MAT_ELEM(*covTmp, float, j, i) / (WNDSIZE * WNDSIZE);
				}
			}

			//printf("%d\n", WNDSIZE);
			//#pragma omp parallel for  firstprivate (  meanMatrix1 ) lastprivate ( meanMatrix1, meanMatrix2 )
			for (int i = 0; i < WNDSIZE * WNDSIZE; ++i)
			{
				CV_MAT_ELEM(*meanMatrix1, float, i, 0) = CV_MAT_ELEM(*meanTmp, float, 0, 0);
				CV_MAT_ELEM(*meanMatrix1, float, i, 1) = CV_MAT_ELEM(*meanTmp, float, 0, 1);
				CV_MAT_ELEM(*meanMatrix1, float, i, 2) = CV_MAT_ELEM(*meanTmp, float, 0, 2);
				CV_MAT_ELEM(*meanMatrix2, float, 0, i) = CV_MAT_ELEM(*meanTmp, float, 0, 0);
				CV_MAT_ELEM(*meanMatrix2, float, 1, i) = CV_MAT_ELEM(*meanTmp, float, 0, 1);
				CV_MAT_ELEM(*meanMatrix2, float, 2, i) = CV_MAT_ELEM(*meanTmp, float, 0, 2);
			}
			cvReleaseMat(&meanTmp);
			cvReleaseMat(&covTmp);

			//get Ii & Ij matrix
			CvMat *Ij = cvCreateMat(3, WNDSIZE * WNDSIZE, CV_32FC1);
			//#pragma omp for
			for (int k = 0; k < WNDSIZE; ++k)
			{
				for (int m = 0; m < WNDSIZE; ++m)
				{
					CV_MAT_ELEM(*Ij, float, 0, k * WNDSIZE + m) = CV_MAT_ELEM(*windowsMat, float, k * WNDSIZE + m, 0);
					CV_MAT_ELEM(*Ij, float, 1, k * WNDSIZE + m) = CV_MAT_ELEM(*windowsMat, float, k * WNDSIZE + m, 1);
					CV_MAT_ELEM(*Ij, float, 2, k * WNDSIZE + m) = CV_MAT_ELEM(*windowsMat, float, k * WNDSIZE + m, 2);
				}
			}

			//get matrix multiply result
			CvMat *mat = cvCreateMat(WNDSIZE * WNDSIZE, WNDSIZE * WNDSIZE, CV_32FC1);
			cvSub(windowsMat, meanMatrix1, windowsMat);
			cvAdd(covMatrix, U3, covMatrix);
			cvInvert(covMatrix, covMatrix);
			cvSub(Ij, meanMatrix2, Ij);
			cvMatMul(windowsMat, covMatrix, windowsMat);
			cvMatMul(windowsMat, Ij, mat);

			cvReleaseMat(&windowsMat);
			cvReleaseMat(&Ij);
			cvReleaseMat(&meanMatrix1);
			cvReleaseMat(&meanMatrix2);
			cvReleaseMat(&covMatrix);

			//add to the L matrix
			for (int i = 0; i < WNDSIZE; ++i)
			{
				for (int j = 0; j < WNDSIZE; ++j)
				{
					for (int k = 0; k < WNDSIZE; ++k)
					{
						for (int m = 0; m < WNDSIZE; ++m)
						{
							int kronecker = (i * WNDSIZE + j) == (k * WNDSIZE + m) ? 1 : 0;
							float tmp = cvGetReal2D(LMat, (height + i) * src->width + width + j, (height + k) * src->width + width + m)
								+ kronecker - (CV_MAT_ELEM(*mat, float, i * WNDSIZE + j, k * WNDSIZE + m) + 1) / (WNDSIZE * WNDSIZE);
							cvSetReal2D(LMat, (height + i) * src->width + width + j, (height + k) * src->width + width + m, tmp);
						}
					}
				}
			}

			cvReleaseMat(&mat);
		}
	}
	printf("test\n");
	//release Matrix
	cvReleaseMat(&U3);
}

//Sparse Matrix Add
//without constraint check & src1 include src2's all nodes
void cvSparseAdd(CvSparseMat *src1, CvSparseMat *src2, CvSparseMat *dst)
{
	CvSparseMatIterator mat_iterator;
	int* idx;
	float val1, val2;

	//#pragma omp parallel for private(val1, val2, idx)
	//for(CvSparseNode* node = cvInitSparseMatIterator(src1, &mat_iterator); node != 0; node = cvGetNextSparseNode(&mat_iterator))
	for(CvSparseNode* node = cvInitSparseMatIterator(src1, &mat_iterator); node != 0; node++)
	{
		
		idx = CV_NODE_IDX(src1, node);
		val1 = *(float *)CV_NODE_VAL(src1, node);
		val2 = cvGetReal2D(src2, idx[0], idx[1]);
		if (fabs(val1 += val2) > 1e-10)
		{
			cvSetReal2D(dst, idx[0], idx[1], val1);
		}
	}
}

//Sparse Matrix Multiply Column Vector
//without constraint check
void cvSparseMul(CvSparseMat *src1, CvMat *src2, CvMat *dst)
{
	CvSparseMatIterator matIterator;
	int *idx;
	float val;

	//#pragma omp parallel for private(malIterator, idx, val)
	for (CvSparseNode* node = cvInitSparseMatIterator(src1, &matIterator); node != 0; node = cvGetNextSparseNode(&matIterator))
	{
		idx = CV_NODE_IDX(src1, node);
		val = *(float*)CV_NODE_VAL(src1, node);

		CV_MAT_ELEM(*dst, float, idx[0], 0) += val * CV_MAT_ELEM(*src2, float, idx[1], 0);
	}
}

//Preconditioned Conjugate Gradient Algorithm
//Implementation of J. R. Shewchuk <An Introduction to the Conjugate Gradient Method Without the Agonizing Pain> p51
void PCGAlgo(CvSparseMat *A, CvMat *b, CvMat *x, CvSparseMat *MInvert, int iterMax, float err, IplImage *src)
{
	int iter = 0;
	CvMat *r = cvCreateMat(src->width * src->height, 1, CV_32FC1);
	cvZero(r);
	cvSparseMul(A, x, r);
	cvSub(b, r, r);
	CvMat *d = cvCreateMat(src->width * src->height, 1, CV_32FC1);
	cvZero(d);
	cvSparseMul(MInvert, r, d);
	CvMat *rTrans = cvCreateMat(1, src->width * src->height, CV_32FC1);
	cvTranspose(r, rTrans);
	CvMat *thNew = cvCreateMat(1, 1, CV_32FC1);
	cvMatMul(rTrans, d, thNew);
	float thOld, thNewNum = CV_MAT_ELEM(*thNew, float, 0, 0), th0Num = thNewNum;

	int i = 0;
	//iteration
	while (i < iterMax && thNewNum > err * th0Num)
	{
		CvMat *q = cvCreateMat(src->width * src->height, 1, CV_32FC1);
		cvZero(q);
		cvSparseMul(A, d, q);
		float den = 0;
		//!!#pragma omp parallel for reduction(+:den)
		for (int p = 0; p < src->width * src->height; ++p)
		{
			den += CV_MAT_ELEM(*d, float, p, 0) * CV_MAT_ELEM(*q, float, p, 0);
		}
		float alfa = thNewNum / den;
		#pragma omp parallel for
		for (int k = 0; k < src->width * src->height; ++k)
		{
			CV_MAT_ELEM(*x, float, k, 0) = CV_MAT_ELEM(*x, float, k, 0) + alfa * CV_MAT_ELEM(*d, float, k, 0);
		}
		if (i % 50 == 0)
		{
			cvZero(r);
			cvSparseMul(A, x, r);
			cvSub(b, r, r);
		}
		else
		{
			for (int k = 0; k < src->width * src->height; ++k)
			{
				CV_MAT_ELEM(*r, float, k, 0) = CV_MAT_ELEM(*r, float, k, 0) - alfa * CV_MAT_ELEM(*q, float, k, 0);
			}
		}
		cvReleaseMat(&q);
		CvMat *s = cvCreateMat(src->width * src->height, 1, CV_32FC1);
		cvZero(s);
		cvSparseMul(MInvert, r, s);
		thOld = thNewNum;
		cvTranspose(r, rTrans);
		cvMatMul(rTrans, s, thNew);
		thNewNum = CV_MAT_ELEM(*thNew, float, 0, 0);
		float beta = thNewNum / thOld;
		//#pragma omp parallel for
		for (int k = 0; k < src->width * src->height; ++k)
		{
			CV_MAT_ELEM(*d, float, k, 0) = CV_MAT_ELEM(*s, float, k, 0) + beta * CV_MAT_ELEM(*d, float, k, 0);
		}
		cvReleaseMat(&s);
		++i;
	}

	cvReleaseMat(&r);
	cvReleaseMat(&d);
	cvReleaseMat(&rTrans);
	cvReleaseMat(&thNew);
}

//Section 4.2 Soft Matting
void SoftMatting(IplImage *src, float **trans)
{
	int demSize[2];
	demSize[0] = src->width * src->height;
	demSize[1] = src->width * src->height;
	CvSparseMat *LMat = cvCreateSparseMat(2, demSize, CV_32FC1);
	GetLMatrix(LMat, src);

	//Identity matrix
	CvSparseMat *U = cvCreateSparseMat(2, demSize, CV_32FC1);
	for (int k = 0; k < src->width * src->height; ++k)
	{
		cvSetReal2D(U, k, k, LAMDA);
	}

	// Ax = b
	CvSparseMat *A = cvCreateSparseMat(2, demSize, CV_32FC1);
	cvSparseAdd(LMat, U, A);
	CvMat *b = cvCreateMat(src->width * src->height, 1, CV_32FC1);
	for (int k = 0; k < src->height; ++k)
	{
		for (int m = 0; m < src->width; ++m)
		{
			CV_MAT_ELEM(*b, float, k * src->width + m, 0) = LAMDA * trans[k][m];
		}
	}

	cvReleaseSparseMat(&LMat);
	cvReleaseSparseMat(&U);

	//x0 = [0]
	CvMat *x0 = cvCreateMat(src->width * src->height, 1, CV_32FC1);
	//#pragma omp parallel for firstprivate(x0) lastprivate(x0)
	for (int k = 0; k < src->width * src->height; ++k)
	{
		CV_MAT_ELEM(*x0, float, k, 0) = 0;
	}
	//Preconditioner is using diagonal preconditioning or Jacobi preconditioning
	//here invert it
	CvSparseMat *MInvert = cvCreateSparseMat(2, demSize, CV_32FC1);
	for (int k = 0; k < src->width * src->height; ++k)
	{
		cvSetReal2D(MInvert, k, k, 1.0f / cvGetReal2D(A, k, k));
	}

	//Preconditioned Conjugate Gradient (PCG) method
	PCGAlgo(A, b, x0, MInvert, ITERMAX, TOLERANCE, src);

	cvReleaseSparseMat(&A);
	cvReleaseMat(&b);
	cvReleaseSparseMat(&MInvert);

	//x0 is the refined transmission
	//#pragma omp parallel for firstprivate(x0) lastprivate(x0)
	for (int k = 0; k < src->height; ++k)
	{
		for (int m = 0; m < src->width; ++m)
		{
			trans[k][m] = CV_MAT_ELEM(*x0, float, k * src->width + m, 0);
		}
	}
	cvReleaseMat(&x0);
}

//Section 4.3 Recovering the Scene Radiance
void RecoverSceneRadiance(IplImage *src, float **transmission, uchar asLight, IplImage *des, CvMat *transMat)
{
	for (int height = 0; height < src->height; ++height)
	{
		uchar *srcPtr = (uchar *)(src->imageData + height * src->widthStep);
		uchar *desPtr = (uchar *)(des->imageData + height * des->widthStep);
		//#pragma omp parallel for 
		for (int width = 0; width < src->width; ++width)
		{
			CV_MAT_ELEM(*transMat, uchar, height, width) = transmission[height][width] * 255;
			float maxReci = T0RECI < 1.0f / transmission[height][width] ? T0RECI : 1.0f / transmission[height][width];
			desPtr[width * 3] = (srcPtr[width * 3] - asLight) * maxReci + asLight;
			desPtr[width * 3 + 1] = (srcPtr[width * 3 + 1] - asLight) * maxReci + asLight;
			desPtr[width * 3 + 2] = (srcPtr[width * 3 + 2] - asLight) * maxReci + asLight;
		}
	}
}


int main( int argc, char ** argv) 
{
	char picname[1][50] = {"./Dehazed.jpg"};
        if (argc!=2) {
            printf("Usage: ./dehaze_new jpgfile\n");
            exit(1);
        }

        strcpy(picname[0],argv[1]);

	double t1, t2;

	for (int n = 0; n < 1; ++n)
	{
		printf("loadimg\n");

		t1 = get_seconds();
		IplImage *src = cvLoadImage(picname[n]);
		//Allocate the 2D dark channel
		uchar **darkChannel = new uchar *[src->height];
		//#pragma omp parallel for
		for (int i = 0; i < src->height; ++i)
		{
			darkChannel[i] = new uchar[src->width];
		}
		// 		//Allocate the dark channel histogram
		// 		uchar histogramDC[256];
		// 		memset(histogramDC, 0, sizeof(histogramDC));
		int hiWidth, hiHeight;

		CalcDarkChannel(src, darkChannel, &hiWidth, &hiHeight);

		//AtmosphericLight
		uchar asLight;

		//		AtmosphericLight(src, hiWidth, hiHeight, asLight);
		asLight = 255;

		//Allocate the transmission
		float **transmission = new float *[src->height];
		//#pragma omp parallel for
		for (int q = 0; q < src->height; ++q)
		{
			transmission[q] = new float[src->width];
		}

		Transmission(src, darkChannel, asLight, transmission);

		SoftMatting(src, transmission);

		//dehaze image
		CvMat *transMat = cvCreateMat(src->height, src->width, CV_8UC1);
		IplImage *dehazeImage = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);

		RecoverSceneRadiance(src, transmission, asLight, dehazeImage, transMat);

		t2 = get_seconds();

		cvNamedWindow("Source Image", CV_WINDOW_AUTOSIZE);
		cvShowImage("Source Image", src);
		cvNamedWindow("Transmission Map", CV_WINDOW_AUTOSIZE);
		cvShowImage("Transmission Map", transMat);	
		cvSaveImage("Trans.jpg",transMat);
		cvNamedWindow("omp-Dehazed", CV_WINDOW_AUTOSIZE);
		cvShowImage("omp-Dehazed", dehazeImage);
                cvSaveImage("omp-Dehazed.jpg",dehazeImage);
		if (cvWaitKey() == 27)
			break;
		for (int i = 0; i < src->height; ++i)
		{
			delete[] darkChannel[i];
			delete[] transmission[i];
		}
		delete[] darkChannel;
		delete[] transmission;
		cvReleaseImage(&src);
		cvReleaseMat(&transMat);
		cvReleaseImage(&dehazeImage);
		cvDestroyWindow("Source Image");
		cvDestroyWindow("Transmission Map");
		cvDestroyWindow("Dehaze Image");
	}
	

	printf("\nExecute Time = %lf\n", (t2 - t1));

	return 0;
}

