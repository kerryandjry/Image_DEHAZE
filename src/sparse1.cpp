#include <stdio.h>
#include <opencv2/opencv.hpp>

main()
{ 
	float A[] = { 2, 0, 0, 0, 4, 0, 0 , 5 };
	CvMat array = cvMat(3, 3, CV_32FC1, &A);
	double sum;
	int i, dims = cvGetDims(array);
	CvSparseMatIterator mat_iterator;
	CvSparseNode* node = cvInitSparseMatIterator(array, &mat_iterator);

	for(; node != 0; node = cvGetNextSparseNode(&mat_iterator ))
	{
		/* get pointer to the element indices */
		int* idx = CV_NODE_IDX(array, node);
		/* get value of the element (assume that the type is CV_32FC1) */
		float val = *(float*)CV_NODE_VAL(array, node);
		printf("(");

		for(i = 0; i < dims; i++ )
			printf("%4d%s", idx[i], i < dims - 1 "," : "): ");

		printf("%g\n", val);
		sum += val;
	}

	printf("\nTotal sum = %g\n", sum);
}
