#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;
int main(int argc, char *argv[]) 
{
	//  create 3x3 matrix mat1
	//int  a[9] = { 1, 0 , 0, 0, 1, 1, 0, 1, 1};
	Mat   mat1 = (Mat_<double>(3,3) << 1,2,0,0,0,6,7,8,9);
	// Mat to SparseMat
	SparseMat sparse_mat1(mat1);
	SparseMat sparse_mat2;
	sparse_mat2 = mat1;
	CvSparseMat cvsparse_mat1(mat1);
	//  SparseMat to Mat
	Mat mat2;
	sparse_mat1.copyTo(mat2);
	CV_Assert(mat1.data != mat2.data);

	// output result
	cout << "mat1" << mat1 << endl;
	cout << "mat2" << mat2 << endl;

	cout << "sparse_mat1:" << endl;
	//SparseMatConstIterator it1 = sparse_mat1.begin();
	SparseMatConstIterator it1;
	for (it1 = sparse_mat1.begin(); it1!=sparse_mat1.end(); ++it1)
		cout << it1.value<double>() << ", ";
	cout << endl;

	SparseMatConstIterator  it2 = sparse_mat2.begin();
	for(; it2 != sparse_mat2.end(); ++it2)
		cout << it2.value<double>() << ", ";
	cout << endl;     

	CvSparseMatIterator mat_iterator;
	CvSparseNode* node = cvInitSparseMatIterator(cvsparse_mat1, &mat_iterator);

	for(; node != 0; node = cvGetNextSparseNode(&mat_iterator))
	{
		int* idx = CV_NODE_IDX(sparse_mat1, node);
		float val = *(float*)CV_NODE_VAL(sparse_mat1, node);
		printf("%d \n", val);
	}
}
