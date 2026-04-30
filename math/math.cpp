#import <math.h>

voif matmult(const float* mat1, const float* mat2, float* output, int mat1rows, int m1colsm2rows, int mat2cols) {
	float* current = output;
	for (int i = 0; i < mat1rows; ++i) {
	for (int j = 0; j < mat2cols; ++j) {

	for (int k = 0; k < mat1rows; ++k) {
	//row i, col j of mat1, row j, col k of mat2
	//update = *(mat1 + i * mat1rows + j) * *(mat2+j*m1colsm2rows+k)
	//using row i col j, row j col k helps with cache as we loop over rows (cache miss) in outet loops and cols (cache hits) in inner loops
	//bettsr inpelemntstiln wohld be n^2.8 optjmization or chunkimg bht this is good enough
	current = output + i*mat1rows + k;
	*current += *(mat1 + i * mat1rows + j) * *(mat2+j*m1colsm2rows+k);
	}
	}
	}

}
