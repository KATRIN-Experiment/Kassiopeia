#ifndef KGMatrixOperations_HH__
#define KGMatrixOperations_HH__

/*
*
*@file KGMatrixOperations.hh
*@class KGMatrixOperations
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov 13 12:22:24 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

#include "KGLinearAlgebraDefinitions.hh"

namespace KGeoBag
{

//allocation/deallocation
kg_matrix* kg_matrix_alloc(unsigned int nrows, unsigned int ncolumns);
kg_matrix* kg_matrix_calloc(unsigned int nrows, unsigned int ncolumns);
kg_sparse_matrix* kg_sparse_matrix_alloc(unsigned int nrows, unsigned int ncolumns, unsigned int n_elements);

void kg_matrix_free(kg_matrix* m);
void kg_sparse_matrix_free(kg_sparse_matrix* m);

//access
double kg_matrix_get(const kg_matrix* m, unsigned int i, unsigned int j);
void kg_matrix_set(kg_matrix* m, unsigned int i, unsigned int j, double x);
void kg_matrix_set_zero(kg_matrix* m);
void kg_matrix_set_identity(kg_matrix* m);
void kg_matrix_set(const kg_matrix* src, kg_matrix* dest);

double kg_sparse_matrix_get(const kg_sparse_matrix* m, unsigned int i, unsigned int j);
void kg_sparse_matrix_set(kg_sparse_matrix* m, unsigned int i, unsigned int j, unsigned int element_index, double x);


//operations
void kg_matrix_transpose(const kg_matrix* in, kg_matrix* out);
void kg_matrix_multiply(const kg_matrix* A, const kg_matrix* B, kg_matrix* C);  // C = A*B
void kg_matrix_multiply_with_transpose(bool transposeA, bool transposeB, const kg_matrix* A, const kg_matrix* B,
                                       kg_matrix* C);

void kg_matrix_sub(kg_matrix* a, const kg_matrix* b);
void kg_matrix_add(kg_matrix* a, const kg_matrix* b);
void kg_matrix_scale(kg_matrix* a, double scale_factor);

//computes the euler angles associated with the orthogonal 3x3 matrix R
//Assumes the rotation is given with the Z-Y'-Z'' convention
void kg_matrix_euler_angles_ZYZ(const kg_matrix* R, double& alpha, double& beta, double& gamma, double& tol);
void kg_matrix_from_euler_angles_ZYZ(kg_matrix* R, double alpha, double beta, double gamma);

void kg_matrix_from_axis_angle(kg_matrix* R, double angle, const kg_vector* axis);
void kg_matrix_from_axis_angle(kg_matrix* R, double cos_angle, double sin_angle, const kg_vector* axis);

//computes the singular value decomposition of the matrix A = U*diag(S)*V^T
void kg_matrix_svd(const kg_matrix* A, kg_matrix* U, kg_vector* S, kg_matrix* V);

//given the singular value decomposition of the matrix A = U*diag(S)*V^T
// this function solves the equation Ax = b in the least squares sense
void kg_matrix_svd_solve(const kg_matrix* U, const kg_vector* S, const kg_matrix* V, const kg_vector* b, kg_vector* x);

//these functions were previously used in SVD, currently they are unused, but provided anyways
void kg_matrix_householder(kg_matrix* H, const kg_vector* w);  //computes the householder matrix H = I - 2*w*w^T
void kg_matrix_householder_bidiagonalize(const kg_matrix* A, kg_matrix* P, kg_matrix* J, kg_matrix* Q);


//debug
void kg_matrix_print(const kg_matrix* m);
}  // namespace KGeoBag


#endif /* KGMatrixOperations_H__ */
