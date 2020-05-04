#ifndef KFMMatrixOperations_HH__
#define KFMMatrixOperations_HH__

/*
*
*@file KFMMatrixOperations.hh
*@class KFMMatrixOperations
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov 13 12:22:24 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

#include "KFMLinearAlgebraDefinitions.hh"

namespace KEMField
{

//allocation/deallocation
kfm_matrix* kfm_matrix_alloc(unsigned int nrows, unsigned int ncolumns);
kfm_matrix* kfm_matrix_calloc(unsigned int nrows, unsigned int ncolumns);
kfm_sparse_matrix* kfm_sparse_matrix_alloc(unsigned int nrows, unsigned int ncolumns, unsigned int n_elements);

void kfm_matrix_free(kfm_matrix* m);
void kfm_sparse_matrix_free(kfm_sparse_matrix* m);

//access
double kfm_matrix_get(const kfm_matrix* m, unsigned int i, unsigned int j);
void kfm_matrix_set(kfm_matrix* m, unsigned int i, unsigned int j, double x);
void kfm_matrix_set_zero(kfm_matrix* m);
void kfm_matrix_set_identity(kfm_matrix* m);
void kfm_matrix_set(const kfm_matrix* src, kfm_matrix* dest);

double kfm_sparse_matrix_get(const kfm_sparse_matrix* m, unsigned int i, unsigned int j);
void kfm_sparse_matrix_set(kfm_sparse_matrix* m, unsigned int i, unsigned int j, unsigned int element_index, double x);


//operations
void kfm_matrix_transpose(const kfm_matrix* in, kfm_matrix* out);
void kfm_matrix_multiply(const kfm_matrix* A, const kfm_matrix* B, kfm_matrix* C);  // C = A*B
void kfm_matrix_multiply_with_transpose(bool transposeA, bool transposeB, const kfm_matrix* A, const kfm_matrix* B,
                                        kfm_matrix* C);

void kfm_matrix_sub(kfm_matrix* a, const kfm_matrix* b);
void kfm_matrix_add(kfm_matrix* a, const kfm_matrix* b);
void kfm_matrix_scale(kfm_matrix* a, double scale_factor);

//computes the euler angles associated with the orthogonal 3x3 matrix R
//Assumes the rotation is given with the Z-Y'-Z'' convention
void kfm_matrix_euler_angles(const kfm_matrix* R, double& alpha, double& beta, double& gamma, double& tol);
void kfm_matrix_from_euler_angles_ZYZ(kfm_matrix* R, double alpha, double beta, double gamma);

void kfm_matrix_from_axis_angle(kfm_matrix* R, double angle, const kfm_vector* axis);
void kfm_matrix_from_axis_angle(kfm_matrix* R, double cos_angle, double sin_angle, const kfm_vector* axis);

//computes the singular value decomposition of the matrix A = U*diag(S)*V^T
void kfm_matrix_svd(const kfm_matrix* A, kfm_matrix* U, kfm_vector* S, kfm_matrix* V);

//given the singular value decomposition of the matrix A = U*diag(S)*V^T
// this function solves the equation Ax = b in the least squares sense
void kfm_matrix_svd_solve(const kfm_matrix* U, const kfm_vector* S, const kfm_matrix* V, const kfm_vector* b,
                          kfm_vector* x);

//solves the system Ax = b for the upper triangular matrix A
//using back substitution
void kfm_matrix_upper_triangular_solve(const kfm_matrix* A, const kfm_vector* b, kfm_vector* x);

//these functions were previously used in SVD, currently they are unused, but provided anyways
void kfm_matrix_householder(kfm_matrix* H, const kfm_vector* w);  //computes the householder matrix H = I - 2*w*w^T
void kfm_matrix_householder_bidiagonalize(const kfm_matrix* A, kfm_matrix* P, kfm_matrix* J, kfm_matrix* Q);


//debug
void kfm_matrix_print(const kfm_matrix* m);
}  // namespace KEMField


#endif /* KFMMatrixOperations_H__ */
