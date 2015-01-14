#ifndef KSVDSOLVER_DEF
#define KSVDSOLVER_DEF

#include "KMatrix.hh"
#include "KVector.hh"

#include <cmath>
#include <cstdlib>

#ifdef KEMFIELD_USE_GSL
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_linalg.h>
#endif

#define SIGN(a,b) ((b) > 0.0 ? fabs(a) : - fabs(a))
#define FMAX(a,b) ((double)(a) > (double)(b) ? (double)(a) : (double)(b))
#define IMIN(a,b) ((int)(a) < (int)(b) ? (int)(a) : (int)(b))
#define SQR(a) ((double)(a) == 0.0 ? 0.0 : (double)(a) * (double)(a))

namespace KEMField
{
  template <typename ValueType>
  class KSVDSolver
  {
  public:
    typedef KMatrix<ValueType> Matrix;
    typedef KVector<ValueType> Vector;

    KSVDSolver()  : fTolerance(1.e-14) {}
    virtual ~KSVDSolver() {}

    bool Solve(const Matrix& A,Vector& x,const Vector& b) const;
    void SetTolerance(double tol) { fTolerance = tol; }

  private:
    int svdcmp(double **a, int nRows, int nCols, double *w, double **v);
    double pythag(double a, double b);

    double fTolerance;

  };

  // calculates sqrt( a^2 + b^2 ) with decent precision
  template <typename ValueType>
  double KSVDSolver<ValueType>::pythag(double a, double b) {
    double absa,absb;

    absa = fabs(a);
    absb = fabs(b);

    if(absa > absb)
      return(absa * sqrt(1.0 + SQR(absb/absa)));
    else
      return(absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa / absb)));
  }

  template <typename ValueType>
  int KSVDSolver<ValueType>::svdcmp(double **a, int nRows, int nCols, double *w, double **v)
  {
  /*
    Modified from Numerical Recipes in C
    Given a matrix a[nRows][nCols], svdcmp() computes its singular value
    decomposition, A = U * W * Vt.  A is replaced by U when svdcmp
    returns.  The diagonal matrix W is output as a vector w[nCols].
    V (not V transpose) is output as the matrix V[nCols][nCols].
  */
    int flag,i,its,j,jj,k,l,nm;
    double anorm,c,f,g,h,s,scale,x,y,z,*rv1;

    rv1 = (double*)(std::malloc(sizeof(double)*nCols));
    if(rv1 == NULL)
    {
      printf("svdcmp(): Unable to allocate vector\n");
      return(-1);
    }

    g = scale = anorm = 0.0;
    for(i=0;i<nCols;i++) {
      l = i+1;
      rv1[i] = scale*g;
      g = s = scale = 0.0;
      if(i < nRows) {
	for(k=i;k<nRows;k++) scale += fabs(a[k][i]);
	if(scale) {
	  for(k=i;k<nRows;k++) {
	    a[k][i] /= scale;
	    s += a[k][i] * a[k][i];
	  }
	  f = a[i][i];
	  g = -SIGN(sqrt(s),f);
	  h = f * g - s;
	  a[i][i] = f - g;
	  for(j=l;j<nCols;j++) {
	    for(s=0.0,k=i;k<nRows;k++) s += a[k][i] * a[k][j];
	    f = s / h;
	    for(k=i;k<nRows;k++) a[k][j] += f * a[k][i];
	  }
	  for(k=i;k<nRows;k++) a[k][i] *= scale;
	}
      }
      w[i] = scale * g;
      g = s = scale = 0.0;
      if(i < nRows && i != nCols-1) {
	for(k=l;k<nCols;k++) scale += fabs(a[i][k]);
	if(scale)  {
	  for(k=l;k<nCols;k++) {
	    a[i][k] /= scale;
	    s += a[i][k] * a[i][k];
	  }
	  f = a[i][l];
	  g = - SIGN(sqrt(s),f);
	  h = f * g - s;
	  a[i][l] = f - g;
	  for(k=l;k<nCols;k++) rv1[k] = a[i][k] / h;
	  for(j=l;j<nRows;j++) {
	    for(s=0.0,k=l;k<nCols;k++) s += a[j][k] * a[i][k];
	    for(k=l;k<nCols;k++) a[j][k] += s * rv1[k];
	  }
	  for(k=l;k<nCols;k++) a[i][k] *= scale;
	}
      }
      anorm = FMAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
    }

    for(i=nCols-1;i>=0;i--) {
      if(i < nCols-1) {
	if(g) {
	  for(j=l;j<nCols;j++)
	    v[j][i] = (a[i][j] / a[i][l]) / g;
	  for(j=l;j<nCols;j++) {
	    for(s=0.0,k=l;k<nCols;k++) s += a[i][k] * v[k][j];
	    for(k=l;k<nCols;k++) v[k][j] += s * v[k][i];
	  }
	}
	for(j=l;j<nCols;j++) v[i][j] = v[j][i] = 0.0;
      }
      v[i][i] = 1.0;
      g = rv1[i];
      l = i;
    }

    for(i=IMIN(nRows,nCols) - 1;i >= 0;i--) {
      l = i + 1;
      g = w[i];
      for(j=l;j<nCols;j++) a[i][j] = 0.0;
      if(g) {
	g = 1.0 / g;
	for(j=l;j<nCols;j++) {
	  for(s=0.0,k=l;k<nRows;k++) s += a[k][i] * a[k][j];
	  f = (s / a[i][i]) * g;
	  for(k=i;k<nRows;k++) a[k][j] += f * a[k][i];
	}
	for(j=i;j<nRows;j++) a[j][i] *= g;
      }
      else
	for(j=i;j<nRows;j++) a[j][i] = 0.0;
      ++a[i][i];
    }

    for(k=nCols-1;k>=0;k--) {
      for(its=0;its<30;its++) {
	flag = 1;
	for(l=k;l>=0;l--) {
	  nm = l-1;
	  if((fabs(rv1[l]) + anorm) == anorm) {
	    flag =  0;
	    break;
	  }
	  if((fabs(w[nm]) + anorm) == anorm) break;
	}
	if(flag) {
	  c = 0.0;
	  s = 1.0;
	  for(i=l;i<=k;i++) {
	    f = s * rv1[i];
	    rv1[i] = c * rv1[i];
	    if((fabs(f) + anorm) == anorm) break;
	    g = w[i];
	    h = pythag(f,g);
	    w[i] = h;
	    h = 1.0 / h;
	    c = g * h;
	    s = -f * h;
	    for(j=0;j<nRows;j++) {
	      y = a[j][nm];
	      z = a[j][i];
	      a[j][nm] = y * c + z * s;
	      a[j][i] = z * c - y * s;
	    }
	  }
	}
	z = w[k];
	if(l == k) {
	  if(z < 0.0) {
	    w[k] = -z;
	    for(j=0;j<nCols;j++) v[j][k] = -v[j][k];
	  }
	  break;
	}
	if(its == 29) printf("no convergence in 30 svdcmp iterations\n");
	x = w[l];
	nm = k-1;
	y = w[nm];
	g = rv1[nm];
	h = rv1[k];
	f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
	g = pythag(f,1.0);
	f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g,f))) - h)) / x;
	c = s = 1.0;
	for(j=l;j<=nm;j++) {
	  i = j+1;
	  g = rv1[i];
	  y = w[i];
	  h = s * g;
	  g = c * g;
	  z = pythag(f,h);
	  rv1[j] = z;
	  c = f/z;
	  s = h/z;
	  f = x * c + g * s;
	  g = g * c - x * s;
	  h = y * s;
	  y *= c;
	  for(jj=0;jj<nCols;jj++) {
	    x = v[jj][j];
	    z = v[jj][i];
	    v[jj][j] = x * c + z * s;
	    v[jj][i] = z * c - x * s;
	  }
	  z = pythag(f,h);
	  w[j] = z;
	  if(z) {
	    z = 1.0 / z;
	    c = f * z;
	    s = h * z;
	  }
	  f = c * g + s * y;
	  x = c * y - s * g;
	  for(jj=0;jj < nRows;jj++) {
	    y = a[jj][j];
	    z = a[jj][i];
	    a[jj][j] = y * c + z * s;
	    a[jj][i] = z * c - y * s;
	  }
	}
	rv1[l] = 0.0;
	rv1[k] = f;
	w[k] = x;
      }
    }

    std::free(rv1);

    return(0);
  }

  template <typename ValueType>
  bool KSVDSolver<ValueType>::Solve(const Matrix& A,
				    Vector& x,
				    const Vector& b) const
  {
#ifdef KEMFIELD_USE_GSL
    unsigned int M = A.Dimension(0);
    unsigned int N = A.Dimension(1);
    if (M < N)
      M = N;
    gsl_matrix* A_ = gsl_matrix_alloc(M,N);
    for (unsigned int i=0;i<M;i++)
      for (unsigned int j=0;j<N;j++)
	if (i < A.Dimension(0))
	  gsl_matrix_set(A_,i,j,A(i,j));
	else
	  gsl_matrix_set(A_,i,j,0.);

    gsl_vector* x_ = gsl_vector_alloc(N);
    gsl_vector* b_ = gsl_vector_alloc(M);
    for (unsigned int i=0;i<M;i++)
      if (i < b.Dimension())
	gsl_vector_set(b_,i,b(i));
      else
	gsl_vector_set(b_,i,0.);

    gsl_matrix* V = gsl_matrix_alloc(A_->size2,A_->size2);
    gsl_vector* S = gsl_vector_alloc(A_->size1);
    gsl_vector* work = gsl_vector_alloc(A_->size1);
    gsl_linalg_SV_decomp(A_, V, S, work);
    gsl_vector_free(work);
    gsl_linalg_SV_solve(A_, V, S, b_, x_);

    for (unsigned int i=0;i<x.Dimension();i++)
      x[i] = gsl_vector_get(x_,i);

    KSimpleVector<double> b_comp(b.Dimension());
    A.Multiply(x,b_comp);
    b_comp -= b;

    return b_comp.InfinityNorm()<fTolerance;
#else
    (void)A;
    (void)x;
    (void)b;
    std::cout << "KSVDSolver::Solve(): Please activate GSL in order to use the singular value decomposition." << std::endl;
    return false;
#endif
  }

}

#endif /* KSVDSOLVER_DEF */
