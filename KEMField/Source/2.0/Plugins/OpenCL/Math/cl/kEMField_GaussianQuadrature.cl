#ifndef KEMFIELD_GAUSSIANQUADRATURE_CL
#define KEMFIELD_GAUSSIANQUADRATURE_CL

#include "kEMField_defines.h"

#define EVALUATE( x )       x
#define CONCATENATE(x, y)  x ## EVALUATE(y)

//______________________________________________________________________________

/**
 * A variation of Dr. Ferenc Glueck's numerical integration routine.
 */

#define GaussianQuadrature(FUNCTION)					\
  CL_TYPE GaussianQuadrature_ ## FUNCTION (CL_TYPE a,			\
					   CL_TYPE b,			\
					   int n,			\
					   CL_TYPE* par)		\
  {									\
    int i,j;								\
    CL_TYPE deln;							\
    CL_TYPE w5[6]={0.3187500000000000e+00,0.1376388888888889e+01,	\
		   0.6555555555555556e+00,0.1212500000000000e+01,	\
		   0.9256944444444445e+00,0.1011111111111111e+01};	\
    CL_TYPE w9[10]={ 0.2803440531305107e0,0.1648702325837748e1,		\
		     -0.2027449845679092e0,0.2797927414021179e1,	\
		     -0.9761199294532843e0,0.2556499393738999e1,	\
		     0.1451083002645404e0,0.1311227127425048e1,		\
		     0.9324249063051143e0,0.1006631393298060e1};	\
    if(n<12) n=12;							\
    deln=(b-a)/n;							\
									\
    CL_TYPE ans = 0.;							\
									\
    CL_TYPE x;								\
									\
    if(n>=20)								\
    {									\
      x = a;								\
      ans=ans+w9[0]*FUNCTION(&x,par);					\
      x = a+deln;							\
      ans=ans+w9[1]*FUNCTION(&x,par);					\
      x = a+deln*2;							\
      ans=ans+w9[2]*FUNCTION(&x,par);					\
      x = a+deln*3;							\
      ans=ans+w9[3]*FUNCTION(&x,par);					\
      x = a+deln*4;							\
      ans=ans+w9[4]*FUNCTION(&x,par);					\
      x = a+deln*5;							\
      ans=ans+w9[5]*FUNCTION(&x,par);					\
      x = a+deln*6;							\
      ans=ans+w9[6]*FUNCTION(&x,par);					\
      x = a+deln*7;							\
      ans=ans+w9[7]*FUNCTION(&x,par);					\
      x = a+deln*8;							\
      ans=ans+w9[8]*FUNCTION(&x,par);					\
      x = a+deln*9;							\
      ans=ans+w9[9]*FUNCTION(&x,par);					\
      for(i=10;i<n-9;i++)						\
      {									\
	x = a+deln*i;							\
	ans=ans+FUNCTION(&x,par);					\
      }									\
      x = a+deln*i;							\
      ans=ans+w9[n-i]*FUNCTION(&x,par);					\
      i++;								\
      x = a+deln*i;							\
      ans=ans+w9[n-i]*FUNCTION(&x,par);					\
      i++;								\
      x = a+deln*i;							\
      ans=ans+w9[n-i]*FUNCTION(&x,par);					\
      i++;								\
      x = a+deln*i;							\
      ans=ans+w9[n-i]*FUNCTION(&x,par);					\
      i++;								\
      x = a+deln*i;							\
      ans=ans+w9[n-i]*FUNCTION(&x,par);					\
      i++;								\
      x = a+deln*i;							\
      ans=ans+w9[n-i]*FUNCTION(&x,par);					\
      i++;								\
      x = a+deln*i;							\
      ans=ans+w9[n-i]*FUNCTION(&x,par);					\
      i++;								\
      x = a+deln*i;							\
      ans=ans+w9[n-i]*FUNCTION(&x,par);					\
      i++;								\
      x = a+deln*i;							\
      ans=ans+w9[n-i]*FUNCTION(&x,par);					\
      i++;								\
      x = a+deln*i;							\
      ans=ans+w9[n-i]*FUNCTION(&x,par);					\
    }									\
    else								\
    {									\
      x = a;								\
      ans=ans+w5[0]*FUNCTION(&x,par);					\
      x = a + deln;							\
      ans=ans+w5[1]*FUNCTION(&x,par);					\
      x = a + deln*2;							\
      ans=ans+w5[2]*FUNCTION(&x,par);					\
      x = a + deln*3;							\
      ans=ans+w5[3]*FUNCTION(&x,par);					\
      x = a + deln*4;							\
      ans=ans+w5[4]*FUNCTION(&x,par);					\
      x = a + deln*5;							\
      ans=ans+w5[5]*FUNCTION(&x,par);					\
      for(i=6;i<n-5;i++)						\
      {									\
	x = a + deln*i;							\
	ans=ans+FUNCTION(&x,par);					\
      }									\
      x = a+deln*i;							\
      ans=ans+w5[n-i]*FUNCTION(&x,par);					\
      i++;								\
      x = a+deln*i;							\
      ans=ans+w5[n-i]*FUNCTION(&x,par);					\
      i++;								\
      x = a+deln*i;							\
      ans=ans+w5[n-i]*FUNCTION(&x,par);					\
      i++;								\
      x = a+deln*i;							\
      ans=ans+w5[n-i]*FUNCTION(&x,par);					\
      i++;								\
      x = a+deln*i;							\
      ans=ans+w5[n-i]*FUNCTION(&x,par);					\
    }									\
    									\
    ans=ans*deln;							\
    									\
    return ans;								\
  }

#endif /* KEMFIELD_GAUSSIANQUADRATURE_CL */
