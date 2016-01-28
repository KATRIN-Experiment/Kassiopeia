#include "KGCubicSplineInterpolator.hh"

#include <cmath>
#include <algorithm>
#include <iostream>

namespace KGeoBag
{
  void KGCubicSplineInterpolator::Initialize(DataSet& data,double yp0,double ypn)
  {
    // From "Numerical Recipes in C":
    //
    // Given values of a tabulated function, and given values yp0 and ypn for
    // the first derivative of the interpolating function and points 0 and n-1,
    // respectively, this routine computes the array fYp that contains the
    // second derivatives of the interpolating function at the tabulated points.
    // If yp0 and/or ypn are equal to 1.e30 or larger, the routine is signaled
    // to set the corresponding boundary condition for a natural spline, with
    // zero second derivative on that boundary.

    fData = data;
    //sort the data in order of coordinate
    std::sort(fData.begin(), fData.end(), fSortingOp);

    fYp.resize(data.size(),0.);

    std::vector<double> u(data.size());

    unsigned int n = data.size()-1;

    // The lower boundary condition is set either to be "natural" or else to
    // have a specified first derivative.
    if (yp0 > 0.99e30)
      fYp[0] = u[0] = 0.;
    else
    {
      fYp[0] = -0.5;
      u[0] = 3./(fData[1][0]-fData[0][0]) * ((fData[1][1]-fData[0][1])/(fData[1][0]-fData[0][0]) - yp0);
    }

    // This is the decomposition loop of the tridiagional algorithm.  fYp and u
    // are used for the temporary storage of the decomposed factors.
    for (unsigned int i=1;i<n;i++)
    {
      double sig = (fData[i][0]-fData[i-1][0])/(fData[i+1][0]-fData[i-1][0]);
      double p = sig* fYp[i-1] + 2.;
      fYp[i] = (sig-1.)/p;
      u[i] = (fData[i+1][1]-fData[i][1])/(fData[i+1][0]-fData[i][0]) - (fData[i][1]-fData[i-1][1])/(fData[i][0]-fData[i-1][0]);
      u[i] = (6.*u[i]/(fData[i+1][0]-fData[i-1][0]) - sig*u[i-1])/p;
    }

    // The upper boundary condition is set either to be "natural" or else to
    // have a specified first derivative.
    double qn,un;
    if (ypn > 0.99e30)
      qn = un = 0.;
    else
    {
      qn = 0.5;
      un = 3./(fData[n][0]-fData[n-1][0])*(ypn-(fData[n][1]-fData[n-1][1])/(fData[n][0]-fData[n-1][0]));
    }
    fYp[n] = (un-qn*u[n-1])/(qn*fYp[n-1]+1.);

    // This is the backsubstitution loop of the tridiagonal algorithm.
    for (int i=n-1;i>=0;i--)
      fYp[i] = fYp[i]*fYp[i+1] + u[i];
  }

  int KGCubicSplineInterpolator::OutOfRange(double x) const
  {
    if (x < fData.front()[0]) return -1;
    if (x > fData.back()[0]) return 1;
    return 0;
  }

  double KGCubicSplineInterpolator::Range(unsigned int i) const
  {
    return (i == 0 ? fData.front()[0] : fData.back()[0]);
  }

  double KGCubicSplineInterpolator::operator()(double x) const
  {
    // From "Numerical Recipes in C"

    unsigned int klo=0;
    unsigned int khi=fData.size()-1;

    unsigned int k;

    // We will find the right place in the table by means of bisection.  This is
    // optimal if sequential calls to this routine are at random values of x.
    // If sequential calls are in order, and closely spaced, one would do better
    // to store previous values of klo and khi and test if they remain
    // appropriate on the next call.
    while (khi-klo > 1)
    {
      k = (khi+klo) >> 1;
      if (fData[k][0] > x) khi=k;
      else klo=k;
    }

    // klo and khi now bracket the input value of x.
    double h = fData[khi][0]-fData[klo][0];
    double a = (fData[khi][0]-x)/h;
    double b = (x-fData[klo][0])/h;

    return (a*fData[klo][1] + b*fData[khi][1] +
	    ((a*a*a-a)*fYp[klo] + (b*b*b-b)*fYp[khi])*(h*h)/6.);
  }
}
