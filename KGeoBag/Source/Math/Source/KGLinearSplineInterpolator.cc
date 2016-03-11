#include "KGLinearSplineInterpolator.hh"

#include <cmath>
#include <algorithm>

namespace KGeoBag
{
  void KGLinearSplineInterpolator::Initialize(DataSet& data)
  {
    fData = data;
    std::sort(fData.begin(), fData.end(), fSortingOp);
  }

  int KGLinearSplineInterpolator::OutOfRange(double x) const
  {
    if (x < fData.front()[0]) return -1;
    if (x > fData.back()[0]) return 1;
    return 0;
  }

  double KGLinearSplineInterpolator::Range(unsigned int i) const
  {
    return (i == 0 ? fData.front()[0] : fData.back()[0]);
  }

  double KGLinearSplineInterpolator::operator()(double x) const
  {
    unsigned int klo=0;
    unsigned int khi=fData.size()-1;

    //if the value of x is outside of the domain, return the end point values
    if(x < fData[klo][0] ){return fData[0][1];};
    if(x > fData[khi][0]){return fData[khi][1];};

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

    return a*fData[klo][1] + b*fData[khi][1];
  }
}
