#ifndef KGDATAPOINT_HH_
#define KGDATAPOINT_HH_

#include <cstdarg>
#include <cstring>

#include <iostream>

namespace KGeoBag
{
  template <unsigned int Dimension = 1>
  class KGDataPoint
  {
  public:
    KGDataPoint();
    KGDataPoint(double,...);
    ~KGDataPoint() {}

    double& operator[](unsigned int i) { return fData[i]; }
    const double& operator[](unsigned int i) const { return fData[i]; }

    bool operator<(const KGDataPoint& rhs) const { return fData[0] < rhs.fData[0]; }

  private:
    double fData[Dimension+1];
  };

  template <unsigned int Dimension>
  KGDataPoint<Dimension>::KGDataPoint()
  {
    // this function is broken for Apple LLVM version 5.0 (clang-500.2.79)
    // (based on LLVM 3.3svn)
    // memset(fData,0.,Dimension+1);

    for (unsigned int i=0;i<Dimension+1;i++)
      fData[i] = 0.;
  }

  template <unsigned int Dimension>
  KGDataPoint<Dimension>::KGDataPoint(double x0,...)
  {
    va_list list;

    va_start(list,x0);

    fData[0] = x0;

    // the compiler should unroll this
    for (unsigned int i=0;i<Dimension;i++)
      fData[i+1] = va_arg(list,double);

    va_end(list);
  }
}

#endif
