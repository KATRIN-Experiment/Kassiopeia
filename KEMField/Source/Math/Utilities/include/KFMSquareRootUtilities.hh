#ifndef KFMSquareRootUtilities_HH__
#define KFMSquareRootUtilities_HH__

#include <cmath>

#define KFM_SQRT_LUT_SIZE 10000

namespace KEMField
{

/*
*
*@file KFMSquareRootUtilities.hh
*@class KFMSquareRootUtilities
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jun 13 16:11:46 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMSquareRootUtilities
{
  public:
    KFMSquareRootUtilities() = default;
    ;
    virtual ~KFMSquareRootUtilities() = default;
    ;

    static double SqrtInteger(int arg);
    static double InverseSqrtInteger(int arg);
    static double SqrtIntegerRatio(int numer, int denom);

  private:
    static const double fIntegerSqrtLUT[KFM_SQRT_LUT_SIZE];
    static const double fInverseIntegerSqrtLUT[KFM_SQRT_LUT_SIZE];
};


}  // namespace KEMField

#endif /* KFMSquareRootUtilities_H__ */
