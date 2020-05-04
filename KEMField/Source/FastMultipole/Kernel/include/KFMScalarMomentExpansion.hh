#ifndef KFMScalarMomentExpansion_H
#define KFMScalarMomentExpansion_H

#include <cmath>
#include <complex>
#include <vector>

namespace KEMField
{

/**
*
*@file KFMScalarMomentExpansion.hh
*@class KFMScalarMomentExpansion
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Aug 24 09:56:34 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMScalarMomentExpansion
{
  public:
    KFMScalarMomentExpansion()
    {
        ;
    };
    virtual ~KFMScalarMomentExpansion()
    {
        ;
    };

    virtual void Clear() = 0;

    virtual void SetNumberOfTermsInSeries(unsigned int n_terms) = 0;
    virtual unsigned int GetNumberOfTermsInSeries() const = 0;

    virtual void SetMoments(const std::vector<std::complex<double>>* mom) = 0;
    virtual void GetMoments(std::vector<std::complex<double>>* mom) const = 0;


  protected:
};

}  // namespace KEMField

#endif /* KFMScalarMomentExpansion_H */
