#ifndef KFMKernelExpansion_HH__
#define KFMKernelExpansion_HH__

#include <complex>
#include <cstddef>

namespace KEMField
{

/*
*
*@file KFMKernelExpansion.hh
*@class KFMKernelExpansion
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Nov  8 13:46:54 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM>  //spatial dimension
class KFMKernelExpansion
{
  public:
    KFMKernelExpansion()
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fSourceOrigin[i] = 0.;
            fTargetOrigin[i] = 0.;
            fDel[i] = 0.;
        }
    };

    virtual ~KFMKernelExpansion() = default;
    ;

    virtual void Initialize()
    {
        ;
    };

    virtual bool IsScaleInvariant() const
    {
        return false;
    }

    int GetSpatialDimension() const
    {
        return NDIM;
    };

    virtual void SetSourceOrigin(const double* origin)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fSourceOrigin[i] = origin[i];
            fDel[i] = fSourceOrigin[i] - fTargetOrigin[i];
        }
    }

    virtual void SetTargetOrigin(const double* origin)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fTargetOrigin[i] = origin[i];
            fDel[i] = fSourceOrigin[i] - fTargetOrigin[i];
        }
    }

    virtual bool IsPhysical(int source_index, int target_index) const = 0;

    virtual std::complex<double> GetResponseFunction(int source_index, int target_index) const = 0;


    virtual std::complex<double> GetNormalizationFactor(int, int) const
    {
        return std::complex<double>(0.0, 0.0);
    }

    virtual std::complex<double> GetIndependentResponseFunction(int) const
    {
        return std::complex<double>(0., 0.);
    }

  protected:
    double fSourceOrigin[NDIM];
    double fTargetOrigin[NDIM];
    double fDel[NDIM];

  private:
};


}  // namespace KEMField

#endif /* KFMKernelExpansion_H__ */
