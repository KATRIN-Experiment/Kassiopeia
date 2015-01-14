#ifndef KFMScaleInvariantKernelExpansion_HH__
#define KFMScaleInvariantKernelExpansion_HH__

#include "KFMKernelExpansion.hh"

namespace KEMField
{

/*
*
*@file KFMScaleInvariantKernelExpansion.hh
*@class KFMScaleInvariantKernelExpansion
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Nov  8 13:46:54 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM> //spatial dimension
class KFMScaleInvariantKernelExpansion: public KFMKernelExpansion<NDIM>
{
    public:

        KFMScaleInvariantKernelExpansion(): KFMKernelExpansion<NDIM>(){;};
        virtual ~KFMScaleInvariantKernelExpansion(){};

        bool IsScaleInvariant() const {return true;};

        virtual std::complex<double> GetSourceScaleFactor(int source_index, std::complex<double>& scale) const = 0;
        virtual std::complex<double> GetTargetScaleFactor(int target_index, std::complex<double>& scale) const = 0;

    protected:

    private:
};


}

#endif /* KFMScaleInvariantKernelExpansion_H__ */
