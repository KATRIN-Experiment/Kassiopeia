#ifndef KFMResponseKernelM2L_H__
#define KFMResponseKernelM2L_H__


#include "KFMResponseKernel.hh"

namespace KEMField{

/**
*
*@file KFMResponseKernelM2L.hh
*@class KFMResponseKernelM2L
*@brief multipole to multipole response (translation) kernel
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Sep 29 13:55:42 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMResponseKernelM2L: public KFMResponseKernel
{
    public:
        KFMResponseKernelM2L():KFMResponseKernel(){;};
        virtual ~KFMResponseKernelM2L(){;};

        virtual void Initialize();

        virtual int GetTargetIndex(int j, int k) const;
        virtual int GetSourceIndex(int /*j*/, int /*k*/, int n, int m) const;
        virtual bool IsPhysical(int j, int k, int n, int m) const;

        virtual std::complex<double> GetNormalizationCoeff(int j, int k, int n, int m) const;
        virtual std::complex<double> GetResponseFunction(int j, int k, int n, int m) const;

        virtual std::complex<double> GetIndependentResponseFunction(int response_index) const;

    protected:

};


}

#endif /* __KFMResponseKernelM2L_H__ */
