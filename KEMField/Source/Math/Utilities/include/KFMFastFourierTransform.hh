#ifndef KFMFastFourierTransform_HH__
#define KFMFastFourierTransform_HH__

#include "KFMArrayWrapper.hh"
#include "KFMBitReversalPermutation.hh"
#include "KFMFastFourierTransformUtilities.hh"
#include "KFMUnaryArrayOperator.hh"

#include <complex>

namespace KEMField
{

/*
*
*@file KFMFastFourierTransform.hh
*@class KFMFastFourierTransform
*@brief This is a class for a one dimensional FFT
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Nov 26 10:33:12 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMFastFourierTransform : public KFMUnaryArrayOperator<std::complex<double>, 1>
{
  public:
    KFMFastFourierTransform();
    ~KFMFastFourierTransform() override;

    virtual void SetSize(unsigned int N);

    virtual void SetForward();
    virtual void SetBackward();

    void Initialize() override;

    void ExecuteOperation() override;

  private:
    virtual void AllocateWorkspace();
    virtual void DealocateWorkspace();

    bool fIsValid;
    bool fForward;
    bool fInitialized;
    bool fSizeIsPowerOfTwo;
    bool fSizeIsPowerOfThree;

    //auxilliary workspace needed for basic 1D transform
    unsigned int fN;
    unsigned int fM;
    unsigned int* fPermutation;
    std::complex<double>* fTwiddle;
    std::complex<double>* fConjugateTwiddle;
    std::complex<double>* fScale;
    std::complex<double>* fCirculant;
    std::complex<double>* fWorkspace;
};

}  // namespace KEMField


#endif /* KFMFastFourierTransform_H__ */
