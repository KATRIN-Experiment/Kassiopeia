#ifndef KFMMultidimensionalFastFourierTransformFFTW_HH__
#define KFMMultidimensionalFastFourierTransformFFTW_HH__

#include "KFMArrayWrapper.hh"
#include "KFMFastFourierTransform.hh"

#include <algorithm>
#include <cstring>
#include <fftw3.h>

int fftw_alignment_of(double*) __attribute__((weak));  // weak declaration must always be present

namespace KEMField
{

/*
*
*@file KFMMultidimensionalFastFourierTransformFFTW.hh
*@class KFMMultidimensionalFastFourierTransformFFTW
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov 27 00:01:00 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM>
class KFMMultidimensionalFastFourierTransformFFTW : public KFMUnaryArrayOperator<std::complex<double>, NDIM>
{
  public:
    KFMMultidimensionalFastFourierTransformFFTW()
    {
        fTotalArraySize = 0;
        fInPtr = NULL;
        fOutPtr = NULL;
        fInPlacePtr = NULL;
        for (unsigned int i = 0; i < NDIM; i++) {
            fDimensionSize[i] = 0;
        }

        fIsValid = false;
        fInitialized = false;
        fForward = true;
    };

    virtual ~KFMMultidimensionalFastFourierTransformFFTW()
    {
        DealocateWorkspace();
        if (fInitialized) {
            fftw_destroy_plan(fPlanForward);
            fftw_destroy_plan(fPlanBackward);
            fftw_destroy_plan(fPlanForwardInPlace);
            fftw_destroy_plan(fPlanBackwardInPlace);
        }
    };

    virtual void SetForward()
    {
        fForward = true;
    }
    virtual void SetBackward()
    {
        fForward = false;
    };

    virtual void Initialize()
    {
        if (DoInputOutputDimensionsMatch()) {
            fIsValid = true;
            this->fInput->GetArrayDimensions(fDimensionSize);
        }
        else {
            fIsValid = false;
        }

        if (!fInitialized && fIsValid) {
            fTotalArraySize = KFMArrayMath::TotalArraySize<NDIM>(fDimensionSize);
            AllocateWorkspace();
            bool success = ConstructPlan();
            fInitialized = success;
        }
    }

    virtual void ExecuteOperation()
    {
        if (fIsValid && fInitialized) {
            //check memory alignment to determine if we can avoid copying the data around
            if ((fftw_alignment_of) &&  // function is not always defined in fftw
                (fftw_alignment_of(reinterpret_cast<double*>(this->fInput->GetData())) ==
                 fftw_alignment_of(reinterpret_cast<double*>(fInPtr))) &&
                (fftw_alignment_of(reinterpret_cast<double*>(this->fOutput->GetData())) ==
                 fftw_alignment_of(reinterpret_cast<double*>(fOutPtr)))) {
                if (this->fInput->GetData() != this->fOutput->GetData()) {
                    //transform is out-of-place
                    if (fForward) {
                        fftw_execute_dft(fPlanForward,
                                         reinterpret_cast<fftw_complex*>(this->fInput->GetData()),
                                         reinterpret_cast<fftw_complex*>(this->fOutput->GetData()));
                    }
                    else {
                        fftw_execute_dft(fPlanBackward,
                                         reinterpret_cast<fftw_complex*>(this->fInput->GetData()),
                                         reinterpret_cast<fftw_complex*>(this->fOutput->GetData()));
                    }
                }
                else {
                    //we have to execute an in-place transform
                    if (fForward) {
                        fftw_execute_dft(fPlanForwardInPlace,
                                         reinterpret_cast<fftw_complex*>(this->fInput->GetData()),
                                         reinterpret_cast<fftw_complex*>(this->fOutput->GetData()));
                    }
                    else {
                        fftw_execute_dft(fPlanBackwardInPlace,
                                         reinterpret_cast<fftw_complex*>(this->fInput->GetData()),
                                         reinterpret_cast<fftw_complex*>(this->fOutput->GetData()));
                    }
                }
            }
            else {
                //alignment doesn't match so we need to use memcpy / copy
                //std::memcpy( fInPtr, this->fInput->GetData() , fTotalArraySize*sizeof(fftw_complex) );
                std::copy(this->fInput->GetData(),
                          this->fInput->GetData() + fTotalArraySize,
                          reinterpret_cast<std::complex<double>*>(fInPtr));
                if (fForward) {
                    fftw_execute(fPlanForward);
                }
                else {
                    fftw_execute(fPlanBackward);
                }
                //std::memcpy(this->fOutput->GetData(), fOutPtr, fTotalArraySize*sizeof(fftw_complex) );
                std::copy(reinterpret_cast<std::complex<double>*>(fOutPtr),
                          reinterpret_cast<std::complex<double>*>(fOutPtr) + fTotalArraySize,
                          this->fOutput->GetData());
            }
        }
    }


  private:
    virtual void AllocateWorkspace()
    {
        fInPtr = fftw_alloc_complex(fTotalArraySize);
        fOutPtr = fftw_alloc_complex(fTotalArraySize);
        fInPlacePtr = fftw_alloc_complex(fTotalArraySize);
    }

    virtual void DealocateWorkspace()
    {
        fftw_free(fInPtr);
        fftw_free(fOutPtr);
        fftw_free(fInPlacePtr);
    }


    bool ConstructPlan()
    {
        if (fInPtr == NULL || fOutPtr == NULL || fInPlacePtr == NULL) {
            return false;
        }

        int rank = NDIM;
        //we force fftw to only do one fft at a time
        //but we could implement a batched interface also...
        int howmany_rank = 0;  //zero disables more than one x-form
        fHowManyDims.n = 1;
        fHowManyDims.is = KFMArrayMath::TotalArraySize<NDIM>(fDimensionSize);
        fHowManyDims.os = KFMArrayMath::TotalArraySize<NDIM>(fDimensionSize);

        for (unsigned int i = 0; i < NDIM; i++) {
            fDims[i].n = fDimensionSize[i];
            fDims[i].is = KFMArrayMath::StrideFromRowMajorIndex<NDIM>(i, fDimensionSize);
            fDims[i].os = KFMArrayMath::StrideFromRowMajorIndex<NDIM>(i, fDimensionSize);
        }

        fPlanForward = fftw_plan_guru_dft(rank,
                                          fDims,
                                          howmany_rank,
                                          &fHowManyDims,
                                          fInPtr,
                                          fOutPtr,
                                          FFTW_FORWARD,
                                          FFTW_EXHAUSTIVE);

        fPlanBackward = fftw_plan_guru_dft(rank,
                                           fDims,
                                           howmany_rank,
                                           &fHowManyDims,
                                           fInPtr,
                                           fOutPtr,
                                           FFTW_BACKWARD,
                                           FFTW_EXHAUSTIVE);

        fPlanForwardInPlace = fftw_plan_guru_dft(rank,
                                                 fDims,
                                                 howmany_rank,
                                                 &fHowManyDims,
                                                 fInPlacePtr,
                                                 fInPlacePtr,
                                                 FFTW_FORWARD,
                                                 FFTW_EXHAUSTIVE);

        fPlanBackwardInPlace = fftw_plan_guru_dft(rank,
                                                  fDims,
                                                  howmany_rank,
                                                  &fHowManyDims,
                                                  fInPlacePtr,
                                                  fInPlacePtr,
                                                  FFTW_BACKWARD,
                                                  FFTW_EXHAUSTIVE);

        if (fPlanForward != NULL && fPlanBackward != NULL && fPlanBackwardInPlace != NULL &&
            fPlanForwardInPlace != NULL) {
            return true;
        }
        else {
            return false;
        }
    }


    virtual bool DoInputOutputDimensionsMatch()
    {
        unsigned int in[NDIM];
        unsigned int out[NDIM];

        this->fInput->GetArrayDimensions(in);
        this->fOutput->GetArrayDimensions(out);

        for (unsigned int i = 0; i < NDIM; i++) {
            if (in[i] != out[i]) {
                return false;
            }
        }
        return true;
    }

    bool fIsValid;
    bool fForward;
    bool fInitialized;

    size_t fTotalArraySize;
    unsigned int fDimensionSize[NDIM];

    fftw_iodim fDims[NDIM];
    fftw_iodim fHowManyDims;
    fftw_plan fPlanForward;
    fftw_plan fPlanBackward;
    fftw_plan fPlanForwardInPlace;
    fftw_plan fPlanBackwardInPlace;
    fftw_complex* fInPtr;
    fftw_complex* fOutPtr;
    fftw_complex* fInPlacePtr;
};


}  // namespace KEMField

#endif /* KFMMultidimensionalFastFourierTransformFFTW_H__ */
