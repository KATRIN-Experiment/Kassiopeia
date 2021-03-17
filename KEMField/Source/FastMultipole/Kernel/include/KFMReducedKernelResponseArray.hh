#ifndef KFMReducedKernelResponseArray_H__
#define KFMReducedKernelResponseArray_H__


#include "KFMArrayFillingOperator.hh"
#include "KFMMessaging.hh"
#include "KFMScalarMultipoleExpansion.hh"

#include <cmath>
#include <complex>
#include <cstdlib>

namespace KEMField
{

/**
*
*@file KFMReducedKernelResponseArray.hh
*@class KFMReducedKernelResponseArray
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Sep 29 21:59:53 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<class KernelType, bool OriginIsSource, unsigned int SpatialNDIM>
class KFMReducedKernelResponseArray : public KFMArrayFillingOperator<std::complex<double>, SpatialNDIM + 1>
{

  public:
    KFMReducedKernelResponseArray() : KFMArrayFillingOperator<std::complex<double>, SpatialNDIM + 1>()
    {
        fNTerms = 0;
        fVerbose = 0;

        fInitialized = false;
        fLength = 0;  //side length of the region

        for (unsigned int i = 0; i < SpatialNDIM + 1; i++) {
            fLowerLimits[i] = 0;
            fUpperLimits[i] = 0;
        }

        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            fOrigin[i] = 0;
            fShift[i] = 0;
        }

        fKernel.SetSourceOrigin(fOrigin);
        fKernel.SetTargetOrigin(fOrigin);

        fZeroMaskSize = 0;
    }

    ~KFMReducedKernelResponseArray() override = default;

    KernelType* GetKernel()
    {
        return &fKernel;
    };

    void SetVerbose(int v)
    {
        fVerbose = v;
    }

    virtual void SetLength(double len)
    {
        fLength = len;
    }


    virtual void SetNumberOfTermsInSeries(unsigned int n_terms)
    {
        fNTerms = n_terms;

        fLowerLimits[0] = 0;
        fUpperLimits[0] = fNTerms;
    }

    virtual void SetLowerSpatialLimits(const int* low)
    {
        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            fLowerLimits[i + 1] = low[i];
        }
    }

    virtual void SetUpperSpatialLimits(const int* up)
    {
        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            fUpperLimits[i + 1] = up[i];
        }
    }

    virtual void SetOrigin(const double* origin)
    {
        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            fOrigin[i] = origin[i];
        }

        //origin is the source, so we compute and out going response
        if (OriginIsSource) {
            fKernel.SetSourceOrigin(fOrigin);
        }

        //origin is not the source, but rather the target, so we compute an in going response function
        if (!OriginIsSource) {
            fKernel.SetTargetOrigin(fOrigin);
        }
    }

    //distance between array points
    virtual void SetDistance(double dist)
    {
        fLength = std::fabs(dist);
    }

    virtual void SetZeroMaskSize(int zmask)
    {
        fZeroMaskSize = std::abs(zmask);
    }

    virtual void SetShift(const int* shift)
    {
        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            fShift[i] = shift[i];
        }
    }

    void Initialize() override
    {
        fInitialized = false;
        if (this->fOutput != nullptr) {
            if (this->IsBoundedDomainSubsetOfArray(this->fOutput, fLowerLimits, fUpperLimits)) {
                fKernel.Initialize();
                fInitialized = true;
            }
            else {
                kfmout
                    << "KFMReducedKernelResponseArray::Initialize: error, not properly initialized, array bounds/bases incorrect."
                    << kfmendl;
            }
        }
        else {
            kfmout << "KFMReducedKernelResponseArray::Initialize: error, initialization failed, output array not set."
                   << kfmendl;
        }
    }

    void ExecuteOperation() override
    {
        //clear out the array and reset to zero everywhere
        this->ResetArray(this->fOutput, std::complex<double>(0, 0));

        if (fInitialized) {
            unsigned int storage_index[SpatialNDIM];
            unsigned int spatial_size[SpatialNDIM];
            int physical_index[SpatialNDIM];
            int del_origin[SpatialNDIM];
            unsigned int total_size = 1;


            for (unsigned int i = 0; i < SpatialNDIM; i++) {
                spatial_size[i] = fUpperLimits[i + 1] - fLowerLimits[i + 1];
                total_size *= spatial_size[i];
            }


            for (unsigned int i = 0; i < total_size; i++) {
                KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(i, spatial_size, storage_index);
                for (unsigned int j = 0; j < SpatialNDIM; j++) {
                    physical_index[j] = (int) storage_index[j] + fLowerLimits[j + 1];
                    del_origin[j] = physical_index[j] + fShift[j];
                    fCoord[j] = (del_origin[j]) * fLength;
                }

                if (OriginIsSource)  //origin is the source, so we compute an outgoing response
                {
                    fKernel.SetTargetOrigin(fCoord);
                }

                if (!OriginIsSource)  //origin is not the source, but rather the target, so we compute an in going response function
                {
                    fKernel.SetSourceOrigin(fCoord);
                }

                if (IsOutsideZeroMask(del_origin)) {
                    ComputeCoefficientsAtPoint(physical_index);
                }
            }
        }
    }

  protected:
    virtual bool IsOutsideZeroMask(const int* index) const
    {
        if (fZeroMaskSize > 0) {
            for (unsigned int i = 0; i < SpatialNDIM; i++) {
                if (std::fabs(index[i]) > fZeroMaskSize) {
                    return true;
                };
            }
            return false;
        }
        else {
            return true;
        }
    }

    virtual void ComputeCoefficientsAtPoint(const int* physical_index)
    {
        std::complex<double> result;
        int index[SpatialNDIM + 1];

        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            index[i + 1] = physical_index[i];
        }

        for (unsigned int response = 0; response < fNTerms; response++) {
            index[0] = response;
            result = std::complex<double>(0, 0);
            //call kernel to get the raw reponse function (no normalization)
            (*this->fOutput)[index] = fKernel.GetIndependentResponseFunction(response);
        }
    }


    int fVerbose;
    bool fInitialized;

    double fLength;  //side length of the region
    int fZeroMaskSize;
    unsigned int fNTerms;

    int fLowerLimits[SpatialNDIM + 1];
    int fUpperLimits[SpatialNDIM + 1];
    double fOrigin[SpatialNDIM];
    int fShift[SpatialNDIM];

    //response kernel calculator
    KernelType fKernel;

    //scratch space
    mutable double fCoord[SpatialNDIM];
};


}  // namespace KEMField

#endif /* __KFMReducedKernelResponseArray_H__ */
