#ifndef KFMKernelResponseArray_H__
#define KFMKernelResponseArray_H__


#include <complex>
#include <cmath>

#include "KFMArrayFillingOperator.hh"

#include "KFMMessaging.hh"

namespace KEMField{

/**
*
*@file KFMKernelResponseArray.hh
*@class KFMKernelResponseArray
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Sep 29 21:59:53 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template <class KernelType, bool OriginIsSource, unsigned int SpatialNDIM>
class KFMKernelResponseArray: public KFMArrayFillingOperator< std::complex<double>, SpatialNDIM + 2 >
{

    public:
        KFMKernelResponseArray():KFMArrayFillingOperator< std::complex<double>, SpatialNDIM + 2 >()
        {
            fNTerms = 0;
            fVerbose = 0;

            fInitialized = false;
            fLength = 0; //side length of the region

            for(unsigned int i=0; i<SpatialNDIM+2; i++)
            {
                fLowerLimits[i] = 0;
                fUpperLimits[i] = 0;
            }

            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                fOrigin[i] = 0;
                fShift[i] = 0;
            }

            fKernel.SetSourceOrigin(fOrigin);
            fKernel.SetTargetOrigin(fOrigin);

            fZeroMaskSize = 0;
        }

        virtual ~KFMKernelResponseArray()
        {

        }

        KernelType* GetKernel(){return &fKernel;};

        void SetVerbose(int v){fVerbose = v;}

        virtual void SetLength(double len)
        {
            fLength = len;
        }


        virtual void SetNumberOfTermsInSeries(unsigned int n_terms)
        {
            fNTerms = n_terms;
            fLowerLimits[0] = 0;
            fLowerLimits[1] = 0;
            fUpperLimits[0] = fNTerms;
            fUpperLimits[1] = fNTerms;
        }

        virtual void SetLowerSpatialLimits(const int* low)
        {
            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                fLowerLimits[i+2] = low[i];
            }
        }

        virtual void SetUpperSpatialLimits(const int* up)
        {
            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                fUpperLimits[i+2] = up[i];
            }
        }

        virtual void SetOrigin(const double* origin)
        {
            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                fOrigin[i] = origin[i];
            }

            //origin is the source, so we compute and out going response
            if(OriginIsSource){ fKernel.SetSourceOrigin(fOrigin); }

            //origin is not the source, but rather the target, so we compute an in going response function
            if(!OriginIsSource){fKernel.SetTargetOrigin(fOrigin);}

        }

        //distance between array points
        virtual void SetDistance(double dist){fLength = std::fabs(dist); }

        virtual void SetZeroMaskSize(int zmask){fZeroMaskSize = std::fabs(zmask);}

        virtual void SetShift(const int* shift)
        {
            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                fShift[i] = shift[i];
            }
        }

        virtual void Initialize()
        {
            fInitialized = false;
            if(this->fOutput != NULL)
            {
                if(this->IsBoundedDomainSubsetOfArray(this->fOutput, fLowerLimits, fUpperLimits))
                {
                    fKernel.Initialize();
                    fInitialized = true;
                }
                else
                {
                    kfmout<<"KFMKernelResponseArray::Initialize: error, not properly initialized, array bounds/bases incorrect."<<kfmendl;
                }
            }
            else
            {
                kfmout<<"KFMKernelResponseArray::Initialize: error, initialization failed, output array not set."<<kfmendl;
            }
        }

        virtual void ExecuteOperation()
        {
            //clear out the array and reset to zero everywhere
            this->ResetArray(this->fOutput, std::complex<double>(0,0) );

            if(fInitialized)
            {
                unsigned int storage_index[SpatialNDIM];
                unsigned int spatial_size[SpatialNDIM];
                int physical_index[SpatialNDIM];
                int del_origin[SpatialNDIM];
                unsigned int total_size = 1;


                for(unsigned int i=0; i<SpatialNDIM; i++)
                {
                    spatial_size[i] = fUpperLimits[i+2] - fLowerLimits[i+2];
                    total_size *= spatial_size[i];
                }


                for(unsigned int i=0; i<total_size; i++)
                {
                    KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(i, spatial_size, storage_index);
                    for(unsigned int j=0; j<SpatialNDIM; j++)
                    {
                        physical_index[j] = (int)storage_index[j] + fLowerLimits[j+2];
                        del_origin[j] = physical_index[j] + fShift[j];
                        fCoord[j] = (del_origin[j])*fLength;
                    }

                    if(OriginIsSource) //origin is the source, so we compute an outgoing response
                    {
                        fKernel.SetTargetOrigin(fCoord);
                    }

                    if(!OriginIsSource) //origin is not the source, but rather the target, so we compute an in going response function
                    {
                        fKernel.SetSourceOrigin(fCoord);
                    }

                    if(IsOutsideZeroMask(del_origin))
                    {
                        ComputeCoefficientsAtPoint(physical_index);
                    }
                }
            }
        }

    protected:

        virtual bool IsOutsideZeroMask(const int* index) const
        {
            if(fZeroMaskSize > 0)
            {
                for(unsigned int i=0; i<SpatialNDIM; i++)
                {
                    if( std::fabs(index[i]) > fZeroMaskSize ){return true;};
                }
                return false;
            }
            else
            {
                return true;
            }
        }

        virtual void ComputeCoefficientsAtPoint(const int* physical_index)
        {
            std::complex<double> result;
            int index[SpatialNDIM + 2];

            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                index[i+2] = physical_index[i];
            }


            for(unsigned int target=0; target<fNTerms; target++)
            {
                index[0] = target;
                result = std::complex<double>(0,0);
                for(unsigned int source=0; source<fNTerms; source++)
                {
                    index[1] = source;
                    if(fKernel.IsPhysical(source,target))
                    {
                        (*this->fOutput)[index] = fKernel.GetResponseFunction(source,target);
                    }
                }
            }
        }


        int fVerbose;
        bool fInitialized;

        double fLength; //side length of the region
        int fZeroMaskSize;
        unsigned int fNTerms;

        int fLowerLimits[SpatialNDIM + 2];
        int fUpperLimits[SpatialNDIM + 2];
        double fOrigin[SpatialNDIM];
        int fShift[SpatialNDIM];

        //response kernel calculator
        KernelType fKernel;

        //scratch space
        mutable double fCoord[SpatialNDIM];
};


}

#endif /* __KFMKernelResponseArray_H__ */
