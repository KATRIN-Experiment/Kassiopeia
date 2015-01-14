#ifndef KFMMultidimensionalFastFourierTransform_HH__
#define KFMMultidimensionalFastFourierTransform_HH__

#include <cstring>

#include "KFMArrayWrapper.hh"
#include "KFMFastFourierTransform.hh"

namespace KEMField
{

/*
*
*@file KFMMultidimensionalFastFourierTransform.hh
*@class KFMMultidimensionalFastFourierTransform
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov 27 00:01:00 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM>
class KFMMultidimensionalFastFourierTransform: public KFMUnaryArrayOperator< std::complex<double>, NDIM >
{
    public:
        KFMMultidimensionalFastFourierTransform()
        {
            for(unsigned int i=0; i<NDIM; i++)
            {
                fDimensionSize[i] = 0;
                fWorkspace[i] = NULL;
                fWorkspaceWrapper[i] = NULL;
                fTransformCalculator[i] = NULL;
            }

            fIsValid = false;
            fInitialized = false;
            fForward = true;
        };

        virtual ~KFMMultidimensionalFastFourierTransform()
        {
            DealocateWorkspace();
        };

        virtual void SetForward(){fForward = true;}
        virtual void SetBackward(){fForward = false;};

        virtual void Initialize()
        {
            if(DoInputOutputDimensionsMatch())
            {
                fIsValid = true;
                this->fInput->GetArrayDimensions(fDimensionSize);
            }
            else
            {
                fIsValid = false;
            }

            if(!fInitialized && fIsValid)
            {
                DealocateWorkspace();
                AllocateWorkspace();

                fInitialized = true;
            }
        }

        virtual void ExecuteOperation()
        {
            if(fIsValid && fInitialized)
            {

                unsigned int total_size = 1;
                for(unsigned int i=0; i<NDIM; i++)
                {
                    total_size *= fDimensionSize[i];
                    if(fForward)
                    {
                        fTransformCalculator[i]->SetForward();
                    }
                    else
                    {
                        fTransformCalculator[i]->SetBackward();
                    }
                }

                //if input and output point to the same array, don't bother copying data over
                if(this->fInput != this->fOutput)
                {
                    //the arrays are not identical so copy the input over to the output
                    std::memcpy( (void*) this->fOutput->GetData(), (void*) this->fInput->GetData(), total_size*sizeof(std::complex<double>) );
                 }

                unsigned int index[NDIM];
                unsigned int non_active_dimension_size[NDIM-1];
                unsigned int non_active_dimension_value[NDIM-1];
                unsigned int non_active_dimension_index[NDIM-1];

                //select the dimension on which to perform the FFT
                for(unsigned int d = 0; d < NDIM; d++)
                {
                    //now we loop over all dimensions not specified by d
                    //first compute the number of FFTs to perform
                    unsigned int n_fft = 1;
                    unsigned int count = 0;
                    for(unsigned int i = 0; i < NDIM; i++)
                    {
                        if(i != d)
                        {
                            n_fft *= fDimensionSize[i];
                            non_active_dimension_index[count] = i;
                            non_active_dimension_size[count] = fDimensionSize[i];
                            count++;
                        }
                    }

                    //loop over the number of FFTs to perform
                    for(unsigned int n=0; n<n_fft; n++)
                    {
                        //invert place in list to obtain indices of block in array
                        KFMArrayMath::RowMajorIndexFromOffset<NDIM-1>(n, non_active_dimension_size, non_active_dimension_value);

                        //copy the value of the non-active dimensions in to index
                        for(unsigned int i=0; i<NDIM-1; i++)
                        {
                            index[ non_active_dimension_index[i] ] = non_active_dimension_value[i];
                        }

                        unsigned int data_location;
                        //copy the row selected by the other dimensions
                        for(unsigned int i=0; i<fDimensionSize[d]; i++)
                        {
                            index[d] = i;
                            data_location = KFMArrayMath::OffsetFromRowMajorIndex<NDIM>(fDimensionSize, index);
                            (*(fWorkspaceWrapper[d]))[i] = (*(this->fOutput))[data_location];
                        }

                        //compute the FFT of the row selected
                        fTransformCalculator[d]->ExecuteOperation();

                        //copy the row selected back
                        for(unsigned int i=0; i<fDimensionSize[d]; i++)
                        {
                            index[d] = i;
                            data_location = KFMArrayMath::OffsetFromRowMajorIndex<NDIM>(fDimensionSize, index);
                            (*(this->fOutput))[data_location] = (*(fWorkspaceWrapper[d]))[i];
                        }
                    }
                }
            }
        }


    private:

        virtual void AllocateWorkspace()
        {
            unsigned int dim[1];
            for(unsigned int i=0; i<NDIM; i++)
            {
                dim[0] = fDimensionSize[i];
                fWorkspace[i] = new std::complex<double>[ fDimensionSize[i] ];
                fWorkspaceWrapper[i] = new KFMArrayWrapper< std::complex<double>, 1 >(fWorkspace[i], dim);
                fTransformCalculator[i] = new KFMFastFourierTransform();
                fTransformCalculator[i]->SetSize(fDimensionSize[i]);
                fTransformCalculator[i]->SetInput(fWorkspaceWrapper[i]);
                fTransformCalculator[i]->SetOutput(fWorkspaceWrapper[i]);
                fTransformCalculator[i]->Initialize();
            }
        }

        virtual void DealocateWorkspace()
        {
            for(unsigned int i=0; i<NDIM; i++)
            {
                delete[] fWorkspace[i]; fWorkspace[i] = NULL;
                delete fWorkspaceWrapper[i]; fWorkspaceWrapper[i] = NULL;
                delete fTransformCalculator[i]; fTransformCalculator[i] = NULL;
            }
        }

        virtual bool DoInputOutputDimensionsMatch()
        {
            unsigned int in[NDIM];
            unsigned int out[NDIM];

            this->fInput->GetArrayDimensions(in);
            this->fOutput->GetArrayDimensions(out);

            for(unsigned int i=0; i<NDIM; i++)
            {
                if(in[i] != out[i])
                {
                    return false;
                }
            }
            return true;
        }

        bool fIsValid;
        bool fForward;
        bool fInitialized;

        unsigned int fDimensionSize[NDIM];

        KFMFastFourierTransform* fTransformCalculator[NDIM];
        std::complex<double>* fWorkspace[NDIM];
        KFMArrayWrapper<std::complex<double>, 1>* fWorkspaceWrapper[NDIM];


};


}

#endif /* KFMMultidimensionalFastFourierTransform_H__ */
