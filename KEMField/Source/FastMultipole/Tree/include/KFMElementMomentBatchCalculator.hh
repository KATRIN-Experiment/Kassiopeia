#ifndef KFMElectrostaticMomentCalculatorBatch_HH__
#define KFMElectrostaticMomentCalculatorBatch_HH__

#include <sstream>

#ifndef KEMFIELD_MULTIPOLE_BUFFER_SIZE_MB
    #define KEMFIELD_MULTIPOLE_BUFFER_SIZE_MB 32 //size of buffer in megabytes
#endif

namespace KEMField{

/**
*
*@file KFMElementMomentBatchCalculator.hh
*@class KFMElementMomentBatchCalculator
*@brief abstract interface to an object which computes the expansion moments of a bunch of boundary elements and stores them in an array
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jun  7 10:06:57 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElementMomentBatchCalculator
{
    public:
        KFMElementMomentBatchCalculator();
        virtual ~KFMElementMomentBatchCalculator();

        void SetBufferSizeInBytes(long buff_size){fMaxBufferSizeInBytes = buff_size;};
        long GetBufferSizeInBytes() const {return fMaxBufferSizeInBytes;};

        //set/get the dimensionality of the space
        virtual void SetDimensions(int dim);
        int GetDimension() const {return fDim;};

        //set/get the degree of the expansion
        virtual void SetDegree(int l_max);
        int GetDegree() const {return fDegree;};

        //set the size of the sub-array of elements that are valid work objects
        //typically this will equal to fNMaxItems, but not necessarily so
        virtual void SetSizeOfValidElements(const unsigned int& valid){fValidSize = valid;};

        //size of the sub-array of valid items in the buffers
        unsigned int GetSizeOfValidElements(){return fValidSize;};

        //initalize the object
        virtual void Initialize();

        //execute the operation to fill the Moment buffer
        virtual void ComputeMoments() = 0; //Moment buffer should be filled using the real basis!

        //used to retrieve/modify the list of primitives we want to process
        int* GetIDBuffer(){return fIDBuffer;};

        //used to retrieve/modify the list of origins about which to do the expansions
        double* GetOriginBuffer(){return fOriginBuffer;};

        //used to retreive the resulting moments
        double* GetMomentBuffer(){return fMomentBuffer;};

        //the total size of the id buffer
        unsigned int GetIDBufferSize(){return fNMaxItems;};

        //the total size of the origin buffer
        unsigned int GetOriginBufferSize(){return fDim*fNMaxItems;};

        //the total size of the moment buffer = 2*fNMaxItems*fStride
        unsigned int GetMomentBufferSize(){return 2*fNMaxItems*fStride;};

        //the number of elements which correspond to the Moment moments of one item
        unsigned int GetMomentBufferStride(){return fStride;};

        virtual void ReleaseMemory();

    protected:

        unsigned int fMaxBufferSizeInBytes;
        bool fInitialized;

        int fDegree;
        int fDim;
        unsigned int fNMaxItems;
        unsigned int fStride;
        unsigned int fValidSize;

        int* fIDBuffer;//size = fNMaxItems
        double* fOriginBuffer;   //size = fDim*fNMaxItems
        //size = 2*fNMaxItems*fStride...we store the Moments in interleaved format
        //i.e. if Moment (l,m) is stored at index i, then the real part is at fMomentBuffer[i]
        //and the imaginary part is stored at fMomentBuffer[i+1]
        double* fMomentBuffer;
};


}//end of KEMField

#endif /* KFMElectrostaticMomentCalculatorBatch_H__ */
