#include <cmath>

#include "KFMElementMomentBatchCalculator.hh"

#include <iostream>

#include "KFMMessaging.hh"


namespace KEMField
{

KFMElementMomentBatchCalculator::KFMElementMomentBatchCalculator()
{
    unsigned int buffer_mb = KEMFIELD_MULTIPOLE_BUFFER_SIZE_MB;
    fMaxBufferSizeInBytes = buffer_mb*1024*1024;

    fIDBuffer = NULL;
    fOriginBuffer = NULL;
    fMomentBuffer = NULL;

    fDim = 3; //default is three spatial dimensions
    fDegree = 0;
    fNMaxItems = 0;
    fStride = 0;
    fValidSize = 0;

    fInitialized = false;
}

KFMElementMomentBatchCalculator::~KFMElementMomentBatchCalculator()
{
    delete[] fIDBuffer; fIDBuffer = NULL;
    delete[] fOriginBuffer; fOriginBuffer = NULL;
    delete[] fMomentBuffer; fMomentBuffer = NULL;
}

void KFMElementMomentBatchCalculator::SetDegree(int l_max)
{
    fDegree = std::fabs(l_max);
    fStride = (fDegree+1)*(fDegree+2)/2;

    ReleaseMemory();
    fInitialized = false;
}

void
KFMElementMomentBatchCalculator::SetDimensions(int dim)
{
    fDim = std::fabs(dim);
    ReleaseMemory();
    fInitialized = false;
}

void KFMElementMomentBatchCalculator::Initialize()
{
    if(!fInitialized)
    {
        //first lets figure out how many elements we can process at a time
        unsigned int bytes_per_element = fStride*2*sizeof(double);
        fNMaxItems = fMaxBufferSizeInBytes/bytes_per_element;

        if(fNMaxItems != 0)
        {
            fIDBuffer = new int[fNMaxItems];
            fMomentBuffer = new double[2*fStride*fNMaxItems];
            fOriginBuffer = new double[fDim*fNMaxItems];
        }
        else
        {
            //warning
            std::stringstream ss;
            ss << "Buffer size of ";
            ss << fMaxBufferSizeInBytes;
            ss <<" bytes is not large enough for a single element. ";
            ss <<"Required bytes per element = "<<bytes_per_element<<". Aborting.";
            kfmout<<ss.str()<<std::endl;
        }
        fInitialized = true;
    }
}

void KFMElementMomentBatchCalculator::ReleaseMemory()
{
    delete[] fIDBuffer; fIDBuffer = NULL;
    delete[] fOriginBuffer; fOriginBuffer = NULL;
    delete[] fMomentBuffer; fMomentBuffer = NULL;
}


}//end of KEMField
