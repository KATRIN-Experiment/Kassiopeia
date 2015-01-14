#ifndef KFMBinaryArrayOperator_H__
#define KFMBinaryArrayOperator_H__

#include "KFMArrayOperator.hh"

namespace KEMField{

/**
*
*@file KFMBinaryArrayOperator.hh
*@class KFMBinaryArrayOperator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Sep 25 10:25:37 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename T, unsigned int NDIM>
class KFMBinaryArrayOperator: public KFMArrayOperator<T,NDIM>
{
    public:
        KFMBinaryArrayOperator():fFirstInput(NULL),fSecondInput(NULL),fOutput(NULL){;};
        virtual ~KFMBinaryArrayOperator(){;};

        virtual void SetFirstInput(KFMArrayWrapper<T,NDIM>* in){fFirstInput = in;};
        virtual void SetSecondInput(KFMArrayWrapper<T,NDIM>* in){fSecondInput = in;};
        virtual void SetOutput(KFMArrayWrapper<T,NDIM>* out){fOutput = out;};

        virtual KFMArrayWrapper<T,NDIM>* GetFirstInput(){return fFirstInput;};
        virtual KFMArrayWrapper<T,NDIM>* GetSecondInput(){return fSecondInput;};
        virtual KFMArrayWrapper<T,NDIM>* GetOutput(){return fOutput;};

        virtual void Initialize(){;};

        virtual void ExecuteOperation() = 0;

    protected:

        KFMArrayWrapper<T,NDIM>* fFirstInput;
        KFMArrayWrapper<T,NDIM>* fSecondInput;
        KFMArrayWrapper<T,NDIM>* fOutput;
};

}


#endif /* __KFMBinaryArrayOperator_H__ */
