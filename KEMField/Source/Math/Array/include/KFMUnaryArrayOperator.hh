#ifndef KFMUnaryArrayOperator_H__
#define KFMUnaryArrayOperator_H__

#include "KFMArrayOperator.hh"

namespace KEMField
{

/**
*
*@file KFMUnaryArrayOperator.hh
*@class KFMUnaryArrayOperator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Sep 25 10:25:37 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ArrayType, unsigned int NDIM> class KFMUnaryArrayOperator : public KFMArrayOperator<ArrayType, NDIM>
{
  public:
    KFMUnaryArrayOperator() : fInput(nullptr), fOutput(nullptr)
    {
        ;
    };
    ~KFMUnaryArrayOperator() override
    {
        ;
    };

    virtual void SetInput(KFMArrayWrapper<ArrayType, NDIM>* in)
    {
        fInput = in;
    };
    virtual void SetOutput(KFMArrayWrapper<ArrayType, NDIM>* out)
    {
        fOutput = out;
    };

    virtual KFMArrayWrapper<ArrayType, NDIM>* GetInput()
    {
        return fInput;
    };
    virtual KFMArrayWrapper<ArrayType, NDIM>* GetOutput()
    {
        return fOutput;
    };

    void Initialize() override
    {
        ;
    };

    void ExecuteOperation() override = 0;

  protected:
    KFMArrayWrapper<ArrayType, NDIM>* fInput;
    KFMArrayWrapper<ArrayType, NDIM>* fOutput;
};

}  // namespace KEMField


#endif /* __KFMUnaryArrayOperator_H__ */
