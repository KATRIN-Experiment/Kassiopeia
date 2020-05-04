#ifndef KFMArrayFillingOperator_H__
#define KFMArrayFillingOperator_H__

#include "KFMArrayOperator.hh"

#include <iostream>

namespace KEMField
{

/**
*
*@file KFMArrayFillingOperator.hh
*@class KFMArrayFillingOperator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Sep 25 10:25:37 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename T, unsigned int NDIM> class KFMArrayFillingOperator : public KFMArrayOperator<T, NDIM>
{
  public:
    KFMArrayFillingOperator() : fOutput(nullptr)
    {
        ;
    };
    ~KFMArrayFillingOperator() override
    {
        ;
    };

    virtual void SetOutput(KFMArrayWrapper<T, NDIM>* out)
    {
        fOutput = out;
        //            std::cout<<"Setting output to: "<<fOutput<<std::endl;
        //            std::cout<<"output raw data ptr = "<<fOutput->GetData()<<std::endl;
    };

    virtual KFMArrayWrapper<T, NDIM>* GetOutput()
    {
        return fOutput;
    };

    void Initialize() override
    {
        ;
    };

    void ExecuteOperation() override = 0;

  protected:
    KFMArrayWrapper<T, NDIM>* fOutput;
};

}  // namespace KEMField


#endif /* __KFMArrayFillingOperator_H__ */
