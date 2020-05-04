#ifndef KVMFieldWrapper_H
#define KVMFieldWrapper_H


#include "KVMField.hh"

namespace KEMField
{


/**
*
*@file KVMFieldWrapper.hh
*@class KVMFieldWrapper
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jul 11 16:20:50 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

#define CALL_MEMBER_FN(object_ptr, ptrToMember) (object_ptr->*(ptrToMember))

template<class T, void (T::*memberFunction)(const double*, double*) const> class KVMFieldWrapper : public KVMField
{
  public:
    KVMFieldWrapper(T* ptr_to_object, unsigned int dimDomain, unsigned int dimRange) :
        fObjectPtr(ptr_to_object),
        fDimDomain(dimDomain),
        fDimRange(dimRange)
    {
        ;
    };

    ~KVMFieldWrapper() override
    {
        ;
    };

    unsigned int GetNDimDomain() const override
    {
        return fDimDomain;
    };
    unsigned int GetNDimRange() const override
    {
        return fDimRange;
    };

    void Evaluate(const double* in, double* out) const override
    {
        return CALL_MEMBER_FN(fObjectPtr, memberFunction)(in, out);
    }

  private:
    T* fObjectPtr;
    unsigned int fDimDomain;
    unsigned int fDimRange;
};


}  // namespace KEMField

#endif /* KVMFieldWrapper_H */
