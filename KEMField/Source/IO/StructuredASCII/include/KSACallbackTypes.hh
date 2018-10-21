#ifndef KSACallbackTypes_HH__
#define KSACallbackTypes_HH__


#define CALL_MEMBER_FN(object_ptr, ptrToMember)  (object_ptr->*(ptrToMember))


namespace KEMField{

/**
*
*@file KSACallbackTypes.hh
*@class KSACallbackTypes
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Dec 23 14:00:39 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//GETTERS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


template<typename CallType, typename ReturnType, ReturnType (CallType::*memberFunction)() const>
struct KSAConstantReturnByValueGet
{
    ReturnType operator()(const CallType* ptr)
    {
        return CALL_MEMBER_FN(ptr, memberFunction)();
    };
};


template<typename CallType, typename ReturnType, const ReturnType* (CallType::*memberFunction)() const>
struct KSAConstantReturnByPointerGet
{
    const ReturnType* operator()(const CallType* ptr)
    {
        return CALL_MEMBER_FN(ptr, memberFunction)();
    };
};

template<typename CallType, typename ReturnType, void (CallType::*memberFunction)(ReturnType& ) const>
struct KSAConstantReturnByPassedReferenceGet
{
    void operator()(const CallType* ptr, ReturnType& val_ref)
    {
        CALL_MEMBER_FN(ptr, memberFunction)(val_ref);
    };
};

template< typename CallType, typename ReturnType, void (CallType::*memberFunction)(ReturnType* ) const>
struct KSAConstantReturnByPassedPointerGet
{
    void operator()(const CallType* ptr, ReturnType* val_ptr)
    {
        CALL_MEMBER_FN(ptr, memberFunction)(val_ptr);
    };
};

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
////SETTERS
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

template< typename CallType, typename SetType, void (CallType::*memberFunction)(SetType) >
struct KSAPassByValueSet
{
    void operator()(CallType* ptr, SetType val)
    {
        CALL_MEMBER_FN(ptr, memberFunction)(val);
    };
};



template< typename CallType, typename SetType, void (CallType::*memberFunction)(const SetType&) >
struct KSAPassByConstantReferenceSet
{
    void operator()(CallType* ptr, const SetType& val)
    {
        CALL_MEMBER_FN(ptr, memberFunction)(val);
    };
};


template< typename CallType, typename SetType, void (CallType::*memberFunction)(const SetType*) >
struct KSAPassByConstantPointerSet
{
    void operator()(CallType* ptr, const SetType* val)
    {
        CALL_MEMBER_FN(ptr, memberFunction)(val);
    };
};



}//end of kemfield namespace

#endif /* KSACallbackTypes_H__ */
