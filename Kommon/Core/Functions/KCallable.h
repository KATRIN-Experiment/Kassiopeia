#ifndef KCALLABLE_H_
#define KCALLABLE_H_

#include "KTypeList.h"

namespace katrin
{

    template< class XSignature >
    class KCallable;

    template< class XR >
    class KCallable< XR() >
    {
        public:
            typedef XR R;
            typedef KTypeNull ArgListType;

            typedef XR (*FreeFunctionPointerType)();
            typedef XR (&FreeFunctionReferenceType)();
            typedef KTypeNull MemberFunctionPointerType;

            virtual R operator()() = 0;
            virtual KCallable* Clone() = 0;
            virtual ~KCallable()
            {
            }
    };

    template< class XR, class XA1 >
    class KCallable< XR( XA1 ) >
    {
        public:
            typedef XR R;
            typedef XA1 A1;
            typedef KTYPELIST1( A1 )ArgListType;

            typedef R(*FreeFunctionPointerType)( A1 );
            typedef R(&FreeFunctionReferenceType)( A1 );
            typedef R(A1::*MemberFunctionPointerType)();

            virtual R operator()( A1 ) = 0;
            virtual KCallable* Clone() = 0;
            virtual ~KCallable()
            {
            }
    };

    template< class XR, class XA1, class XA2 >
    class KCallable< XR( XA1, XA2 ) >
    {
        public:
            typedef XR R;
            typedef XA1 A1;
            typedef XA2 A2;
            typedef KTYPELIST2( A1, A2 )ArgListType;

            typedef R(*FreeFunctionPointerType)( A1, A2 );
            typedef R(&FreeFunctionReferenceType)( A1, A2 );
            typedef R(A1::*MemberFunctionPointerType)( A2 );

            virtual R operator()( A1, A2 ) = 0;
            virtual KCallable* Clone() = 0;
            virtual ~KCallable()
            {
            }

    };

    template< class XR, class XA1, class XA2, class XA3 >
    class KCallable< XR( XA1, XA2, XA3 ) >
    {
        public:
            typedef XR R;
            typedef XA1 A1;
            typedef XA2 A2;
            typedef XA3 A3;
            typedef KTYPELIST3( A1, A2, A3 )ArgListType;

            typedef R(*FreeFunctionPointerType)( A1, A2, A3 );
            typedef R(&FreeFunctionReferenceType)( A1, A2, A3 );
            typedef R(A1::*MemberFunctionPointerType)( A2, A3 );

            virtual R operator()( A1, A2 ) = 0;
            virtual KCallable* Clone() = 0;
            virtual ~KCallable()
            {
            }
    };

}

#endif
