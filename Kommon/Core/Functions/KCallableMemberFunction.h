#ifndef KCALLABLEMEMBERFUNCTION_H_
#define KCALLABLEMEMBERFUNCTION_H_

#include "KCallable.h"

namespace katrin
{

    template< class XSignature >
    class KCallableMemberFunction :
        public KCallable< XSignature >
    {
        public:
            KCallableMemberFunction( typename KCallable< XSignature >::MemberFunctionPointerType aMemberFunction ) :
                fMemberFunction( aMemberFunction )
            {
            }

            KCallableMemberFunction* Clone() const
            {
                return new KCallableMemberFunction( *this );
            }

            template< class XR, class XA1 >
            XR operator()( XA1 a1 )
            {
                return (a1->*fMemberFunction)();
            }

            template< class XR, class XA1, class XA2 >
            XR operator()( XA1 a1, XA2 a2 )
            {
                return (a1->*fMemberFunction)( a2 );
            }

            template< class XR, class XA1, class XA2, class XA3 >
            XR operator()( XA1 a1, XA2 a2, XA3 a3 )
            {
                return (a1->*fMemberFunction)( a2, a3 );
            }

            template< class XR, class XA1, class XA2, class XA3, class XA4 >
            XR operator()( XA1 a1, XA2 a2, XA3 a3, XA4 a4 )
            {
                return (a1->*fMemberFunction)( a2, a3, a4 );
            }

        private:
            typename KCallable< XSignature >::MemberFunctionPointerType fMemberFunction;
    };

}

#endif
