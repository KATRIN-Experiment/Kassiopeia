#ifndef KCALLABLEFREEFUNCTION_H_
#define KCALLABLEFREEFUNCTION_H_

#include "KCallable.h"

namespace katrin
{

    template< class XSignature >
    class KCallableFreeFunction :
        public KCallable< XSignature >
    {
        public:
            KCallableFreeFunction( typename KCallable< XSignature >::FreeFunctionPointerType aFreeFunction ) :
                fFreeFunction( aFreeFunction )
            {
            }
            KCallableFreeFunction( typename KCallable< XSignature >::FreeFunctionReferenceType aFreeFunction ) :
                fFreeFunction( &FreeFunction )
            {
            }

            KCallableFreeFunction* Clone() const
            {
                return new KCallableFreeFunction( *this );
            }

            template< class XR >
            XR operator()()
            {
                return (*fFreeFunction)();
            }

            template< class XR, class XA1 >
            XR operator()( XA1 a1 )
            {
                return (*fFreeFunction)( a1 );
            }

            template< class XR, class XA1, class XA2 >
            XR operator()( XA1 a1, XA2 a2 )
            {
                return (*fFreeFunction)( a1, a2 );
            }

            template< class XR, class XA1, class XA2, class XA3 >
            XR operator()( XA1 a1, XA2 a2, XA3 a3 )
            {
                return (*fFreeFunction)( a1, a2, a3 );
            }

            template< class XR, class XA1, class XA2, class XA3, class XA4 >
            XR operator()( XA1 a1, XA2 a2, XA3 a3, XA4 a4 )
            {
                return (*fFreeFunction)( a1, a2, a3, a4 );
            }

        private:
            typename KCallable< XSignature >::FreeFunctionPointerType fFreeFunction;
    };
}

#endif
