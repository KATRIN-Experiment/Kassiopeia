#ifndef KCALLABLEFUNCTOR_H_
#define KCALLABLEFUNCTOR_H_

#include "KCallable.h"

namespace katrin
{

    template< class XSignature, class XFunctor >
    class KCallableFunctor :
        public KCallable< XSignature >
    {
        public:
            KCallableFunctor( const XFunctor& aFunctor ) :
                    fFunctor( aFunctor )
            {
            }

            KCallableFunctor* Clone() const
            {
                return new KCallableFunctor( *this );
            }

            template< class XR >
            XR operator()()
            {
                return fFunctor();
            }

            template< class XR, class XA1 >
            XR operator()( XA1 a1 )
            {
                return fFunctor( a1 );
            }

            template< class XR, class XA1, class XA2 >
            XR operator()( XA1 a1, XA2 a2 )
            {
                return fFunctor( a1, a2 );
            }

            template< class XR, class XA1, class XA2, class XA3 >
            XR operator()( XA1 a1, XA2 a2, XA3 a3 )
            {
                return fFunctor( a1, a2, a3 );
            }

            template< class XR, class XA1, class XA2, class XA3, class XA4 >
            XR operator()( XA1 a1, XA2 a2, XA3 a3, XA4 a4 )
            {
                return fFunctor( a1, a2, a3, a4 );
            }

        private:
            XFunctor fFunctor;
    };

}

#endif
