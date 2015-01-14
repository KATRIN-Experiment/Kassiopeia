#ifndef KFUNCTION_H_
#define KFUNCTION_H_

#include "KCallable.h"

namespace katrin
{

    template< class XSignature >
    class Function
    {
        public:
            Function();
            Function( const Function& aCopy );

            template< class XR >
            XR operator()()
            {
                return (*fCallable)();
            }

            template< class XR, class XA1 >
            XR operator()( XA1 a1 )
            {
                return (*fCallable)( a1 );
            }

            template< class XR, class XA1, class XA2 >
            XR operator()( XA1 a1, XA2 a2 )
            {
                return (*fCallable)( a1, a2 );
            }

            template< class XR, class XA1, class XA2, class XA3 >
            XR operator()( XA1 a1, XA2 a2, XA3 a3 )
            {
                return (*fCallable)( a1, a2, a3 );
            }

            template< class XR, class XA1, class XA2, class XA3, class XA4 >
            XR operator()( XA1 a1, XA2 a2, XA3 a3, XA4 a4 )
            {
                return (*fCallable)( a1, a2, a3, a4 );
            }

        private:
            KCallable< XSignature >* fCallable;
    };

}



#endif
