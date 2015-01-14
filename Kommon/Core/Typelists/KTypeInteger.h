#ifndef KINTEGERTYPE_H_
#define KINTEGERTYPE_H_

namespace katrin
{

    template< int XValue >
    class KTypeInteger
    {
        public:
            enum { Value = XValue };
    };


    template< class XLeft, class XRight >
    class KPlus;

    template< int XLeftValue, int XRightValue >
    class KPlus< KTypeInteger< XLeftValue >, KTypeInteger< XRightValue > >
    {
        public:
            typedef KTypeInteger< XLeftValue + XRightValue > Type;
    };


    template< class XLeft, class XRight >
    class KMinus;

    template< int XLeftValue, int XRightValue >
    class KMinus< KTypeInteger< XLeftValue >, KTypeInteger< XRightValue > >
    {
        public:
            typedef KTypeInteger< XLeftValue - XRightValue > Type;
    };


}


#endif
