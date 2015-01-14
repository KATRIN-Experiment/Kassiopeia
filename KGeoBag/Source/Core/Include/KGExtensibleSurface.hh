#ifndef KGCORE_HH_
#error "do not include KGExtensibleSurface.hh directly; include KGCore.hh instead."
#else

namespace KGeoBag
{

    class KGExtensibleSurface
    {
        protected:
            KGExtensibleSurface();

        public:
            virtual ~KGExtensibleSurface();

        public:
            //********
            //clonable
            //********

        public:
            virtual KGExtensibleSurface* Clone( KGSurface* aParent = NULL ) const = 0;

            //*********
            //visitable
            //*********

        public:
            virtual void Accept( KGVisitor* aVisitor ) = 0;
    };

}

#endif
