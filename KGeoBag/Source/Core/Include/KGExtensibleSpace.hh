#ifndef KGCORE_HH_
#error "do not include KGExtensibleSpace.hh directly; include KGCore.hh instead."
#else

namespace KGeoBag
{

    class KGExtensibleSpace
    {
        protected:
            KGExtensibleSpace();

        public:
            virtual ~KGExtensibleSpace();

        public:
            //********
            //clonable
            //********

        public:
            virtual KGExtensibleSpace* Clone( KGSpace* aParent = NULL ) const = 0;

            //*********
            //visitable
            //*********

        public:
            virtual void Accept( KGVisitor* aVisitor ) = 0;
    };

}

#endif
