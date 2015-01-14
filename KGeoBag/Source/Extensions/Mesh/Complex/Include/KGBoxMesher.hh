#ifndef KGeoBag_KGBoxMesher_hh_
#define KGeoBag_KGBoxMesher_hh_

#include "KGBox.hh"

#include "KGComplexMesher.hh"

namespace KGeoBag
{
    class KGBoxMesher :
        virtual public KGComplexMesher,
        public KGBox::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGBoxMesher()
            {
            }
            virtual ~KGBoxMesher()
            {
            }

        protected:
            void VisitBox( KGBox* box );
    };

}

#endif
