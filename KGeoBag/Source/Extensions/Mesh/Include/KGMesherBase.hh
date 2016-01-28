#ifndef KGeoBag_KGMesherBase_hh_
#define KGeoBag_KGMesherBase_hh_

#include "KGCore.hh"

#include "KGMesh.hh"

namespace KGeoBag
{

    class KGMesherBase :
        public KGVisitor,
        public KGExtendedSurface< KGMesh >::Visitor,
        public KGExtendedSpace< KGMesh >::Visitor
    {
        protected:
            KGMesherBase();

        public:
            virtual ~KGMesherBase();

        public:
            void VisitExtendedSurface( KGExtendedSurface< KGMesh >* aSurface );
            void VisitExtendedSpace( KGExtendedSpace< KGMesh >* aSpace );

        protected:
            KGMeshElementVector* fCurrentElements;
            KGExtendedSurface< KGMesh >* fCurrentSurface;
            KGExtendedSpace< KGMesh >* fCurrentSpace;
    };
}

#endif
