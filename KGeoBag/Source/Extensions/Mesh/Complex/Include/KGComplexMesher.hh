#ifndef KGeoBag_KGComplexMesher_hh_
#define KGeoBag_KGComplexMesher_hh_

#include "KGMesherBase.hh"

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"

namespace KGeoBag
{

    class KGComplexMesher :
        virtual public KGMesherBase
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        protected:
            KGComplexMesher();

        public:
            virtual ~KGComplexMesher();

            static void DiscretizeInterval( double interval, int nSegments, double power, std::vector< double >& segments );

        protected:
            void AddElement( KGMeshElement* e );
            void RefineAndAddElement( KGMeshRectangle* rectangle, int nElements_A, double power_A, int nElements_B, double power_B );
            void RefineAndAddElement( KGMeshTriangle* triangle, int nElements, double power );
            void RefineAndAddElement( KGMeshWire* wire, int nElements, double power );
    };
}

#endif
