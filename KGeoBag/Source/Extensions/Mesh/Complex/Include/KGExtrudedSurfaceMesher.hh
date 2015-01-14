#ifndef KGeoBag_KGExtrudedSurfaceMesher_hh_
#define KGeoBag_KGExtrudedSurfaceMesher_hh_

#include "KGExtrudedSurface.hh"

#include "KGComplexMesher.hh"

namespace KGeoBag
{
    class KGExtrudedSurfaceMesher :
        virtual public KGComplexMesher,
        public KGExtrudedSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGExtrudedSurfaceMesher() :
                    fExtrudedObject( NULL ),
                    fIsModifiable( false )
            {
            }
            virtual ~KGExtrudedSurfaceMesher()
            {
            }

        protected:
            void VisitWrappedSurface( KGExtrudedSurface* extrudedSurface );

            void Discretize( KGExtrudedObject* object );
            void DiscretizeSegment( const KGExtrudedObject::Line* line, const unsigned int nDisc, std::vector< std::vector< double > >& coords, unsigned int& counter );
            void DiscretizeSegment( const KGExtrudedObject::Arc* arc, const unsigned int nDisc, std::vector< std::vector< double > >& coords, unsigned int& counter );
            void DiscretizeEnclosedEnds( std::vector< std::vector< double > >& iCoords, std::vector< std::vector< double > >& oCoords, unsigned int nDisc );
            void DiscretizeLoopEnds();

            virtual void ModifyInnerSegment( int, std::vector< std::vector< double > >& )
            {
            }
            virtual void ModifyOuterSegment( int, std::vector< std::vector< double > >& )
            {
            }
            virtual void ModifySurface( std::vector< std::vector< double > >&, std::vector< std::vector< double > >&, std::vector< unsigned int >&, std::vector< unsigned int >& )
            {
            }

            KGExtrudedObject* fExtrudedObject;

            bool fIsModifiable;
    };
}

#endif /* KGEXTRUDEDSURFACEDISCRETIZER_DEF */
