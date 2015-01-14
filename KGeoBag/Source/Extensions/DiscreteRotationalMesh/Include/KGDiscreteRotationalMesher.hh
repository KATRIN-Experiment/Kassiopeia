#ifndef KGeoBag_KGDiscreteRotationalMesher_hh_
#define KGeoBag_KGDiscreteRotationalMesher_hh_

#include "KGCore.hh"

#include "KGConicalWireArrayDiscreteRotationalMesher.hh"
#include "KGAxialMesh.hh"

namespace KGeoBag
{
    class KGAxialMeshElement;
    class KGAxialMeshLoop;
    class KGAxialMeshRing;

    class KGDiscreteRotationalMesher :
    public KGExtendedSurface< KGAxialMesh >::Visitor,
    public KGExtendedSpace< KGAxialMesh >::Visitor,
    virtual public KGConicalWireArrayDiscreteRotationalMesher
    {
        public:
            using KGDiscreteRotationalMesherBase::VisitExtendedSurface;
            using KGDiscreteRotationalMesherBase::VisitExtendedSpace;

            using KGConicalWireArrayDiscreteRotationalMesher::VisitWrappedSurface;

      void VisitExtendedSurface( KGExtendedSurface< KGAxialMesh >* aSurface );
      void VisitExtendedSpace( KGExtendedSpace< KGAxialMesh >* aSpace );

        public:
            KGDiscreteRotationalMesher();
            virtual ~KGDiscreteRotationalMesher();

      void SetAxialAngle(double d) { fAxialAngle = d; }
      void SetAxialCount(unsigned int i) { fAxialCount = i; }

    private:
      void AddAxialMeshElement(KGAxialMeshElement* e);

      void AddAxialMeshLoop(KGAxialMeshLoop* l);
      void AddAxialMeshRing(KGAxialMeshRing* r);

      double fAxialAngle;
      unsigned int fAxialCount;
    };
}

#endif
