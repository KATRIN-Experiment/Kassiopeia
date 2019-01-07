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

    public KGSurface::Visitor,
	public KGSpace::Visitor,
	public KGVisitor

    {
        public:
    	virtual void VisitSurface( KGSurface* aSurface );
        virtual void VisitSpace( KGSpace* aSpace );

        public:
            KGDiscreteRotationalMesher();
            virtual ~KGDiscreteRotationalMesher();

      void SetAxialAngle(double d) { fAxialAngle = d; }
      void SetAxialCount(unsigned int i) { fAxialCount = i; }


    private:
      void MeshAxialSurface( KGExtendedSurface< KGAxialMesh >* aSurface );
      void MeshAxialSpace( KGExtendedSpace< KGAxialMesh >* aSpace );
      void AddAxialMeshElement(KGAxialMeshElement* e);

      void AddAxialMeshLoop(KGAxialMeshLoop* l);
      void AddAxialMeshRing(KGAxialMeshRing* r);

      KGDiscreteRotationalMeshElementVector* fCurrentElements;


      double fAxialAngle;
      unsigned int fAxialCount;
    };
}

#endif
