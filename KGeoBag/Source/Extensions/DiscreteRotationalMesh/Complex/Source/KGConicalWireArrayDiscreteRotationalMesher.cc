#include "KGConicalWireArrayDiscreteRotationalMesher.hh"

#include "KGDiscreteRotationalMeshElement.hh"

#include "KGComplexMesher.hh"

namespace KGeoBag
{
    void KGConicalWireArrayDiscreteRotationalMesher::VisitWrappedSurface(KGWrappedSurface< KGConicalWireArray >* conicalWireArraySurface)
    {
      KTransformation transform;
      transform.SetRotationAxisAngle(conicalWireArraySurface->GetObject()->GetThetaStart(),0.,0.);

      std::vector<double> segments(conicalWireArraySurface->GetObject()->GetNDisc(),0.);

      KGComplexMesher::DiscretizeInterval(conicalWireArraySurface->GetObject()->GetLength(),conicalWireArraySurface->GetObject()->GetNDisc(),2.,segments);

      KThreeVector start(conicalWireArraySurface->GetObject()->GetR1(),
			 0.,
			 conicalWireArraySurface->GetObject()->GetZ1());

      KThreeVector end = start;

      KThreeVector v1( conicalWireArraySurface->GetObject()->GetR1(), 0., conicalWireArraySurface->GetObject()->GetZ1());
      KThreeVector v2( conicalWireArraySurface->GetObject()->GetR2(), 0., conicalWireArraySurface->GetObject()->GetZ2());
      KThreeVector vd = v2-v1;
      double dist = vd.Magnitude();

      KThreeVector unit(vd/dist);

      for (unsigned int i=0;i<conicalWireArraySurface->GetObject()->GetNDisc();i++)
      {
	start = end;
	end += segments[i]*unit;
	KGMeshWire singleWire(start,
			      start + segments[i]*unit,
			      conicalWireArraySurface->GetObject()->GetDiameter());
	singleWire.Transform(transform);
	KGDiscreteRotationalMeshWire* w = new KGDiscreteRotationalMeshWire(singleWire);
	w->NumberOfElements(conicalWireArraySurface->GetObject()->GetNWires());
	fCurrentElements->push_back(w);
      }
    }
}
