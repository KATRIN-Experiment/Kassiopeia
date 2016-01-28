#include "KGConicalWireArrayMesher.hh"

#include "KGMeshElement.hh"

#include "KGComplexMesher.hh"

namespace KGeoBag
{
    void KGConicalWireArrayMesher::VisitWrappedSurface(KGWrappedSurface< KGConicalWireArray >* conicalWireArraySurface)
    {
      KTransformation transform;
      transform.SetRotationAxisAngle(conicalWireArraySurface->GetObject()->GetThetaStart(),0.,0.);

      double tAngleStep = 2*KConst::Pi()/conicalWireArraySurface->GetObject()->GetNWires();
      unsigned int tAngleIt( 0 ), tLongitudinalDiscIt( 0 );

      std::vector<double> segments( conicalWireArraySurface->GetObject()->GetNDisc(), 0. );

      KGComplexMesher::DiscretizeInterval(
    		  conicalWireArraySurface->GetObject()->GetLength(),
    		  conicalWireArraySurface->GetObject()->GetNDisc(),
    		  conicalWireArraySurface->GetObject()->GetNDiscPower(),
    		  segments);

      KThreeVector startPoint, endPoint, v1, v2, distanceVector, unitVector;

      for( tAngleIt=0; tAngleIt<conicalWireArraySurface->GetObject()->GetNWires(); tAngleIt++ )
      {
		  startPoint.SetComponents( conicalWireArraySurface->GetObject()->GetR1()*cos(tAngleStep*tAngleIt),
				  conicalWireArraySurface->GetObject()->GetR1()*sin(tAngleStep*tAngleIt),
				  conicalWireArraySurface->GetObject()->GetZ1());
		  endPoint.SetComponents( conicalWireArraySurface->GetObject()->GetR2()*cos(tAngleStep*tAngleIt),
			  conicalWireArraySurface->GetObject()->GetR2()*sin(tAngleStep*tAngleIt),
			  conicalWireArraySurface->GetObject()->GetZ2());

		  distanceVector = endPoint - startPoint;
		  unitVector.SetComponents( distanceVector/distanceVector.Magnitude() );

		  endPoint = startPoint;

		  for( tLongitudinalDiscIt=0; tLongitudinalDiscIt<conicalWireArraySurface->GetObject()->GetNDisc(); tLongitudinalDiscIt++ )
		  {
			  endPoint += segments[tLongitudinalDiscIt]*unitVector;
			  KGMeshWire singleWire( startPoint, endPoint, conicalWireArraySurface->GetObject()->GetDiameter());
			  singleWire.Transform(transform);
			  KGMeshWire* w = new KGMeshWire(singleWire);
			  AddElement(w);
			  startPoint = endPoint;
		  }
      }
    }
}
