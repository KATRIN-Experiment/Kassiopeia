#include <iostream>

#include "KGDiscreteRotationalMesher.hh"
#include "KGDiscreteRotationalAreaMesher.hh"

#include "KGAxialMeshLoop.hh"
#include "KGAxialMeshRing.hh"

namespace KGeoBag
{

  KGDiscreteRotationalMesher::KGDiscreteRotationalMesher() :
	fCurrentElements( NULL ),
    fAxialAngle(0.),
    fAxialCount(100)
    {
    }
    KGDiscreteRotationalMesher::~KGDiscreteRotationalMesher()
    {
    }

    void KGDiscreteRotationalMesher::MeshAxialSurface( KGExtendedSurface< KGAxialMesh >* aSurface )
  {
    for (KGAxialMeshElementIt it = aSurface->Elements()->begin();it!=aSurface->Elements()->end();++it)
      AddAxialMeshElement(*it);
  }

    void KGDiscreteRotationalMesher::MeshAxialSpace( KGExtendedSpace< KGAxialMesh >* aSpace )
  {
    for (KGAxialMeshElementIt it = aSpace->Elements()->begin();it!=aSpace->Elements()->end();++it)
      AddAxialMeshElement(*it);
  }

  void KGDiscreteRotationalMesher::AddAxialMeshElement(KGAxialMeshElement* e)
  {
    if (KGAxialMeshLoop* l = dynamic_cast<KGAxialMeshLoop*>(e))
      AddAxialMeshLoop(l);
    else if (KGAxialMeshRing* r = dynamic_cast<KGAxialMeshRing*>(e))
      AddAxialMeshRing(r);
  }

  void KGDiscreteRotationalMesher::AddAxialMeshLoop(KGAxialMeshLoop* l)
  {
    KTransformation transform;
    transform.SetRotationAxisAngle(fAxialAngle,0.,0.);

    if( fabs( (l->GetP0()[1]) - (l->GetP1()[1]) ) > 1.e-10 ) {

      KThreeVector P00( (l->GetP0()[1]), 0., l->GetP0()[0] );
      KThreeVector P01( (l->GetP0()[1])*cos(2.*KConst::Pi()/fAxialCount), (l->GetP0()[1])*sin(2.*KConst::Pi()/fAxialCount), l->GetP0()[0] );

      KThreeVector P10( (l->GetP1()[1]), 0., l->GetP1()[0] );
      KThreeVector P11( (l->GetP1()[1])*cos(2.*KConst::Pi()/fAxialCount), (l->GetP1()[1])*sin(2.*KConst::Pi()/fAxialCount), l->GetP1()[0] );


      KGMeshTriangle singleTriangle1( P00,P01,P11 );
      KGMeshTriangle singleTriangle2( P10,P00,P11 );


      singleTriangle1.Transform(transform);
      singleTriangle2.Transform(transform);
      KGDiscreteRotationalMeshTriangle* t1 = new KGDiscreteRotationalMeshTriangle(singleTriangle1);
      KGDiscreteRotationalMeshTriangle* t2 = new KGDiscreteRotationalMeshTriangle(singleTriangle2);
      t1->NumberOfElements(fAxialCount);
      t2->NumberOfElements(fAxialCount);
      fCurrentElements->push_back(t1);
      fCurrentElements->push_back(t2);
    }
    else
    {
    	KThreeVector P0( (l->GetP0()[1]), 0., l->GetP0()[0] );
    	KThreeVector P1( (l->GetP0()[1])*cos(2.*KConst::Pi()/fAxialCount), (l->GetP0()[1])*sin(2.*KConst::Pi()/fAxialCount), l->GetP0()[0] );

        KGMeshRectangle singleRectangle(fabs(l->GetP1()[0]-l->GetP0()[0]),(P1-P0).Magnitude(),P0,KThreeVector(0.,0.,1.),(P1-P0).Unit() );

		singleRectangle.Transform(transform);
		KGDiscreteRotationalMeshRectangle* r = new KGDiscreteRotationalMeshRectangle(singleRectangle);
		r->NumberOfElements(fAxialCount);
		fCurrentElements->push_back(r);
    }
  }

  void KGDiscreteRotationalMesher::VisitSurface(KGSurface* aSurface)
  {
  	KGExtendedSurface<KGDiscreteRotationalMesh>* discRotMesh = aSurface->AsExtension<KGDiscreteRotationalMesh>();
  	if( !discRotMesh )
  		std::cerr << "KGDiscreteRotationalMesh assumes that extension is already present.\n";

  	fCurrentElements = discRotMesh->Elements();

  	KGExtendedSurface<KGAxialMesh>* axialMesh = aSurface->AsExtension<KGAxialMesh>();
  	if( axialMesh )
  		MeshAxialSurface( axialMesh );
  	else {
  		KGDiscreteRotationalAreaMesher* areaMesher = new KGDiscreteRotationalAreaMesher();
  		areaMesher->SetMeshElementOutput( fCurrentElements );
  		aSurface->AcceptNode( areaMesher );
  		delete areaMesher;
  	}

  }

  void KGDiscreteRotationalMesher::VisitSpace(KGSpace* aSpace)
  {
  	KGExtendedSpace<KGDiscreteRotationalMesh>* discRotMesh = aSpace->AsExtension<KGDiscreteRotationalMesh>();
  	if( !discRotMesh )
  		std::cerr << "KGDiscreteRotationalMesh assumes that extension is already present.\n";

  	fCurrentElements = discRotMesh->Elements();

  	KGExtendedSpace<KGAxialMesh>* axialMesh = aSpace->AsExtension<KGAxialMesh>();
  	if( axialMesh )
  		MeshAxialSpace( axialMesh );
  }


  void KGDiscreteRotationalMesher::AddAxialMeshRing(KGAxialMeshRing* r)
  {
    KTransformation transform;
    transform.SetRotationAxisAngle(fAxialAngle,0.,0.);

    KGMeshWire singleWire(KThreeVector(r->GetP0()[1],0.,r->GetP0()[0]),
			  KThreeVector(r->GetP0()[1]*cos(2.*KConst::Pi()/fAxialCount),
				       r->GetP0()[1]*sin(2.*KConst::Pi()/fAxialCount),
				       r->GetP0()[0]),
			  r->GetD());
    singleWire.Transform(transform);
    KGDiscreteRotationalMeshWire* w = new KGDiscreteRotationalMeshWire(singleWire);
    w->NumberOfElements(fAxialCount);
    fCurrentElements->push_back(w);
  }
}
