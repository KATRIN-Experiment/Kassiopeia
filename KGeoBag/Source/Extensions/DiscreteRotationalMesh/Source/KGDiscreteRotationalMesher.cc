#include "KGDiscreteRotationalMesher.hh"

#include "KGAxialMeshLoop.hh"
#include "KGAxialMeshRing.hh"

namespace KGeoBag
{

  KGDiscreteRotationalMesher::KGDiscreteRotationalMesher() :
    fAxialAngle(0.),
    fAxialCount(8)
    {
    }
    KGDiscreteRotationalMesher::~KGDiscreteRotationalMesher()
    {
    }

  void KGDiscreteRotationalMesher::VisitExtendedSurface( KGExtendedSurface< KGAxialMesh >* aSurface )
  {
    for (KGAxialMeshElementIt it = aSurface->Elements()->begin();it!=aSurface->Elements()->end();++it)
      AddAxialMeshElement(*it);
  }

  void KGDiscreteRotationalMesher::VisitExtendedSpace( KGExtendedSpace< KGAxialMesh >* aSpace )
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

    if (fabs(l->GetP0()[1]-l->GetP1()[1])>1.e-10)
    {
      double tmp = 1.;
      if (l->GetP1()[0] < l->GetP0()[0])
	tmp = -1.;

      KGMeshTriangle singleTriangle1((l->GetP1()-l->GetP0()).Magnitude(),
				     2.*KConst::Pi()*l->GetP0()[1]/fAxialCount,
				     KThreeVector(l->GetP0()[1],0.,l->GetP0()[0]),
				     KThreeVector(l->GetP1()[1]-l->GetP0()[1],0.,tmp).Unit(),
				     KThreeVector(0.,1.,0.));

      KGMeshTriangle singleTriangle2((l->GetP1()-l->GetP0()).Magnitude(),
				     2.*KConst::Pi()*l->GetP1()[1]/fAxialCount,
				     KThreeVector(l->GetP1()[1]*cos(2.*KConst::Pi()/fAxialCount),
						  l->GetP1()[1]*sin(2.*KConst::Pi()/fAxialCount),
						  l->GetP1()[0]),
				     KThreeVector(l->GetP0()[1]-l->GetP1()[1],0.,tmp).Unit(),
				     KThreeVector(0.,-1.,0.));

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
      double tmp = 1.;
      if (l->GetP1()[0] < l->GetP0()[0])
	tmp = -1.;

      KGMeshRectangle singleRectangle(fabs(l->GetP1()[0] - l->GetP0()[0]),
				      2.*KConst::Pi()*l->GetP0()[1]/fAxialCount,
				      KThreeVector(l->GetP0()[1],0.,l->GetP0()[0]),
				      KThreeVector(0.,0.,tmp),
				      KThreeVector(0.,1.,0.));
      singleRectangle.Transform(transform);
      KGDiscreteRotationalMeshRectangle* r = new KGDiscreteRotationalMeshRectangle(singleRectangle);
      r->NumberOfElements(fAxialCount);
      fCurrentElements->push_back(r);
    }
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
