#include "KGPortHousingSpace.hh"

#include "KGPortHousingSurface.hh"
#include "KGRectangle.hh"
#include "KGDisk.hh"

namespace KGeoBag
{
  template <>
  void KGWrappedSpace< KGPortHousing >::VolumeInitialize(BoundaryContainer& aBoundaryContainer) const
  {
    fObject->Initialize();

    auto tSurface = std::make_shared<KGPortHousingSurface>(fObject);
    aBoundaryContainer.push_back( tSurface );

    double normal_local[3] = {0.,0.,1.};
    double normal_global[3];
    fObject->GetCoordinateTransform()->ConvertToGlobalCoords(normal_local,normal_global,true);
    KThreeVector n(normal_global[0],normal_global[1],normal_global[2]);
    n = n.Unit();

    {
      auto disk = std::make_shared<KGDisk>();
      disk->SetP0(KThreeVector(fObject->GetAMain(0),
			       fObject->GetAMain(1),
			       fObject->GetAMain(2)));
      disk->SetNormal(-1.*n);
      disk->SetRadius(fObject->GetRMain());
      aBoundaryContainer.push_back(disk);
    }

    {
      auto disk = std::make_shared<KGDisk>();
      disk->SetP0(KThreeVector(fObject->GetBMain(0),
			       fObject->GetBMain(1),
			       fObject->GetBMain(2)));
      disk->SetNormal(n);
      disk->SetRadius(fObject->GetRMain());
      aBoundaryContainer.push_back(disk);
    }

    for (unsigned int i=0;i<fObject->GetNPorts();i++)
    {
      const KGPortHousing::Port* p = fObject->GetPort(i);

      if (const KGPortHousing::RectangularPort* r =
	  dynamic_cast<const KGPortHousing::RectangularPort*>(p))
      {
	double a = r->GetLength();
	double b = r->GetWidth();
	KThreeVector n2 = n;
	KThreeVector n3(r->GetASub(0),r->GetASub(1),0.);
	n3 = n3.Unit();
	KThreeVector n1 = n2.Cross(n3).Unit();
	KThreeVector p0(r->GetASub(0) - .5*a,
			r->GetASub(1) - .5*b,
			r->GetASub(2));

    auto rect = std::make_shared<KGRectangle>(a,b,p0,n1,n2);
	aBoundaryContainer.push_back(rect);
      }
      else if (const KGPortHousing::CircularPort* c =
	       dynamic_cast<const KGPortHousing::CircularPort*>(p))
      {
	auto disk = std::make_shared<KGDisk>();
	disk->SetP0(KThreeVector(c->GetASub(0),
				 c->GetASub(1),
				 c->GetASub(2)));
	disk->SetNormal(KThreeVector(-1.*c->GetNorm(0),
				     -1.*c->GetNorm(1),
				     -1.*c->GetNorm(2)));
	disk->SetRadius(c->GetRSub());
	aBoundaryContainer.push_back(disk);
      }
    }

    return;
  }
}
