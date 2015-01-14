#include "KGRotatedSpace.hh"

#include "KGRotatedSurface.hh"
#include "KGDisk.hh"

namespace KGeoBag
{
  template <>
  void KGWrappedSpace< KGRotatedObject >::VolumeInitialize( BoundaryContainer& aBoundaryContainer ) const
  {
    fObject->Initialize();

    KGRotatedSurface* tSurface = new KGRotatedSurface(fObject);
    aBoundaryContainer.push_back( tSurface );

    if (fObject->GetStartPoint(1) > 1.e-8)
    {
      KGDisk* disk = new KGDisk();
      disk->SetP0(KThreeVector(0.,0.,fObject->GetStartPoint(0)));
      disk->SetNormal(KThreeVector(0.,0.,-1.));
      disk->SetRadius(fObject->GetStartPoint(1));
      disk->SetName("top");
      aBoundaryContainer.push_back(disk);
    }

    if (fObject->GetEndPoint(1) > 1.e-8)
    {
      KGDisk* disk = new KGDisk();
      disk->SetP0(KThreeVector(0.,0.,fObject->GetEndPoint(0)));
      disk->SetNormal(KThreeVector(0.,0.,1.));
      disk->SetRadius(fObject->GetEndPoint(1));
      disk->SetName("bottom");
      aBoundaryContainer.push_back(disk);
    }

    return;
  }
}
