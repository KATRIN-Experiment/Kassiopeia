#ifndef KGCOMPLEXANNULUSBUILDER_HH_
#define KGCOMPLEXANNULUSBUILDER_HH_

#include "KComplexElement.hh"

#include "KGWrappedSurface.hh"
#include "KGWrappedSpace.hh"
#include "KGComplexAnnulus.hh"
using namespace KGeoBag;

namespace katrin
{

  typedef KComplexElement< KGComplexAnnulus::Ring > KGComplexAnnulusRingBuilder;

  template< >
  inline bool KGComplexAnnulusRingBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "radius")
    {
      anAttribute->CopyTo( fObject, &KGComplexAnnulus::Ring::SetRSub );
      return true;
    }
    if (anAttribute->GetName() == "x")
    {
      double a[ 2 ] = {};
      anAttribute->CopyTo(a[ 0 ]);
      a[ 1 ] = fObject->GetASub(1);
      fObject->SetASub(a);
      return true;
    }
    if (anAttribute->GetName() == "y")
    {
      double a[ 2 ] = {};
      a[ 0 ] = fObject->GetASub(0);
      anAttribute->CopyTo(a[ 1 ]);
      fObject->SetASub(a);
      return true;
    }
    return false;
  }


  typedef KComplexElement< KGComplexAnnulus > KGComplexAnnulusBuilder;

  template< >
  inline bool KGComplexAnnulusBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "radius")
    {
      anAttribute->CopyTo(fObject, &KGComplexAnnulus::SetRMain );
      return true;
    }
    if (anAttribute->GetName() == "radial_mesh_count")
    {
      fObject->SetRadialMeshMain( anAttribute->AsReference<int>() );
      return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count")
    {
      fObject->SetPolyMain( anAttribute->AsReference<int>() );
      return true;
    }
    return false;
  }

  template< >
  inline bool KGComplexAnnulusBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "ring")
    {
      KGComplexAnnulus::Ring* Ring = NULL;
      anElement->ReleaseTo(Ring);
      fObject->AddRing(Ring);
      Ring->Initialize();
      return true;
    }

    return false;
  }

  typedef KComplexElement< KGWrappedSurface< KGComplexAnnulus > > KGComplexAnnulusSurfaceBuilder;

  template<>
  inline bool KGComplexAnnulusSurfaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSurface< KGComplexAnnulus >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGComplexAnnulusSurfaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "complex_annulus")
    {
      KGComplexAnnulus* object = NULL;
      anElement->ReleaseTo(object);
      KSmartPointer< KGComplexAnnulus > smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }

}

#endif
