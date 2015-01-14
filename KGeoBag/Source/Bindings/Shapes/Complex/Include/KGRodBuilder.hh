#ifndef KGRODBUILDER_HH_
#define KGRODBUILDER_HH_

#include "KComplexElement.hh"

#include "KGWrappedSurface.hh"
#include "KGWrappedSpace.hh"
#include "KGRod.hh"

using namespace KGeoBag;

namespace katrin
{
  struct KGRodVertex
  {
    double x;
    double y;
    double z;
  };

  typedef KComplexElement<KGRodVertex>  KGRodVertexBuilder;

  template< >
  inline bool KGRodVertexBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "x")
    {
      anAttribute->CopyTo(fObject->x);
      return true;
    }
    if (anAttribute->GetName() == "y")
    {
      anAttribute->CopyTo(fObject->y);
      return true;
    }
    if (anAttribute->GetName() == "z")
    {
      anAttribute->CopyTo(fObject->z);
      return true;
    }
    return false;
  }

  typedef KComplexElement<KGRod> KGRodBuilder;

  template<>
  inline bool KGRodBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "radius")
    {
      double radius;
      anAttribute->CopyTo(radius);
      fObject->SetRadius(radius);
      return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count")
    {
      int nDiscLong;
      anAttribute->CopyTo(nDiscLong);
      fObject->SetNDiscLong(nDiscLong);
      return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count")
    {
      int nDiscRad;
      anAttribute->CopyTo(nDiscRad);
      fObject->SetNDiscRad(nDiscRad);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGRodBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "vertex")
    {
      KGRodVertex* vtx = anElement->AsPointer<KGRodVertex>();
      double p[3] = {vtx->x,vtx->y,vtx->z};
      fObject->AddPoint(p);
      return true;
    }
    return false;
  }

  typedef KComplexElement<KGWrappedSurface<KGRod> > KGRodSurfaceBuilder;

  template<>
  inline bool KGRodSurfaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSurface< KGRod >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGRodSurfaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "rod")
    {
      KGRod* object;
      anElement->ReleaseTo(object);
      object->Initialize();
      KSmartPointer< KGRod > smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }


  typedef KComplexElement<KGWrappedSpace<KGRod> > KGRodSpaceBuilder;

  template<>
  inline bool KGRodSpaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSpace< KGRod >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGRodSpaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "rod")
    {
      KGRod* object;
      anElement->ReleaseTo(object);
      object->Initialize();
      KSmartPointer< KGRod > smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }

}

#endif
