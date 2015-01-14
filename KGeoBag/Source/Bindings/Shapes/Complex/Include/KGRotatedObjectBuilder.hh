#ifndef KGROTATEDOBJECTBUILDER_HH_
#define KGROTATEDOBJECTBUILDER_HH_

#include "KComplexElement.hh"

#include "KGWrappedSurface.hh"
#include "KGWrappedSpace.hh"
#include "KGRotatedObject.hh"

using namespace KGeoBag;

namespace katrin
{
  typedef KComplexElement<KGRotatedObject::Line>  KGRotatedObjectLineBuilder;

  template<>
  inline bool KGRotatedObjectLineBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "z1")
    {
      double p1[2];
      anAttribute->CopyTo(p1[0]);
      p1[1] = fObject->GetP1(1);
      fObject->SetP1(p1);
      return true;
    }
    if (anAttribute->GetName() == "r1")
    {
      double p1[2];
      p1[0] = fObject->GetP1(0);
      anAttribute->CopyTo(p1[1]);
      fObject->SetP1(p1);
      return true;
    }
    if (anAttribute->GetName() == "z2")
    {
      double p2[2];
      anAttribute->CopyTo(p2[0]);
      p2[1] = fObject->GetP2(1);
      fObject->SetP2(p2);
      return true;
    }
    if (anAttribute->GetName() == "r2")
    {
      double p2[2];
      p2[0] = fObject->GetP2(0);
      anAttribute->CopyTo(p2[1]);
      fObject->SetP2(p2);
      return true;
    }
    return false;
  }

  typedef KComplexElement<KGRotatedObject::Arc>  KGRotatedObjectArcBuilder;

  template<>
  inline bool KGRotatedObjectArcBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "z1")
    {
      double p1[2];
      anAttribute->CopyTo(p1[0]);
      p1[1] = fObject->GetP1(1);
      fObject->SetP1(p1);
      return true;
    }
    if (anAttribute->GetName() == "r1")
    {
      double p1[2];
      p1[0] = fObject->GetP1(0);
      anAttribute->CopyTo(p1[1]);
      fObject->SetP1(p1);
      return true;
    }
    if (anAttribute->GetName() == "z2")
    {
      double p2[2];
      anAttribute->CopyTo(p2[0]);
      p2[1] = fObject->GetP2(1);
      fObject->SetP2(p2);
      return true;
    }
    if (anAttribute->GetName() == "r2")
    {
      double p2[2];
      p2[0] = fObject->GetP2(0);
      anAttribute->CopyTo(p2[1]);
      fObject->SetP2(p2);
      return true;
    }
    if (anAttribute->GetName() == "radius")
    {
      double radius;
      anAttribute->CopyTo(radius);
      fObject->SetRadius(radius);
      return true;
    }
    if (anAttribute->GetName() == "positive_orientation")
    {
      bool positiveOrientation;
      anAttribute->CopyTo(positiveOrientation);
      fObject->SetOrientation(positiveOrientation);
      return true;
    }
    return false;
  }

  typedef KComplexElement<KGRotatedObject> KGRotatedObjectBuilder;

  template<>
  inline bool KGRotatedObjectBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "longitudinal_mesh_count_start")
    {
      int nPolyBegin;
      anAttribute->CopyTo(nPolyBegin);
      fObject->SetNPolyBegin(nPolyBegin);
      return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count_end")
    {
      int nPolyEnd;
      anAttribute->CopyTo(nPolyEnd);
      fObject->SetNPolyEnd(nPolyEnd);
      return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count")
    {
      int nPoly;
      anAttribute->CopyTo(nPoly);
      fObject->SetNPolyBegin(nPoly);
      fObject->SetNPolyEnd(nPoly);
      return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power")
    {
      double discretizationPower;
      anAttribute->CopyTo(discretizationPower);
      fObject->SetDiscretizationPower(discretizationPower);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGRotatedObjectBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "line")
    {
      KGRotatedObject::Line* line;
      anElement->ReleaseTo(line);
      line->Initialize();
      fObject->AddSegment(line);
      return true;
    }
    if (anElement->GetName() == "arc")
    {
      KGRotatedObject::Arc* arc;
      anElement->ReleaseTo(arc);
      arc->Initialize();
      fObject->AddSegment(arc);
      return true;
    }
    return false;
  }

  typedef KComplexElement<KGWrappedSurface<KGRotatedObject> > KGRotatedSurfaceBuilder;

  template<>
  inline bool KGRotatedSurfaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSurface< KGRotatedObject >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGRotatedSurfaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "rotated_object")
    {
      KGRotatedObject* object;
      anElement->ReleaseTo(object);
      object->Initialize();
      KSmartPointer<KGRotatedObject> smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }


  typedef KComplexElement<KGWrappedSpace<KGRotatedObject> > KGRotatedSpaceBuilder;

  template<>
  inline bool KGRotatedSpaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSpace<KGRotatedObject>::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGRotatedSpaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "rotated_object")
    {
      KGRotatedObject* object;
      anElement->ReleaseTo(object);
      object->Initialize();
      KSmartPointer<KGRotatedObject> smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }

}

#endif
