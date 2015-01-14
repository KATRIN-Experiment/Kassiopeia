#ifndef KGEXTRUDEDOBJECTBUILDER_HH_
#define KGEXTRUDEDOBJECTBUILDER_HH_

#include "KComplexElement.hh"

#include "KGWrappedSurface.hh"
#include "KGWrappedSpace.hh"
#include "KGExtrudedObject.hh"

using namespace KGeoBag;

namespace katrin
{
  typedef KComplexElement<KGExtrudedObject::Line>  KGExtrudedObjectLineBuilder;

  template<>
  inline bool KGExtrudedObjectLineBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "x1")
    {
      double p1[2];
      anAttribute->CopyTo(p1[0]);
      p1[1] = fObject->GetP1(1);
      fObject->SetP1(p1);
      return true;
    }
    if (anAttribute->GetName() == "y1")
    {
      double p1[2];
      p1[0] = fObject->GetP1(0);
      anAttribute->CopyTo(p1[1]);
      fObject->SetP1(p1);
      return true;
    }
    if (anAttribute->GetName() == "x2")
    {
      double p2[2];
      anAttribute->CopyTo(p2[0]);
      p2[1] = fObject->GetP2(1);
      fObject->SetP2(p2);
      return true;
    }
    if (anAttribute->GetName() == "y2")
    {
      double p2[2];
      p2[0] = fObject->GetP2(0);
      anAttribute->CopyTo(p2[1]);
      fObject->SetP2(p2);
      return true;
    }
    return false;
  }

  typedef KComplexElement<KGExtrudedObject::Arc>  KGExtrudedObjectArcBuilder;

  template<>
  inline bool KGExtrudedObjectArcBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "x1")
    {
      double p1[2];
      anAttribute->CopyTo(p1[0]);
      p1[1] = fObject->GetP1(1);
      fObject->SetP1(p1);
      return true;
    }
    if (anAttribute->GetName() == "y1")
    {
      double p1[2];
      p1[0] = fObject->GetP1(0);
      anAttribute->CopyTo(p1[1]);
      fObject->SetP1(p1);
      return true;
    }
    if (anAttribute->GetName() == "x2")
    {
      double p2[2];
      anAttribute->CopyTo(p2[0]);
      p2[1] = fObject->GetP2(1);
      fObject->SetP2(p2);
      return true;
    }
    if (anAttribute->GetName() == "y2")
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
      fObject->IsPositivelyOriented(positiveOrientation);
      return true;
    }
    return false;
  }

  typedef KComplexElement<KGExtrudedObject> KGExtrudedObjectBuilder;

  template<>
  inline bool KGExtrudedObjectBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "z_min")
    {
      double zMin;
      anAttribute->CopyTo(zMin);
      fObject->SetZMin(zMin);
      return true;
    }
    if (anAttribute->GetName() == "z_max")
    {
      double zMax;
      anAttribute->CopyTo(zMax);
      fObject->SetZMax(zMax);
      return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count")
    {
      int nDisc;
      anAttribute->CopyTo(nDisc);
      fObject->SetNDisc(nDisc);
      return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power")
    {
      double discretizationPower;
      anAttribute->CopyTo(discretizationPower);
      fObject->SetDiscretizationPower(discretizationPower);
      return true;
    }
    if (anAttribute->GetName() == "closed_form")
    {
      bool closedLoops;
      anAttribute->CopyTo(closedLoops);
      if (closedLoops)
	fObject->Close();
      else
	fObject->Open();
      return true;
    }
    return false;
  }

  template<>
  inline bool KGExtrudedObjectBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "outer_line")
    {
      KGExtrudedObject::Line* line;
      anElement->ReleaseTo(line);
      line->Initialize();
      fObject->AddOuterSegment(line);
      return true;
    }
    if (anElement->GetName() == "inner_line")
    {
      KGExtrudedObject::Line* line;
      anElement->ReleaseTo(line);
      line->Initialize();
      fObject->AddInnerSegment(line);
      return true;
    }
    if (anElement->GetName() == "outer_arc")
    {
      KGExtrudedObject::Arc* arc;
      anElement->ReleaseTo(arc);
      arc->Initialize();
      fObject->AddOuterSegment(arc);
      return true;
    }
    if (anElement->GetName() == "inner_arc")
    {
      KGExtrudedObject::Arc* arc;
      anElement->ReleaseTo(arc);
      arc->Initialize();
      fObject->AddInnerSegment(arc);
      return true;
    }
    return false;
  }

  typedef KComplexElement<KGWrappedSurface<KGExtrudedObject> > KGExtrudedSurfaceBuilder;

  template<>
  inline bool KGExtrudedSurfaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSurface< KGExtrudedObject >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGExtrudedSurfaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "extruded_object")
    {
      KGExtrudedObject* object;
      anElement->ReleaseTo(object);
      object->Initialize();
      KSmartPointer<KGExtrudedObject> smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }


  typedef KComplexElement<KGWrappedSpace<KGExtrudedObject> > KGExtrudedSpaceBuilder;

  template<>
  inline bool KGExtrudedSpaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSpace<KGExtrudedObject>::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGExtrudedSpaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "extruded_object")
    {
      KGExtrudedObject* object;
      anElement->ReleaseTo(object);
      object->Initialize();
      KSmartPointer<KGExtrudedObject> smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }

}

#endif
