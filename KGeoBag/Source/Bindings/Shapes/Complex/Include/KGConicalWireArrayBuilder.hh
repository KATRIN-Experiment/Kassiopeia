#ifndef KGCONICALWIREARRAYBUILDER_HH_
#define KGCONICALWIREARRAYBUILDER_HH_

#include "KComplexElement.hh"

#include "KGWrappedSurface.hh"
#include "KGWrappedSpace.hh"
#include "KGConicalWireArray.hh"

using namespace KGeoBag;

namespace katrin
{
  typedef KComplexElement<KGConicalWireArray> KGConicalWireArrayBuilder;

  template<>
  inline bool KGConicalWireArrayBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "radius1")
    {
      double radius1;
      anAttribute->CopyTo(radius1);
      fObject->SetR1(radius1);
      return true;
    }
    if (anAttribute->GetName() == "radius2")
    {
      double radius2;
      anAttribute->CopyTo(radius2);
      fObject->SetR2(radius2);
      return true;
    }
    if (anAttribute->GetName() == "z1")
    {
      double z1;
      anAttribute->CopyTo(z1);
      fObject->SetZ1(z1);
      return true;
    }
    if (anAttribute->GetName() == "z2")
    {
      double z2;
      anAttribute->CopyTo(z2);
      fObject->SetZ2(z2);
      return true;
    }
    if (anAttribute->GetName() == "wire_count")
    {
      int i;
      anAttribute->CopyTo(i);
      fObject->SetNWires(i);
      return true;
    }
    if (anAttribute->GetName() == "theta_start")
    {
      double thetaStart;
      anAttribute->CopyTo(thetaStart);
      fObject->SetThetaStart(thetaStart);
      return true;
    }
    if (anAttribute->GetName() == "diameter")
    {
      double diameter;
      anAttribute->CopyTo(diameter);
      fObject->SetDiameter(diameter);
      return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count")
    {
      int nDiscLong;
      anAttribute->CopyTo(nDiscLong);
      fObject->SetNDisc(nDiscLong);
      return true;
    }
    return false;
  }

  typedef KComplexElement<KGWrappedSurface<KGConicalWireArray> > KGConicalWireArraySurfaceBuilder;

  template<>
  inline bool KGConicalWireArraySurfaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSurface< KGConicalWireArray >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGConicalWireArraySurfaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "conical_wire_array")
    {
      KGConicalWireArray* object;
      anElement->ReleaseTo(object);
      object->Initialize();
      KSmartPointer< KGConicalWireArray > smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }


  typedef KComplexElement<KGWrappedSpace<KGConicalWireArray> > KGConicalWireArraySpaceBuilder;

  template<>
  inline bool KGConicalWireArraySpaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSpace< KGConicalWireArray >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGConicalWireArraySpaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "conical_wire_array")
    {
      KGConicalWireArray* object;
      anElement->ReleaseTo(object);
      object->Initialize();
      KSmartPointer< KGConicalWireArray > smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }

}

#endif
