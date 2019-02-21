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
      anAttribute->CopyTo(fObject, &KGConicalWireArray::SetR1);
      return true;
    }
    if (anAttribute->GetName() == "radius2")
    {
      anAttribute->CopyTo(fObject, &KGConicalWireArray::SetR2);
      return true;
    }
    if (anAttribute->GetName() == "z1")
    {
      anAttribute->CopyTo(fObject, &KGConicalWireArray::SetZ1);
      return true;
    }
    if (anAttribute->GetName() == "z2")
    {
      anAttribute->CopyTo(fObject, &KGConicalWireArray::SetZ2);
      return true;
    }
    if (anAttribute->GetName() == "wire_count")
    {
      anAttribute->CopyTo(fObject, &KGConicalWireArray::SetNWires);
      return true;
    }
    if (anAttribute->GetName() == "theta_start")
    {
      anAttribute->CopyTo(fObject, &KGConicalWireArray::SetThetaStart);
	  return true;
    }
    if (anAttribute->GetName() == "diameter")
    {
      anAttribute->CopyTo(fObject, &KGConicalWireArray::SetDiameter);
      return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count")
    {
      anAttribute->CopyTo(fObject, &KGConicalWireArray::SetNDisc);
      return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power")
    {
      anAttribute->CopyTo(fObject, &KGConicalWireArray::SetNDiscPower);
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
      KGConicalWireArray* object = NULL;
      anElement->ReleaseTo(object);
      object->Initialize();
      std::shared_ptr< KGConicalWireArray > smartPtr(object);
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
      KGConicalWireArray* object = NULL;
      anElement->ReleaseTo(object);
      object->Initialize();
      std::shared_ptr< KGConicalWireArray > smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }

}

#endif
