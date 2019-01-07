#ifndef KGCIRCLEWIREBUILDER_HH_
#define KGCIRCLEWIREBUILDER_HH_

#include "KComplexElement.hh"

#include "KGWrappedSurface.hh"
#include "KGWrappedSpace.hh"
#include "KGCircleWire.hh"

using namespace KGeoBag;

namespace katrin
{
  typedef KComplexElement<KGCircleWire> KGCircleWireBuilder;

  template<>
  inline bool KGCircleWireBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "radius")
    {
      anAttribute->CopyTo(fObject, &KGCircleWire::SetR);
      return true;
    }

    if (anAttribute->GetName() == "diameter")
    {
      anAttribute->CopyTo(fObject, &KGCircleWire::SetDiameter);
      return true;
    }
    if (anAttribute->GetName() == "mesh_count")
    {
      anAttribute->CopyTo(fObject, &KGCircleWire::SetNDisc);
      return true;
    }

    return false;
  }

  typedef KComplexElement<KGWrappedSurface<KGCircleWire> > KGCircleWireSurfaceBuilder;

  template<>
  inline bool KGCircleWireSurfaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSurface< KGCircleWire >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGCircleWireSurfaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "circle_wire")
    {
      KGCircleWire* object = NULL;
      anElement->ReleaseTo(object);
      object->Initialize();
      std::shared_ptr< KGCircleWire > smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }


  typedef KComplexElement<KGWrappedSpace<KGCircleWire> > KGCircleWireSpaceBuilder;

  template<>
  inline bool KGCircleWireSpaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSpace< KGCircleWire >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGCircleWireSpaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "circle_wire")
    {
      KGCircleWire* object = NULL;
      anElement->ReleaseTo(object);
      object->Initialize();
      std::shared_ptr< KGCircleWire > smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }

}

#endif
