#ifndef KGBEAMBUILDER_HH_
#define KGBEAMBUILDER_HH_

#include "KComplexElement.hh"

#include "KGWrappedSurface.hh"
#include "KGWrappedSpace.hh"
#include "KGBeam.hh"

using namespace KGeoBag;

namespace katrin
{
  struct KGBeamLine
  {
    double x1;
    double y1;
    double z1;
    double x2;
    double y2;
    double z2;
  };

  typedef KComplexElement<KGBeamLine>  KGBeamLineBuilder;

  template< >
  inline bool KGBeamLineBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "x1")
    {
      anAttribute->CopyTo(fObject->x1);
      return true;
    }
    if (anAttribute->GetName() == "y1")
    {
      anAttribute->CopyTo(fObject->y1);
      return true;
    }
    if (anAttribute->GetName() == "z1")
    {
      anAttribute->CopyTo(fObject->z1);
      return true;
    }
    if (anAttribute->GetName() == "x2")
    {
      anAttribute->CopyTo(fObject->x2);
      return true;
    }
    if (anAttribute->GetName() == "y2")
    {
      anAttribute->CopyTo(fObject->y2);
      return true;
    }
    if (anAttribute->GetName() == "z2")
    {
      anAttribute->CopyTo(fObject->z2);
      return true;
    }
    return false;
  }

  typedef KComplexElement<KGBeam> KGBeamBuilder;

  template<>
  inline bool KGBeamBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "longitudinal_mesh_count")
    {
      anAttribute->CopyTo(fObject, &KGBeam::SetNDiscLong);
      return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count")
    {      
      anAttribute->CopyTo(fObject, &KGBeam::SetNDiscRad);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGBeamBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "start_line")
    {
      KGBeamLine* startLine = anElement->AsPointer<KGBeamLine>();
      double p1[3] = {startLine->x1,startLine->y1,startLine->z1};
      double p2[3] = {startLine->x2,startLine->y2,startLine->z2};
      fObject->AddStartLine(p1,p2);
      return true;
    }
    if (anElement->GetName() == "end_line")
    {
      KGBeamLine* endLine = anElement->AsPointer<KGBeamLine>();
      double p1[3] = {endLine->x1,endLine->y1,endLine->z1};
      double p2[3] = {endLine->x2,endLine->y2,endLine->z2};
      fObject->AddEndLine(p1,p2);
      return true;
    }
    return false;
  }

  typedef KComplexElement<KGWrappedSurface<KGBeam> > KGBeamSurfaceBuilder;

  template<>
  inline bool KGBeamSurfaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSurface< KGBeam >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGBeamSurfaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "beam")
    {
        KGBeam* object = NULL;
        anElement->ReleaseTo(object);
        object->Initialize();
        KSmartPointer< KGBeam > smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
  }


  typedef KComplexElement<KGWrappedSpace<KGBeam> > KGBeamSpaceBuilder;

  template<>
  inline bool KGBeamSpaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSpace< KGBeam >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGBeamSpaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "beam")
    {
        KGBeam* object = NULL;
        anElement->ReleaseTo(object);
        object->Initialize();
        KSmartPointer< KGBeam > smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
  }

}

#endif
