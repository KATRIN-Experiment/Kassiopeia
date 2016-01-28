#ifndef KGPORTHOUSINGBUILDER_HH_
#define KGPORTHOUSINGBUILDER_HH_

#include "KComplexElement.hh"

#include "KGWrappedSurface.hh"
#include "KGWrappedSpace.hh"
#include "KGPortHousing.hh"
using namespace KGeoBag;

namespace katrin
{

  typedef KComplexElement< KGPortHousing::RectangularPort > KGPortHousingRectangularPortBuilder;

  template< >
  inline bool KGPortHousingRectangularPortBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "length")
    {      
      anAttribute->CopyTo(fObject, &KGPortHousing::RectangularPort::SetLength );
      return true;
    }
    if (anAttribute->GetName() == "width")
    {
      anAttribute->CopyTo(fObject, &KGPortHousing::RectangularPort::SetWidth );
      return true;
    }
    if (anAttribute->GetName() == "x")
    {
      double a[ 3 ] = {};
      anAttribute->CopyTo(a[ 0 ]);
      a[ 1 ] = fObject->GetASub(1);
      a[ 2 ] = fObject->GetASub(2);
      fObject->SetASub(a);
      return true;
    }
    if (anAttribute->GetName() == "y")
    {
      double a[ 3 ] = {};
      a[ 0 ] = fObject->GetASub(0);
      anAttribute->CopyTo(a[ 1 ]);
      a[ 2 ] = fObject->GetASub(2);
      fObject->SetASub(a);
      return true;
    }
    if (anAttribute->GetName() == "z")
    {
      double a[ 3 ] = {};
      a[ 0 ] = fObject->GetASub(0);
      a[ 1 ] = fObject->GetASub(1);
      anAttribute->CopyTo(a[ 2 ]);
      fObject->SetASub(a);
      return true;
    }
    return false;
  }


  typedef KComplexElement< KGPortHousing::CircularPort > KGPortHousingCircularPortBuilder;

  template< >
  inline bool KGPortHousingCircularPortBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "radius")
    {      
      anAttribute->CopyTo( fObject, &KGPortHousing::CircularPort::SetRSub );
      return true;
    }
    if (anAttribute->GetName() == "x")
    {
      double a[ 3 ] = {};
      anAttribute->CopyTo(a[ 0 ]);
      a[ 1 ] = fObject->GetASub(1);
      a[ 2 ] = fObject->GetASub(2);
      fObject->SetASub(a);
      return true;
    }
    if (anAttribute->GetName() == "y")
    {
      double a[ 3 ] = {};
      a[ 0 ] = fObject->GetASub(0);
      anAttribute->CopyTo(a[ 1 ]);
      a[ 2 ] = fObject->GetASub(2);
      fObject->SetASub(a);
      return true;
    }
    if (anAttribute->GetName() == "z")
    {
      double a[ 3 ] = {};
      a[ 0 ] = fObject->GetASub(0);
      a[ 1 ] = fObject->GetASub(1);
      anAttribute->CopyTo(a[ 2 ]);
      fObject->SetASub(a);
      return true;
    }
    return false;
  }


  typedef KComplexElement< KGPortHousing > KGPortHousingBuilder;

  template< >
  inline bool KGPortHousingBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "radius")
    {
      anAttribute->CopyTo(fObject, &KGPortHousing::SetRMain );
      return true;
    }
    if (anAttribute->GetName() == "x1")
    {
      double a[ 3 ] = {};
      anAttribute->CopyTo(a[ 0 ]);
      a[ 1 ] = fObject->GetAMain(1);
      a[ 2 ] = fObject->GetAMain(2);
      fObject->SetAMain(a);
      return true;
    }
    if (anAttribute->GetName() == "y1")
    {
      double a[ 3 ] = {};
      a[ 0 ] = fObject->GetAMain(0);
      anAttribute->CopyTo(a[ 1 ]);
      a[ 2 ] = fObject->GetAMain(2);
      fObject->SetAMain(a);
      return true;
    }
    if (anAttribute->GetName() == "z1")
    {
      double a[ 3 ] = {};
      a[ 0 ] = fObject->GetAMain(0);
      a[ 1 ] = fObject->GetAMain(1);
      anAttribute->CopyTo(a[ 2 ]);
      fObject->SetAMain(a);
      return true;
    }
    if (anAttribute->GetName() == "x2")
    {
      double a[ 3 ] = {};
      anAttribute->CopyTo(a[ 0 ]);
      a[ 1 ] = fObject->GetBMain(1);
      a[ 2 ] = fObject->GetBMain(2);
      fObject->SetBMain(a);
      return true;
    }
    if (anAttribute->GetName() == "y2")
    {
      double a[ 3 ] = {};
      a[ 0 ] = fObject->GetBMain(0);
      anAttribute->CopyTo(a[ 1 ]);
      a[ 2 ] = fObject->GetBMain(2);
      fObject->SetBMain(a);
      return true;
    }
    if (anAttribute->GetName() == "z2")
    {
      double a[ 3 ] = {};
      a[ 0 ] = fObject->GetBMain(0);
      a[ 1 ] = fObject->GetBMain(1);
      anAttribute->CopyTo(a[ 2 ]);
      fObject->SetBMain(a);
      return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count")
    {
      fObject->SetNumDiscMain( anAttribute->AsReference<int>() );
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
  inline bool KGPortHousingBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "rectangular_port")
    {
      KGPortHousing::RectangularPort* rectangularPort = NULL;
      anElement->ReleaseTo(rectangularPort);
      fObject->AddPort(rectangularPort);
      rectangularPort->Initialize();
      return true;
    }
    if (anElement->GetName() == "circular_port")
    {
      KGPortHousing::CircularPort* circularPort = NULL;
      anElement->ReleaseTo(circularPort);
      fObject->AddPort(circularPort);
      circularPort->Initialize();
      return true;
    }
    return false;
  }

  typedef KComplexElement< KGWrappedSurface< KGPortHousing > > KGPortHousingSurfaceBuilder;

  template<>
  inline bool KGPortHousingSurfaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSurface< KGPortHousing >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGPortHousingSurfaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "port_housing")
    {
      KGPortHousing* object = NULL;
      anElement->ReleaseTo(object);
      KSmartPointer< KGPortHousing > smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }


  typedef KComplexElement< KGWrappedSpace< KGPortHousing > > KGPortHousingSpaceBuilder;

  template<>
  inline bool KGPortHousingSpaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSpace< KGPortHousing >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGPortHousingSpaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "port_housing")
    {
      KGPortHousing* object = NULL;
      anElement->ReleaseTo(object);
      KSmartPointer< KGPortHousing > smartPtr(object);
      fObject->SetObject(smartPtr);
      return true;
    }
    return false;
  }

}

#endif
