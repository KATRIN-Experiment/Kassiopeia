#ifndef KGCONICSECTPORTHOUSINGBUILDER_HH_
#define KGCONICSECTPORTHOUSINGBUILDER_HH_

#include "KComplexElement.hh"

#include "KGWrappedSurface.hh"
#include "KGWrappedSpace.hh"
#include "KGConicSectPortHousing.hh"

using namespace KGeoBag;

namespace katrin
{

  typedef KComplexElement< KGConicSectPortHousing::OrthogonalPort > KGConicSectPortHousingOrthogonalPortBuilder;

  template< >
  inline bool KGConicSectPortHousingOrthogonalPortBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "box_radial_mesh_count")
    {      
      fObject->SetXDisc( anAttribute->AsReference<int>() );
      return true;
    }
    if (anAttribute->GetName() == "box_curve_mesh_count")
    {
      fObject->SetAlphaPolySub( anAttribute->AsReference<int>() );
      return true;
    }
    if (anAttribute->GetName() == "cylinder_longitudinal_mesh_count")
    {
      fObject->SetCylDisc( anAttribute->AsReference<int>() );
      return true;
    }
    if (anAttribute->GetName() == "cylinder_axial_mesh_count")
    {
      fObject->SetPolySub( anAttribute->AsReference<int>() );
      return true;
    }
    if (anAttribute->GetName() == "radius")
    {
      fObject->SetRSub( anAttribute->AsReference<double>() );
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

  typedef KComplexElement< KGConicSectPortHousing::ParaxialPort > KGConicSectPortHousingParaxialPortBuilder;

  template< >
  inline bool KGConicSectPortHousingParaxialPortBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "box_radial_mesh_count")
    {
      fObject->SetXDisc( anAttribute->AsReference<int>() );
      return true;
    }
    if (anAttribute->GetName() == "box_curve_mesh_count")
    {
      fObject->SetAlphaPolySub( anAttribute->AsReference<int>() );
      return true;
    }
    if (anAttribute->GetName() == "cylinder_longitudinal_mesh_count")
    {
      fObject->SetCylDisc( anAttribute->AsReference<int>() );
      return true;
    }
    if (anAttribute->GetName() == "cylinder_axial_mesh_count")
    {
      fObject->SetPolySub( anAttribute->AsReference<int>() );
      return true;
    }
    if (anAttribute->GetName() == "radius")
    {
      fObject->SetRSub( anAttribute->AsReference<double>() );
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

  typedef KComplexElement< KGConicSectPortHousing > KGConicSectPortHousingBuilder;

  template< >
  inline bool KGConicSectPortHousingBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "r1")
    {
      fObject->SetRAMain( anAttribute->AsReference<double>() );
      return true;
    }
    if (anAttribute->GetName() == "z1")
    {
      fObject->SetZAMain( anAttribute->AsReference<double>() );
      return true;
    }
    if (anAttribute->GetName() == "r2")
    {
      fObject->SetRBMain( anAttribute->AsReference<double>() );
      return true;
    }
    if (anAttribute->GetName() == "z2")
    {
      fObject->SetZBMain( anAttribute->AsReference<double>() );
      return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count")
    {
      fObject->SetPolyMain( anAttribute->AsReference<int>() );
      return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count")
    {
      fObject->SetNumDiscMain( anAttribute->AsReference<int>() );
      return true;
    }
    return false;
  }

  template< >
  inline bool KGConicSectPortHousingBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "orthogonal_port")
    {
        KGConicSectPortHousing::OrthogonalPort* orthogonalPort = NULL;
        anElement->ReleaseTo(orthogonalPort);
        fObject->AddPort(orthogonalPort);
        orthogonalPort->Initialize();
        return true;
    }
    if (anElement->GetName() == "paraxial_port")
    {
        KGConicSectPortHousing::ParaxialPort* paraxialPort = NULL;
        anElement->ReleaseTo(paraxialPort);
        fObject->AddPort(paraxialPort);
        paraxialPort->Initialize();
        return true;
    }
    return false;
  }

  typedef KComplexElement< KGWrappedSurface< KGConicSectPortHousing > > KGConicSectPortHousingSurfaceBuilder;

  template<>
  inline bool KGConicSectPortHousingSurfaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSurface< KGConicSectPortHousing >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGConicSectPortHousingSurfaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "conic_section_port_housing")
    {
        KGConicSectPortHousing* object = NULL;
        anElement->ReleaseTo(object);
        std::shared_ptr< KGConicSectPortHousing > smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
  }


  typedef KComplexElement< KGWrappedSpace< KGConicSectPortHousing > > KGConicSectPortHousingSpaceBuilder;

  template<>
  inline bool KGConicSectPortHousingSpaceBuilder::AddAttribute(KContainer* anAttribute)
  {
    if (anAttribute->GetName() == "name")
    {
      anAttribute->CopyTo(fObject, &KGWrappedSpace< KGConicSectPortHousing >::SetName);
      return true;
    }
    return false;
  }

  template<>
  inline bool KGConicSectPortHousingSpaceBuilder::AddElement(KContainer* anElement)
  {
    if (anElement->GetName() == "conic_section_port_housing")
    {
        KGConicSectPortHousing* object = NULL;
        anElement->ReleaseTo(object);
        std::shared_ptr< KGConicSectPortHousing > smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
  }

}

#endif
