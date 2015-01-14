#include "KGConicSectPortHousingBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

  static const int sKGConicSectPortHousingOrthogonalPortBuilderStructure =
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<int>("box_radial_mesh_count") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<int>("box_curve_mesh_count") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<int>("cylinder_longitudinal_mesh_count") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<int>("cylinder_axial_mesh_count") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<double>("radius") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<double>("x") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<double>("y") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<double>("z");

  static const int sKGConicSectPortHousingParaxialPortBuilderStructure =
    KGConicSectPortHousingParaxialPortBuilder::Attribute<int>("box_radial_mesh_count") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<int>("box_curve_mesh_count") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<int>("cylinder_longitudinal_mesh_count") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<int>("cylinder_axial_mesh_count") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<double>("radius") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<double>("x") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<double>("y") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<double>("z");

  static const int sKGConicSectPortHousingBuilderStructure =
    KGConicSectPortHousingBuilder::Attribute<double>("r1") +
    KGConicSectPortHousingBuilder::Attribute<double>("z1") +
    KGConicSectPortHousingBuilder::Attribute<double>("r2") +
    KGConicSectPortHousingBuilder::Attribute<double>("z2") +
    KGConicSectPortHousingBuilder::Attribute<int>("axial_mesh_count") +
    KGConicSectPortHousingBuilder::Attribute<int>("longitudinal_mesh_count") +
    KGConicSectPortHousingBuilder::ComplexElement< KGConicSectPortHousing::OrthogonalPort >("orthogonal_port") +
    KGConicSectPortHousingBuilder::ComplexElement< KGConicSectPortHousing::ParaxialPort >("paraxial_port");

  static const int sKGConicSectPortHousingSurfaceBuilderStructure =
    KGConicSectPortHousingSurfaceBuilder::Attribute<string>("name") +
    KGConicSectPortHousingSurfaceBuilder::ComplexElement<KGConicSectPortHousing>("conic_section_port_housing");

  static const int sKGConicSectPortHousingSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGConicSectPortHousing> >("conic_section_port_housing_surface");

  static const int sKGConicSectPortHousingSpaceBuilderStructure =
    KGConicSectPortHousingSpaceBuilder::Attribute<string>("name") +
    KGConicSectPortHousingSpaceBuilder::ComplexElement<KGConicSectPortHousing>("conic_section_port_housing");

  static const int sKGConicSectPortHousingSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGConicSectPortHousing> >("conic_section_port_housing_space");

}
