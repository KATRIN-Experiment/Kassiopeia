#include "KGConicSectPortHousingBuilder.hh"
#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

  STATICINT sKGConicSectPortHousingOrthogonalPortBuilderStructure =
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<int>("box_radial_mesh_count") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<int>("box_curve_mesh_count") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<int>("cylinder_longitudinal_mesh_count") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<int>("cylinder_axial_mesh_count") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<double>("radius") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<double>("x") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<double>("y") +
    KGConicSectPortHousingOrthogonalPortBuilder::Attribute<double>("z");

  STATICINT sKGConicSectPortHousingParaxialPortBuilderStructure =
    KGConicSectPortHousingParaxialPortBuilder::Attribute<int>("box_radial_mesh_count") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<int>("box_curve_mesh_count") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<int>("cylinder_longitudinal_mesh_count") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<int>("cylinder_axial_mesh_count") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<double>("radius") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<double>("x") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<double>("y") +
    KGConicSectPortHousingParaxialPortBuilder::Attribute<double>("z");

  STATICINT sKGConicSectPortHousingBuilderStructure =
    KGConicSectPortHousingBuilder::Attribute<double>("r1") +
    KGConicSectPortHousingBuilder::Attribute<double>("z1") +
    KGConicSectPortHousingBuilder::Attribute<double>("r2") +
    KGConicSectPortHousingBuilder::Attribute<double>("z2") +
    KGConicSectPortHousingBuilder::Attribute<int>("axial_mesh_count") +
    KGConicSectPortHousingBuilder::Attribute<int>("longitudinal_mesh_count") +
    KGConicSectPortHousingBuilder::ComplexElement< KGConicSectPortHousing::OrthogonalPort >("orthogonal_port") +
    KGConicSectPortHousingBuilder::ComplexElement< KGConicSectPortHousing::ParaxialPort >("paraxial_port");

  STATICINT sKGConicSectPortHousingSurfaceBuilderStructure =
    KGConicSectPortHousingSurfaceBuilder::Attribute<string>("name") +
    KGConicSectPortHousingSurfaceBuilder::ComplexElement<KGConicSectPortHousing>("conic_section_port_housing");

  STATICINT sKGConicSectPortHousingSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGConicSectPortHousing> >("conic_section_port_housing_surface");

  STATICINT sKGConicSectPortHousingSpaceBuilderStructure =
    KGConicSectPortHousingSpaceBuilder::Attribute<string>("name") +
    KGConicSectPortHousingSpaceBuilder::ComplexElement<KGConicSectPortHousing>("conic_section_port_housing");

  STATICINT sKGConicSectPortHousingSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGConicSectPortHousing> >("conic_section_port_housing_space");

}
