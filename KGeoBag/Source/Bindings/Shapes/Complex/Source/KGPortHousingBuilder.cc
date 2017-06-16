#include "KGPortHousingBuilder.hh"
#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

  STATICINT sKGPortHousingRectangularPortBuilderStructure =
    KGPortHousingRectangularPortBuilder::Attribute<double>("length") +
    KGPortHousingRectangularPortBuilder::Attribute<double>("width") +
    KGPortHousingRectangularPortBuilder::Attribute<double>("x") +
    KGPortHousingRectangularPortBuilder::Attribute<double>("y") +
    KGPortHousingRectangularPortBuilder::Attribute<double>("z");

  STATICINT sKGPortHousingCircularPortBuilderStructure =
    KGPortHousingCircularPortBuilder::Attribute<double>("radius") +
    KGPortHousingCircularPortBuilder::Attribute<double>("x") +
    KGPortHousingCircularPortBuilder::Attribute<double>("y") +
    KGPortHousingCircularPortBuilder::Attribute<double>("z");

  STATICINT sKGPortHousingBuilderStructure =
    KGPortHousingBuilder::Attribute<double>("radius") +
    KGPortHousingBuilder::Attribute<double>("x1") +
    KGPortHousingBuilder::Attribute<double>("y1") +
    KGPortHousingBuilder::Attribute<double>("z1") +
    KGPortHousingBuilder::Attribute<double>("x2") +
    KGPortHousingBuilder::Attribute<double>("y2") +
    KGPortHousingBuilder::Attribute<double>("z2") +
    KGPortHousingBuilder::Attribute<int>("longitudinal_mesh_count") +
    KGPortHousingBuilder::Attribute<int>("axial_mesh_count") +
    KGPortHousingBuilder::ComplexElement< KGPortHousing::RectangularPort >("rectangular_port") +
    KGPortHousingBuilder::ComplexElement< KGPortHousing::CircularPort >("circular_port");

  STATICINT sKGPortHousingSurfaceBuilderStructure =
    KGPortHousingSurfaceBuilder::Attribute<string>("name") +
    KGPortHousingSurfaceBuilder::ComplexElement<KGPortHousing>("port_housing");

  STATICINT sKGPortHousingSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGPortHousing> >("port_housing_surface");

  STATICINT sKGPortHousingSpaceBuilderStructure =
    KGPortHousingSpaceBuilder::Attribute<string>("name") +
    KGPortHousingSpaceBuilder::ComplexElement<KGPortHousing>("port_housing");

  STATICINT sKGPortHousingSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGPortHousing> >("port_housing_space");

}
