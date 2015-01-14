#include "KZHCoefficientGeneratorElement.hh"

namespace KEMField
{
  bool KZHCoefficientGeneratorElement::IsCoaxial(const KEMCoordinateSystem& coordinateSystem) const
  {
    // first, make sure the z-axes are either parallel or antiparallel
    if (1.-fabs(GetCoordinateSystem().GetZAxis().Dot(coordinateSystem.GetZAxis())) > 1.e-10)
      return false;

    // then, check that they are coaxial
    KDirection betweenOrigins = (GetCoordinateSystem().GetOrigin()-coordinateSystem.GetOrigin());
    if (betweenOrigins.MagnitudeSquared() < 1.e-10)
      return true;
    else
    {
      betweenOrigins = betweenOrigins.Unit();

      if (1.-fabs(GetCoordinateSystem().GetZAxis().Dot(betweenOrigins)) > 1.e-10)
  	return false;
    }

    return true;
  }

  double KZHCoefficientGeneratorElement::AxialOffset(const KEMCoordinateSystem& coordinateSystem) const
  {
    KPosition origin = coordinateSystem.ToLocal(GetCoordinateSystem().GetOrigin());
    return origin[2];
  }
}
