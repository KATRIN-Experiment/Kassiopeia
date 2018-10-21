#include "KZHCoefficientGeneratorElement.hh"

namespace KEMField
{
  bool KZHCoefficientGeneratorElement::IsCoaxial(const KEMCoordinateSystem& coordinateSystem, double coaxialityTolerance ) const
  {
    // first, make sure the z-axes are either parallel or antiparallel
    if (1.-fabs(GetCoordinateSystem().GetZAxis().Dot(coordinateSystem.GetZAxis())) > coaxialityTolerance)
      return false;

    // then, check that they are coaxial
    KDirection betweenOrigins = (GetCoordinateSystem().GetOrigin()-coordinateSystem.GetOrigin());
    if (betweenOrigins.MagnitudeSquared() < coaxialityTolerance)
    {
      return true;
    }
    else
    {
      betweenOrigins = betweenOrigins.Unit();

      if (1.-fabs(GetCoordinateSystem().GetZAxis().Dot(betweenOrigins)) > coaxialityTolerance)
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
