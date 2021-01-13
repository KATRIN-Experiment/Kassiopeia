#ifndef KSHAPE_DEF
#define KSHAPE_DEF

#include "KThreeVector_KEMField.hh"

#include <string>

namespace KEMField
{

/**
* @class KShape
*
* @brief Base class for Shapes.
*
* KShape is a base class for all classes used as a Shape policy.  The Shape
* policy is used to describe the geometric characteristics of a surface.
*
* @author T.J. Corona
*/

class KShape
{
  protected:
    KShape() = default;
    virtual ~KShape() = default;

  public:
    virtual double Area() const = 0;

    virtual const KPosition Centroid() const = 0;

    virtual double DistanceTo(const KPosition&, KPosition&) = 0;

    virtual const KDirection Normal() const = 0;
};
}  // namespace KEMField

#endif /* KSHAPE_DEF */
