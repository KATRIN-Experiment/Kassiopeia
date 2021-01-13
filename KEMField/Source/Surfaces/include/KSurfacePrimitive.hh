#ifndef KSURFACEPRIMITIVE_DEF
#define KSURFACEPRIMITIVE_DEF

#include <string>

namespace KEMField
{

/**
* @class KSurfacePrimitive
*
* @brief Base class for KSurface.
*
* KSurfacePrimitive is a policy-agnostic base class for all surfaces.  It is
* used as the primary handle by which surface elements are manipulated.  It is
* also the base element for the visitor pattern (see "Modern C++ Design: Generic
* Programming and Design Patterns Applied" by Andrei Alexandrescu, 2001),
* one means of interacting with surfaces via this handle.
*
* @author T.J. Corona
*/

struct KSurfaceID;
class KBasis;
class KBoundary;
class KShape;
class KBasisVisitor;
class KBoundaryVisitor;
class KShapeVisitor;

class KSurfacePrimitive
{
  protected:
    KSurfacePrimitive() = default;

  public:
    virtual ~KSurfacePrimitive() = default;

    static std::string Name()
    {
        return "SurfacePrimitive";
    }

    virtual std::string GetName() const = 0;
    virtual KSurfaceID& GetID() const = 0;
    virtual KSurfacePrimitive* Clone() const = 0;

    virtual KBasis* GetBasis() = 0;
    virtual KBoundary* GetBoundary() = 0;
    virtual KShape* GetShape() = 0;

    virtual void Accept(KBasisVisitor& visitor) = 0;
    virtual void Accept(KBoundaryVisitor& visitor) = 0;
    virtual void Accept(KShapeVisitor& visitor) = 0;
};
}  // namespace KEMField

#endif /* KSURFACEPRIMITIVE_DEF */
