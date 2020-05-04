#ifndef KSURFACEVISITORS_DEF
#define KSURFACEVISITORS_DEF

#include "../../../Surfaces/include/KSurfaceTypes.hh"
#include "KTypelistVisitor.hh"

namespace KEMField
{

/**
* @class KBasisVisitor
*
* @brief Base class for Basis visitors.
*
* KBasisVisitor is the base class for visiting different Bases.
*
* @author T.J. Corona
*/

class KBasisVisitor : public KGenLinearHierarchy<KBasisTypes, KVisitorType, KVisitorBase>
{
  public:
    typedef KBasisTypes AcceptedTypes;

    KBasisVisitor() {}
    ~KBasisVisitor() override {}
};

/**
* @class KBoundaryVisitor
*
* @brief Base class for Boundary visitors.
*
* KBoundaryVisitor is the base class for visiting different Boundaries.
*
* @author T.J. Corona
*/

class KBoundaryVisitor : public KGenLinearHierarchy<KBoundaryTypes, KVisitorType, KVisitorBase>
{
  public:
    typedef KBoundaryTypes AcceptedTypes;

    KBoundaryVisitor() {}
    ~KBoundaryVisitor() override {}
};

/**
* @class KShapeVisitor
*
* @brief Base class for Boundary visitors.
*
* KShapeVisitor is the base class for visiting different shapes.
*
* @author T.J. Corona
*/

class KShapeVisitor : public KGenLinearHierarchy<KShapeTypes, KVisitorType, KVisitorBase>
{
  public:
    typedef KShapeTypes AcceptedTypes;

    KShapeVisitor() {}
    ~KShapeVisitor() override {}
};
}  // namespace KEMField

#endif /* KSURFACEVISITORS_DEF */
