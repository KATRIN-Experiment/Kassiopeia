#ifndef KGInsertionCondition_HH__
#define KGInsertionCondition_HH__

#include "KGBoundaryCalculator.hh"
#include "KGCube.hh"
#include "KGNavigableMeshElement.hh"

#include <cmath>
#include <cstddef>


namespace KGeoBag
{


/*
*
*@file KGInsertionCondition.hh
*@class KGInsertionCondition
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 26 10:27:26 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KGInsertionCondition
{
  public:
    KGInsertionCondition() = default;
    ;
    virtual ~KGInsertionCondition() = default;
    ;

    virtual bool ElementIntersectsCube(const KGNavigableMeshElement* element, const KGCube<KGMESH_DIM>* cube) const;
    virtual bool ElementEnclosedByCube(const KGNavigableMeshElement* element, const KGCube<KGMESH_DIM>* cube) const;

  private:
    static bool LineSegmentIntersectsCube(const katrin::KThreeVector& start, const katrin::KThreeVector& end,
                                          const KGCube<KGMESH_DIM>* cube);


    mutable KGBoundaryCalculator<KGMESH_DIM> fBoundaryCalculator;
};


}  // namespace KGeoBag

#endif /* KGInsertionCondition_H__ */
