#ifndef KGInsertionCondition_HH__
#define KGInsertionCondition_HH__

#include <cstddef>
#include <cmath>

#include "KGCube.hh"
#include "KGNavigableMeshElement.hh"
#include "KGBoundaryCalculator.hh"


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

        KGInsertionCondition(){};
        virtual ~KGInsertionCondition(){};

        virtual bool ElementIntersectsCube(const KGNavigableMeshElement* element, const KGCube<KGMESH_DIM>* cube) const;
        virtual bool ElementEnclosedByCube(const KGNavigableMeshElement* element, const KGCube<KGMESH_DIM>* cube) const;

    private:

        bool LineSegmentIntersectsCube(KThreeVector start, KThreeVector end, const KGCube<KGMESH_DIM>* cube) const;


        mutable KGBoundaryCalculator<KGMESH_DIM> fBoundaryCalculator;

};



}

#endif /* KGInsertionCondition_H__ */
