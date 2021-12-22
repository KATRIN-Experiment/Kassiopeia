#ifndef __KGVertexSideDescriptor_H__
#define __KGVertexSideDescriptor_H__

#include "KTwoVector.hh"

namespace KGeoBag
{
/**
*
*@file KGVertexSideDescriptor.hh
*@class KGVertexSideDescriptor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Aug  8 16:28:47 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

struct KGVertexSideDescriptor
{
    katrin::KTwoVector Vertex;  //the starting vertex associated with this side
    bool IsArc;         //if the side is an arc or a straight line segment
    bool IsRight;       //if it is an arc, does it lie to the left or right of the line joining the vertices
    bool IsCCW;         //pick the CW arc, or the CCW arc
    double Radius;      //radius, if it is an arc
};

}  // namespace KGeoBag


#endif /* __KGVertexSideDescriptor_H__ */
