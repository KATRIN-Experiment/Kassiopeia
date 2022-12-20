#ifndef __KG2DArea_H__
#define __KG2DArea_H__


#include "KG2DShape.hh"

#include "KTwoVector.hh"

#include <vector>

namespace KGeoBag
{

/**
*
*@file KG2DArea.hh
*@class KG2DArea
*@brief 2-dimensional rip-off of KGArea with no transformations allowed
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Jul 28 20:47:47 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KG2DArea : public KG2DShape
{
  public:
    KG2DArea()
    {
        ;
    };
    ~KG2DArea() override
    {
        ;
    };

    //geometry
    virtual bool IsInside(const katrin::KTwoVector& aPoint) const = 0;
    virtual double Area() const = 0;
};


}  // namespace KGeoBag

#endif /* __KG2DArea_H__ */
