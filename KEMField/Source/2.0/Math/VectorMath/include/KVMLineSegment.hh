#ifndef KVMLineSegment_DEF
#define KVMLineSegment_DEF

#include "KVMCompactCurve.hh"
#include "KVMSpaceLineSegment.hh"

namespace KEMField{


/**
*
*@file KVMLineSegment.hh
*@class KVMLineSegment
*@brief
*@details
* parameterization class for a line segment in R^3, orientation is from point1 to point2
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jan 27 14:57:07 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KVMLineSegment: public KVMCompactCurve, public KVMSpaceLineSegment
{
    public:

        KVMLineSegment(); 
        virtual ~KVMLineSegment(){;};

        inline KVMLineSegment(const KVMLineSegment &copyObject);

        inline KVMLineSegment& operator=(const KVMSpaceLineSegment& rhs);
        
        virtual void Initialize(); 

    protected:

        ///functions which define the curve's jacobian
        virtual double dxdu(const double& /*u*/) const;
        virtual double dydu(const double& /*u*/) const;
        virtual double dzdu(const double& /*u*/) const;

        ///functions which define the curve
        virtual double x(const double& u) const;
        virtual double y(const double& u) const;
        virtual double z(const double& u) const;
};

inline KVMLineSegment::KVMLineSegment(const KVMLineSegment &copyObject):
KVMCompactCurve(copyObject),
KVMSpaceLineSegment(copyObject)
{
    Initialize();
};

inline KVMLineSegment& KVMLineSegment::operator=(const KVMSpaceLineSegment& rhs)
{
    if(this != &rhs)
    {
        KVMSpaceLineSegment::operator=(rhs);
        Initialize();
    }
    return *this;
}

} //end of KEMField namespace

#endif
