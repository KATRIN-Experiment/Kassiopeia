#ifndef KVMCompactVolume_DEF
#define KVMCompactVolume_DEF


#include "KVMMap.hh"
#include "KVMFixedArray.hh"

namespace KEMField{

/**
*
*@file KVMCompactVolume.hh
*@class KVMCompactVolume
*@brief
*@details
* abstract base class for a parametric volume in R^3
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jan 27 14:57:07 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

#define KVMVolumeDDim 3
#define KVMVolumeRDim 3


class KVMCompactVolume: public KVMMap<KVMVolumeDDim, KVMVolumeRDim>
{
    public:

        KVMCompactVolume(){;};
        virtual ~KVMCompactVolume(){;};

        ///returns false if (u) outside of domain
        virtual bool PointInDomain(const KVMFixedArray<double, KVMVolumeDDim >* in) const;

        ///evaluates the function which defines the curve
        virtual bool Evaluate(const KVMFixedArray<double, KVMVolumeDDim >* in, 
                                KVMFixedArray<double, KVMVolumeRDim >* out) const;

        ///returns the derivative of the variable (specified by outputvarindex)
        ///with respect to the input (u), outputvarindex must be either 0,1, or 2
        ///otherwise it will return NaN.
        virtual bool Jacobian(const KVMFixedArray<double, KVMVolumeDDim >* in,
                                KVMFixedArray< KVMFixedArray<double, KVMVolumeRDim>, KVMVolumeDDim >* jacobian) const;


        inline KVMCompactVolume(const KVMCompactVolume &copyObject);

        virtual void Initialize(){;}; 
        //initializes any internal variables after free parameters have been set

    protected:


        //functions that define the jacobian
        virtual double dxdu(double u, double v, double w) const = 0;
        virtual double dydu(double u, double v, double w) const = 0;
        virtual double dzdu(double u, double v, double w) const = 0;
        virtual double dxdv(double u, double v, double w) const = 0;
        virtual double dydv(double u, double v, double w) const = 0;
        virtual double dzdv(double u, double v, double w) const = 0;
        virtual double dxdw(double u, double v, double w) const = 0;
        virtual double dydw(double u, double v, double w) const = 0;
        virtual double dzdw(double u, double v, double w) const = 0;


        //functions which define the volume
        virtual double x(double u, double v, double w) const = 0;
        virtual double y(double u, double v, double w) const = 0;
        virtual double z(double u, double v, double w) const = 0;

};


inline KVMCompactVolume::KVMCompactVolume(const KVMCompactVolume &copyObject):
KVMMap<KVMVolumeDDim, KVMVolumeRDim>(copyObject)
{

}


} //end of KEMField namespace
#endif
