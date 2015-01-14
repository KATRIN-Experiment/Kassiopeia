#ifndef KVMMap_H
#define KVMMap_H

#include "KVMFixedArray.hh"


/**
*
*@file KVMMap.hh
*@class KVMMap
*@brief abstract base class for a map from some sub-set of R^n to R^m
* map must be a function which as a jacobian defined, i.e. it must be C1
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jul  6 14:28:08 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int domainDim, unsigned int rangeDim>
class KVMMap
{
    public:
        KVMMap()
        {
            //initalize limits to zero;
            for(unsigned int i=0; i<domainDim; i++)
            {
                fLowerLimits[i] = 0;
                fUpperLimits[i] = 0;
            }
        }
        virtual ~KVMMap(){;};

        ///returns false if (u) outside of domain
        virtual bool PointInDomain(const KVMFixedArray<double, domainDim >* in) const = 0;

        ///evaluates the function which defines the curve
        virtual bool Evaluate(const KVMFixedArray<double, domainDim >* in, 
                                KVMFixedArray<double, rangeDim >* out) const = 0;

        ///returns the derivative of the variable (specified by outputvarindex)
        ///with respect to the input (u), outputvarindex must be either 0,1, or 2
        ///otherwise it will return NaN.
        virtual bool Jacobian(const KVMFixedArray<double, domainDim >* in,
                                KVMFixedArray< KVMFixedArray<double, rangeDim>, domainDim >* jacobian) const = 0;

        ///The domain need not be a rectangular region of R^n, but 
        ///we must be able to specify a bounding box for the domain
        virtual void GetDomainBoundingBox(KVMFixedArray<double, domainDim>* lower, KVMFixedArray<double, domainDim>* upper) const
        {
            *lower = fLowerLimits;
            *upper = fUpperLimits;
        }

        inline KVMMap(const KVMMap &copyObject)
        {
            fLowerLimits = copyObject.fLowerLimits;
            fUpperLimits = copyObject.fUpperLimits;
        }

    protected:

        bool InBoundingBox(const KVMFixedArray<double, domainDim >* in) const
        {
            bool result = true;
            for(unsigned int i=0; i<domainDim; i++)
            {
                if( ((*in)[i] < fLowerLimits[i] ) || ((*in)[i] > fUpperLimits[i]) )
                {
                    result = false;
                }
            }
            return result;
        }

    //bounding box
    KVMFixedArray<double, domainDim > fLowerLimits;
    KVMFixedArray<double, domainDim > fUpperLimits;

    

};


#endif /* KVMMap_H */

