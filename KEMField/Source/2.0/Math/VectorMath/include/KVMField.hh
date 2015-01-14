#ifndef KVMVectorField_H
#define KVMVectorField_H

namespace KEMField{


/**
*
*@file KVMVectorField.hh
*@class KVMVectorField
*@brief simple wrapper for function that takes a point in R^n to R^m,
*its domain must be all of R^n, as no domain checking is performed, unlike
*KVMMap. Also there is no need for a jacobian to be defined so this
*function does not necesarily need to be C1.
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jul  6 10:03:12 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KVMField
{
    public:
        KVMField(){;};
        virtual ~KVMField(){;};

        virtual unsigned int GetNDimDomain() const = 0;
        virtual unsigned int GetNDimRange() const = 0;

        virtual void Evaluate(const double* in, double* out) const = 0;

    private:
};

}


#endif /* KVMVectorField_H */
