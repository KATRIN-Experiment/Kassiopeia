#ifndef KFMElectrostaticMultipoleCalculatorNumeric_HH__
#define KFMElectrostaticMultipoleCalculatorNumeric_HH__

#include "KFMElectrostaticMultipoleCalculator.hh"

//vector math includes
#include "KVMPathIntegral.hh"
#include "KVMSurfaceIntegral.hh"

#include "KVMField.hh"
#include "KVMFieldWrapper.hh"

#include "KVMLineSegment.hh"
#include "KVMTriangularSurface.hh"
#include "KVMRectangularSurface.hh"


namespace KEMField
{

/*
*
*@file KFMElectrostaticMultipoleCalculatorNumeric.hh
*@class KFMElectrostaticMultipoleCalculatorNumeric
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Dec 19 14:52:58 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticMultipoleCalculatorNumeric: public KFMElectrostaticMultipoleCalculator
{
    public:
        KFMElectrostaticMultipoleCalculatorNumeric();
        virtual ~KFMElectrostaticMultipoleCalculatorNumeric();

        virtual void SetDegree(int l_max);

        virtual void SetNumberOfQuadratureTerms(unsigned int n);

        //constructs unscaled multipole expansion, assuming constant charge density
        //assumes a point cloud with 2 vertics is a wire electrode, 3 vertices is a triangle, and 4 is a rectangle/quadrilateral
        virtual bool ConstructExpansion(double* target_origin, const KFMPointCloud<3>* vertices, KFMScalarMultipoleExpansion* moments) const;

    private:

        unsigned int fSize;

        virtual void RegularSolidHarmonic(const double* point, double* result) const;

        KVMPathIntegral<2>* fNumInt1D;
        KVMSurfaceIntegral<2>* fNumInt2D;

        KVMFieldWrapper< KFMElectrostaticMultipoleCalculatorNumeric,
                        &KFMElectrostaticMultipoleCalculatorNumeric::RegularSolidHarmonic >* fSolidHarmonicWrapper;

        //internal state to compute solid harmonic
        mutable double fOrigin[3];
        mutable double fDel[3];
        mutable double fL;
        mutable double fM;

        mutable std::vector< std::complex<double> > fMoments;

        mutable KVMLineSegment* fLine;
        mutable KVMTriangularSurface* fTriangle;
        mutable KVMRectangularSurface* fRectangle;

        mutable std::complex<double>* fSolidHarmonics;


};


}


#endif /* KFMElectrostaticMultipoleCalculatorNumeric_H__ */
