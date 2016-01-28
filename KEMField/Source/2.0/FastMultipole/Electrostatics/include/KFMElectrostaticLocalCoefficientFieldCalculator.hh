#ifndef KFMElectrostaticLocalCoefficientFieldCalculator_HH__
#define KFMElectrostaticLocalCoefficientFieldCalculator_HH__

#include "KFMElectrostaticLocalCoefficientSet.hh"
#include "KFMLinearAlgebraDefinitions.hh"

#include "KFMPinchonJMatrixCalculator.hh"
#include "KFMComplexSphericalHarmonicExpansionRotator.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticLocalCoefficientFieldCalculator.hh
*@class KFMElectrostaticLocalCoefficientFieldCalculator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jan 23 10:56:51 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticLocalCoefficientFieldCalculator
{
    public:
        KFMElectrostaticLocalCoefficientFieldCalculator();
        virtual ~KFMElectrostaticLocalCoefficientFieldCalculator();

        void SetDegree(int degree);
        void SetExpansionOrigin(const double* origin);

        void SetLocalCoefficients(const KFMElectrostaticLocalCoefficientSet* set);
        void SetRealMoments(const double* real_mom);
        void SetImaginaryMoments(const double* imag_mom);

        //computes the potential and field given a point
        double Potential(const double* p) const;
        void ElectricField(const double* p, double* f) const;


        void ElectricFieldNearZPole(const double* p, double* f) const;
        void ElectricFieldNumerical(const double* p, double* f) const;

    private:

        //computes the potential and field
        double Potential() const;
        void ElectricField(double* f) const;

        int fDegree;
        unsigned int fNTerms;
        unsigned int fSize;
        double fKFactor;
        double fOrigin[3];

        mutable double fDel[3];
        mutable double fCosTheta;
        mutable double fSinTheta;
//        mutable double fPhi;
//        mutable double fRadius;
        mutable double* fPlmArr;
        mutable double* fPlmDervArr;
        mutable double* fRadPowerArr;
        mutable double* fCosMPhiArr;
        mutable double* fSinMPhiArr;

        mutable kfm_matrix* fXForm;
        mutable kfm_vector* fSphField;
        mutable kfm_vector* fCartField;

        const KFMElectrostaticLocalCoefficientSet* fLocalCoeff;
        const double* fRealMoments;
        const double* fImagMoments;

        bool fEvaluate;

        //internal members needed for computing fields under rotation
        KFMPinchonJMatrixCalculator* fJCalc;
        std::vector<kfm_matrix*> fJMatrix;
        mutable KFMComplexSphericalHarmonicExpansionRotator* fRotator;
        mutable std::vector< std::complex<double> > fMomentsA;
        mutable std::vector< std::complex<double> > fMomentsB;

        mutable kfm_vector* fDisplacement;
        mutable kfm_vector* fRotDisplacement;
        mutable kfm_matrix* fRotation;
        mutable kfm_matrix* fTempMx;
        mutable kfm_matrix* fTempMx2;
        mutable double* fRealMomentsB;
        mutable double* fImagMomentsB;


};



}//end of KEMField namespace

#endif /* KFMElectrostaticLocalCoefficientFieldCalculator_H__ */
