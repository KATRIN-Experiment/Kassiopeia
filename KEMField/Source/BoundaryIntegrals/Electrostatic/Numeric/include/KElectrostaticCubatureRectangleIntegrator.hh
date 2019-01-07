#ifndef KELECTROSTATICCUBATURERECTANGLEINTEGRATOR_DEF
#define KELECTROSTATICCUBATURERECTANGLEINTEGRATOR_DEF

#include "KElectrostaticRWGRectangleIntegrator.hh"

#include "KSurface.hh"
#include "KEMConstants.hh"
#include "KSymmetryGroup.hh"

#define GRECTCUB7INDEX1

namespace KEMField
{
  class KElectrostaticCubatureRectangleIntegrator :
    public KElectrostaticRWGRectangleIntegrator
  {
  public:
    typedef KRectangle Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticCubatureRectangleIntegrator() {}
    ~KElectrostaticCubatureRectangleIntegrator() {}

    void GaussPoints_Rect4P( const double* data, double* Q ) const;
    void GaussPoints_Rect7P( const double* data, double* Q ) const;
    void GaussPoints_Rect9P( const double* data, double* Q ) const;
    void GaussPoints_Rect12P( const double* data, double* Q ) const;
    void GaussPoints_Rect17P( const double* data, double* Q ) const;
    void GaussPoints_Rect20P( const double* data, double* Q ) const;
    void GaussPoints_Rect33P( const double* data, double* Q ) const;

    double Potential_RectNP( const double* data, const KPosition& P,
    		const unsigned short noPoints, double* Q, const double* weights ) const;
    KThreeVector ElectricField_RectNP( const double* data, const KPosition& P,
    		const unsigned short noPoints, double* Q, const double* weights ) const;
    std::pair<KThreeVector, double> ElectricFieldAndPotential_RectNP( const double* source, const KPosition& P,
    		const unsigned short noPoints, double* Q, const double* weights) const;

    double Potential( const KRectangle* source, const KPosition& P ) const;
    KThreeVector ElectricField( const KRectangle* source, const KPosition& P ) const;
    std::pair<KThreeVector, double> ElectricFieldAndPotential( const KRectangle* source, const KPosition& P ) const;

    double Potential(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const;
    KThreeVector ElectricField(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const;
    std::pair<KThreeVector, double> ElectricFieldAndPotential( const KSymmetryGroup<KRectangle>* source, const KPosition& P ) const;

  private:

    // Choice of distance ratio values and integrators based on KEMField test program 'TestIntegratorDistRatioRectangleROOT'
    // and distance ratio distribution in KATRIN 3-D main spec model (determined with KSC program 'MainSpectrometerDistanceRatioROOT').

    const double fDrCutOffRWG = 4.6; // distance ratio for switch from RWG to 33-point cubature
    const double fDrCutOffCub33 = 44.6; // distance ratio for switch from 33-point cubature to 12-point cubature
    const double fDrCutOffCub12 = 196.2; // distance ratio for switch from 12-point cubature to 7-point cubature

  };

  // 4-point cubature, Gaussian weights
  static const double gRectCub4w[4] = {1./4., 1./4., 1./4., 1./4.};

  // 7-point cubature, Gaussian weights

  // GRECTCUB7INDEX1 defined     :  5-1 7-point formula on p. 246 of Stroud book
  // GRECTCUB7INDEX1 not defined :  5-2 7-point formula on p. 247 of Stroud book
#ifdef GRECTCUB7INDEX1
  static const double gRectCub7w[7] = {2./7., 5./63., 5./63., 5./36., 5./36.,5./36.,5./36.};
#else
  static const double gRectCub7w[7] = {2./7., 25./168., 25./168., 5./48., 5./48.,5./48.,5./48.};
#endif

	// 9-point cubature
  static const double gRectCub9term1[1] = {sqrt(0.6)};
  static const double gRectCub9term2[1] = {1./324.};
  static const double gRectCub9w[9]={gRectCub9term2[0]*64., gRectCub9term2[0]*40., gRectCub9term2[0]*40.,
			gRectCub9term2[0]*40., gRectCub9term2[0]*40., gRectCub9term2[0]*25.,
			gRectCub9term2[0]*25., gRectCub9term2[0]*25., gRectCub9term2[0]*25. };

  // 12-point cubature, Gaussian weights
  static const double gRectCub12B1 = 49/810.;
  static const double gRectCub12B2 = (178981.+2769.*sqrt(583))/1888920.;
  static const double gRectCub12B3 = (178981.-2769.*sqrt(583))/1888920.;
  static const double gRectCub12w[12] = {gRectCub12B1, gRectCub12B1, gRectCub12B1, gRectCub12B1,
		  gRectCub12B2, gRectCub12B2, gRectCub12B2, gRectCub12B2,
		  gRectCub12B3, gRectCub12B3, gRectCub12B3, gRectCub12B3 };

  // 17-point cubature, Gaussian weights
  static const double gRectCub17w0 = 0.52674897119341563786/4.;
  static const double gRectCub17w1 = 0.08887937817019870697/4.;
  static const double gRectCub17w2 = 0.11209960212959648528/4.;
  static const double gRectCub17w3 = 0.39828243926207009528/4.;
  static const double gRectCub17w4 = 0.26905133763978080301/4.;
  static const double gRectCub17w[17]={ gRectCub17w0,
		  gRectCub17w1,gRectCub17w1,gRectCub17w1,gRectCub17w1,
		  gRectCub17w2,gRectCub17w2,gRectCub17w2,gRectCub17w2,
		  gRectCub17w3,gRectCub17w3,gRectCub17w3,gRectCub17w3,
		  gRectCub17w4,gRectCub17w4,gRectCub17w4,gRectCub17w4 };

  // 20-point cubature
  static const double gRectCub20term1[1] = {0.9845398119422523};
  static const double gRectCub20term2[1] = {0.4888863428423724};
  static const double gRectCub20term3[1] = {0.9395672874215217};
  static const double gRectCub20term4[1] = {0.8367103250239890};
  static const double gRectCub20term5[1] = {0.5073767736746132};

  static const double gRectCub20w1[1] = {0.0716134247098111*0.25};
  static const double gRectCub20w2[1] = {0.4540903525515453*0.25};
  static const double gRectCub20w3[1] = {0.0427846154667780*0.25};
  static const double gRectCub20w4[1] = {0.2157558036359328*0.25};

  static const double gRectCub20w[20]={gRectCub20w1[0], gRectCub20w1[0], gRectCub20w1[0], gRectCub20w1[0],
		  gRectCub20w2[0], gRectCub20w2[0], gRectCub20w2[0], gRectCub20w2[0],
		  gRectCub20w3[0], gRectCub20w3[0], gRectCub20w3[0], gRectCub20w3[0],
		  gRectCub20w4[0], gRectCub20w4[0], gRectCub20w4[0], gRectCub20w4[0], gRectCub20w4[0], gRectCub20w4[0], gRectCub20w4[0], gRectCub20w4[0]};

  // 33-point cubature, Gaussian weights
  static const double gRectCub33W[9] = { 0.30038211543122536139/4.,
	0.29991838864499131666e-1/4.,
	0.38174421317083669640e-1/4.,
	0.60424923817749980681e-1/4.,
	0.77492738533105339358e-1/4.,
	0.11884466730059560108/4.,
	0.12976355037000271129/4.,
	0.21334158145718938943/4.,
	0.25687074948196783651/4. };

  static const double gRectCub33w[33] = {gRectCub33W[0],
		  gRectCub33W[1], gRectCub33W[1], gRectCub33W[1], gRectCub33W[1],
		  gRectCub33W[2], gRectCub33W[2], gRectCub33W[2], gRectCub33W[2],
		  gRectCub33W[3], gRectCub33W[3], gRectCub33W[3], gRectCub33W[3],
		  gRectCub33W[4], gRectCub33W[4], gRectCub33W[4], gRectCub33W[4],
		  gRectCub33W[5], gRectCub33W[5], gRectCub33W[5], gRectCub33W[5],
		  gRectCub33W[6], gRectCub33W[6], gRectCub33W[6], gRectCub33W[6],
		  gRectCub33W[7], gRectCub33W[7], gRectCub33W[7], gRectCub33W[7],
		  gRectCub33W[8], gRectCub33W[8], gRectCub33W[8], gRectCub33W[8] };

}

#endif /* KELECTROSTATICCUBATURERECTANGLEINTEGRATOR_DEF */
