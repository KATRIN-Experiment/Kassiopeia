#ifndef KELECTROSTATICCUBATURETRIANGLEINTEGRATOR_DEF
#define KELECTROSTATICCUBATURETRIANGLEINTEGRATOR_DEF

#include "KElectrostaticRWGTriangleIntegrator.hh"

#include "KSurface.hh"
#include "KEMConstants.hh"
#include <cmath>


namespace KEMField
{
  class KElectrostaticCubatureTriangleIntegrator :
	public KElectrostaticRWGTriangleIntegrator
  {
  public:
    typedef KTriangle Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticCubatureTriangleIntegrator() {}
    ~KElectrostaticCubatureTriangleIntegrator() {}

    void GaussPoints_Tri4P( const double* data, double* Q ) const;
    void GaussPoints_Tri7P( const double* data, double* Q ) const;
    void GaussPoints_Tri12P( const double* data, double* Q ) const;
    void GaussPoints_Tri16P( const double* data, double* Q ) const;
    void GaussPoints_Tri19P( const double* data, double* Q ) const;
    void GaussPoints_Tri33P( const double* data, double* Q ) const;

    double Potential_TriNP( const double* data, const KPosition& P,
    		const unsigned short noPoints, double* Q, const double* weights ) const;
    KEMThreeVector ElectricField_TriNP( const double* data, const KPosition& P,
    		const unsigned short noPoints, double* Q, const double* weights ) const;
    std::pair<KEMThreeVector, double> ElectricFieldAndPotential_TriNP( const double* data, const KPosition& P,
    		const unsigned short noPoints, double* Q, const double* weights) const;

    double Potential(const KTriangle* source, const KPosition& P) const;
    KEMThreeVector ElectricField(const KTriangle* source, const KPosition& P) const;
    std::pair<KEMThreeVector, double> ElectricFieldAndPotential( const KTriangle* source, const KPosition& P ) const;

    double Potential(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const;
    KEMThreeVector ElectricField(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const;
    std::pair<KEMThreeVector, double> ElectricFieldAndPotential( const KSymmetryGroup<KTriangle>* source, const KPosition& P ) const;

  private:

    // Choice of distance ratio values and integrators based on KEMField test program 'TestIntegratorDistRatioTriangleROOT'
    // and distance ratio distribution in KATRIN 3-D main spec model (determined with KSC program 'MainSpectrometerDistanceRatioROOT').

    const double fDrCutOffRWG = 3.42; // distance ratio for switch from RWG to 33-point cubature
    const double fDrCutOffCub33 = 30.4; // distance ratio for switch from 33-point cubature to 12-point cubature
    const double fDrCutOffCub12 = 131.4; // distance ratio for switch from 12-point cubature to 7-point cubature

  };

  // general values
  static const double gTriCubCst0 = 1./3.; /* t */
  static const double gTriCubCst1 = sqrt(15);
  static const double gTriCubCst2 = (1.+gTriCubCst1)/7.; /* r */
  static const double gTriCubCst3 = (1.-gTriCubCst1)/7.; /* s */

  // 4-point cubature

  // barycentric (area) coordinates of the Gaussian points
  static const double gTriCub4alpha[3] = { gTriCubCst0, 3./5. };
  static const double gTriCub4beta[3] = { gTriCubCst0, 1./5. };
  static const double gTriCub4gamma[3] = { gTriCubCst0, 1./5. };
  // Gaussian weights
  static const double gTriCub4w2[2] = { -9./16., 25./48. };
  static const double gTriCub4w[4] = { gTriCub4w2[0], gTriCub4w2[1], gTriCub4w2[1], gTriCub4w2[1] };

  // 7-point cubature

  // barycentric (area) coordinates of the Gaussian points
  static const double gTriCub7alpha[3] = { gTriCubCst0,
		  gTriCubCst0+(2.*gTriCubCst0*gTriCubCst3),
		  gTriCubCst0+(2.*gTriCubCst0*gTriCubCst2) };
  static const double gTriCub7beta[3] = { gTriCubCst0,
		  gTriCubCst0-(gTriCubCst0*gTriCubCst3),
		  gTriCubCst0-(gTriCubCst0*gTriCubCst2) };
  static const double gTriCub7gamma[3] = { gTriCubCst0,
		  gTriCubCst0-(gTriCubCst0*gTriCubCst3),
		  gTriCubCst0-(gTriCubCst0*gTriCubCst2) };
  // Gaussian weights
  static const double gTriCub7w3[3] = { 9/40., (155. + gTriCubCst1)/1200., (155. - gTriCubCst1)/1200.};
  static const double gTriCub7w[7] = {gTriCub7w3[0], gTriCub7w3[1], gTriCub7w3[1], gTriCub7w3[1], gTriCub7w3[2], gTriCub7w3[2], gTriCub7w3[2]};

  // 12-point cubature

  // barycentric (area) coordinates of the Gaussian points
  static const double gTriCub12alpha[4] = { 0.6238226509439084e-1,  0.5522545665692000e-1, 0.3432430294509488e-1, 0.5158423343536001 };
  static const double gTriCub12beta[4] = { 0.6751786707392436e-1, 0.3215024938520156, 0.6609491961867980, 0.2777161669764050 };
  static const double gTriCub12gamma[4] = { 0.8700998678316848, 0.6232720494910644, 0.3047265008681072, 0.2064414986699949 };
  // Gaussian weights
  static const double gTriCub12w4[4] = { 0.2651702815743450e-1, 0.4388140871444811e-1, 0.2877504278497528e-1, 0.6749318700980879e-1 };
  static const double gTriCub12w[12] = {gTriCub12w4[0]*2., gTriCub12w4[0]*2., gTriCub12w4[0]*2.,
		  gTriCub12w4[1]*2., gTriCub12w4[1]*2., gTriCub12w4[1]*2.,
		  gTriCub12w4[2]*2., gTriCub12w4[2]*2., gTriCub12w4[2]*2.,
		  gTriCub12w4[3]*2., gTriCub12w4[3]*2., gTriCub12w4[3]*2.};

  // 16-point cubature

  // barycentric (area) coordinates of the Gaussian points
  static const double gTriCub16alpha[5] = { gTriCubCst0, 0.081414823414554, 0.658861384496480, 0.898905543365938, 0.008394777409958 };
  static const double gTriCub16beta[5] = { gTriCubCst0, 0.459292588292723, 0.170569307751760, 0.050547228317031, 0.263112829634638 };
  static const double gTriCub16gamma[5] = { gTriCubCst0, 0.459292588292723, 0.170569307751760, 0.050547228317031, 0.728492392955404 };
  // Gaussian weights
  static const double gTriCub16w5[5] = { 0.144315607677787, 0.095091634267285, 0.103217370534718, 0.032458497623198, 0.027230314174435 };
  static const double gTriCub16w[16] = { gTriCub16w5[0],
		  gTriCub16w5[1], gTriCub16w5[1], gTriCub16w5[1],
		  gTriCub16w5[2], gTriCub16w5[2], gTriCub16w5[2],
		  gTriCub16w5[3], gTriCub16w5[3], gTriCub16w5[3],
		  gTriCub16w5[4], gTriCub16w5[4], gTriCub16w5[4], gTriCub16w5[4], gTriCub16w5[4], gTriCub16w5[4] };

  // 19-point cubature

  // barycentric (area) coordinates of the Gaussian points
  static const double gTriCub19alpha[6] = { gTriCubCst0, 2.063496160252593e-2, 1.258208170141290e-1, 6.235929287619356e-1, 9.105409732110941e-1, 3.683841205473626e-2 };
  static const double gTriCub19beta[6] = { gTriCubCst0, 4.896825191987370e-1, 4.370895914929355e-1, 1.882035356190322e-1, 4.472951339445297e-2, 7.411985987844980e-1 };
  static const double gTriCub19gamma[6] = { 1.-gTriCub19alpha[0]-gTriCub19beta[0],
		  1.-gTriCub19alpha[1]-gTriCub19beta[1],
		  1.-gTriCub19alpha[2]-gTriCub19beta[2],
		  1.-gTriCub19alpha[3]-gTriCub19beta[3],
		  1.-gTriCub19alpha[4]-gTriCub19beta[4],
		  1.-gTriCub19alpha[5]-gTriCub19beta[5] };
  // Gaussian weights
  static const double gTriCub19w6[6] = { 9.713579628279610e-2, 9.400410068141950e-2/3.,  2.334826230143263e-1/3., 2.389432167816273e-1/3., 7.673302697609430e-2/3., 2.597012362637364e-1/6. };
  static const double gTriCub19w[19] = { gTriCub19w6[0],
		  gTriCub19w6[1], gTriCub19w6[1], gTriCub19w6[1],
		  gTriCub19w6[2], gTriCub19w6[2], gTriCub19w6[2],
		  gTriCub19w6[3], gTriCub19w6[3], gTriCub19w6[3],
		  gTriCub19w6[4], gTriCub19w6[4], gTriCub19w6[4],
		  gTriCub19w6[5], gTriCub19w6[5], gTriCub19w6[5], gTriCub19w6[5], gTriCub19w6[5], gTriCub19w6[5]};

  // 33-point cubature

  // barycentric (area) coordinates of the Gaussian points
  // static const double gTriCub33alpha[8] = { 0.023565220452390,0.120551215411079,0.457579229975768,0.744847708916828,0.957365299093579,
  //                                           0.115343494534698,0.022838332222257,0.025734050548330 };
  // static const double gTriCub33beta[8] = { 0.488217389773805,0.439724392294460,0.271210385012116,0.127576145541586,0.021317350453210,
  //                                          0.275713269685514,0.281325580989940,0.116251915907597 };
  static const double gTriCub33alpha[8] = {
		  4.570749859701478e-01,
		  1.197767026828138e-01,
		  2.359249810891690e-02,
		  7.814843446812914e-01,
		  9.507072731273288e-01,
		  1.162960196779266e-01,
		  2.303415635526714e-02,
		  2.138249025617059e-02 };
  static const double gTriCub33beta[8] = {
		  2.714625070149261e-01,
		  4.401116486585931e-01,
		  4.882037509455416e-01,
		  1.092578276593543e-01,
		  2.464636343633559e-02,
		  2.554542286385173e-01,
		  2.916556797383410e-01,
		  1.272797172335894e-01  };
  static const double gTriCub33gamma[8] = {
		  1. - gTriCub33alpha[0] - gTriCub33beta[0],
		  1. - gTriCub33alpha[1] - gTriCub33beta[1],
		  1. - gTriCub33alpha[2] - gTriCub33beta[2],
		  1. - gTriCub33alpha[3] - gTriCub33beta[3],
		  1. - gTriCub33alpha[4] - gTriCub33beta[4],
		  1. - gTriCub33alpha[5] - gTriCub33beta[5],
		  1. - gTriCub33alpha[6] - gTriCub33beta[6],
		  1. - gTriCub33alpha[7] - gTriCub33beta[7]
  };
  // Gaussian weights
  // static const double gTriCub33w8[8] = {0.025731066440455,0.043692544538038,0.062858224217885,0.034796112930709,0.006166261051559,
  //                                       0.040371557766381,0.022356773202303,0.017316231108659 };
  const double gTriCub33w8[8] = {6.254121319590276e-02, 4.991833492806094e-02, 2.426683808145203e-02, 2.848605206887754e-02,
		  7.931642509973638e-03,  4.322736365941421e-02, 2.178358503860756e-02, 1.508367757651144e-02 };
  static const double gTriCub33w[33] = {
		  gTriCub33w8[0], gTriCub33w8[0], gTriCub33w8[0],
		  gTriCub33w8[1], gTriCub33w8[1], gTriCub33w8[1],
		  gTriCub33w8[2], gTriCub33w8[2], gTriCub33w8[2],
		  gTriCub33w8[3], gTriCub33w8[3], gTriCub33w8[3],
		  gTriCub33w8[4], gTriCub33w8[4], gTriCub33w8[4],
		  gTriCub33w8[5], gTriCub33w8[5], gTriCub33w8[5], gTriCub33w8[5], gTriCub33w8[5], gTriCub33w8[5],
		  gTriCub33w8[6], gTriCub33w8[6], gTriCub33w8[6], gTriCub33w8[6], gTriCub33w8[6], gTriCub33w8[6],
		  gTriCub33w8[7], gTriCub33w8[7], gTriCub33w8[7], gTriCub33w8[7], gTriCub33w8[7], gTriCub33w8[7]
  };

}

#endif /* KELECTROSTATICCUBATURETRIANGLEINTEGRATOR_DEF */
