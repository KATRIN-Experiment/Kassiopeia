#ifndef KELECTROSTATICQUADRATURELINESEGMENTINTEGRATOR_DEF
#define KELECTROSTATICQUADRATURELINESEGMENTINTEGRATOR_DEF

#include "KSurface.hh"
#include "KEMConstants.hh"

#include <iostream>
#include "KElectrostaticAnalyticLineSegmentIntegrator.hh"

namespace KEMField
{
  class KElectrostaticQuadratureLineSegmentIntegrator :
    public KElectrostaticAnalyticLineSegmentIntegrator
  {
  public:
    typedef KLineSegment Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticQuadratureLineSegmentIntegrator() {}
    ~KElectrostaticQuadratureLineSegmentIntegrator() {}

    double Potential_nNodes( const double* data, const KPosition& P,
  		  const unsigned short halfNoNodes, const double* nodes, const double* weights ) const;
    KEMThreeVector ElectricField_nNodes( const double* data, const KPosition& P,
  		  const unsigned short halfNoNodes, const double* nodes, const double* weights ) const;
    std::pair<KEMThreeVector,double> ElectricFieldAndPotential_nNodes(const double* data, const KPosition& P,
  		  const unsigned short halfNoNodes, const double* nodes, const double* weights ) const;

    double Potential(const KLineSegment* source, const KPosition& P) const;
    KEMThreeVector ElectricField(const KLineSegment* source, const KPosition& P) const;
    std::pair<KEMThreeVector, double> ElectricFieldAndPotential( const KLineSegment* source, const KPosition& P ) const;

    double Potential(const KSymmetryGroup<KLineSegment>* source, const KPosition& P) const;
    KEMThreeVector ElectricField(const KSymmetryGroup<KLineSegment>* source, const KPosition& P) const;
    std::pair<KEMThreeVector, double> ElectricFieldAndPotential( const KSymmetryGroup<KLineSegment>* source, const KPosition& P ) const;

  private:

    // Choice of distance ratio values and integrators based on KEMField test program 'TestIntegratorDistRatioLineSegmentROOT'
    // and distance ratio distribution in KATRIN 3-D main spec model (determined with KSC program 'MainSpectrometerDistanceRatioROOT').

    const double fDrCutOffAna = 2.; // distance ratio for switch from analytic integration to 16-node quadrature
    const double fDrCutOffQuad16 = 27.7; // distance ratio for switch from 16-node quadrature to 4-node quadrature
  };

  static const double gQuadx2[1] = {0.577350269189626};
  static const double gQuadw2[1] = {1.};

  // 3-node quadrature used with n-node function -> node at x=0 counted twice in loop, factor 0.5 necessary before gQuadw3[0]
  static const double gQuadx3[2] = {0., 0.774596669241483};
  static const double gQuadw3[2] = {0.5*0.888888888888889, 0.555555555555556};

  static const double gQuadx4[2] = {0.339981043584856, 0.861136311594053};
  static const double gQuadw4[2] = {0.652145154862546, 0.347854845137454};

  static const double gQuadx6[3] = {0.238619186083197, 0.661209386466265, 0.932469514203152};
  static const double gQuadw6[3] = {0.467913934572691, 0.360761573048139, 0.171324492379170};

  static const double gQuadx8[4] = {0.183434642495650, 0.525532409916329, 0.796666477413627, 0.960289856497536};
  static const double gQuadw8[4] = {0.362683783378362, 0.313706645877887, 0.222381034453374, 0.101228536290376};

  static const double gQuadx16[8] = {0.09501250983763744, 0.28160355077925891, 0.45801677765722739, 0.61787624440264375,
		  0.75540440835500303, 0.86563120238783174, 0.94457502307323258, 0.98940093499164993};
  static const double gQuadw16[8] = {0.189450610455068496, 0.182603415044923589, 0.169156519395002532,
		  0.149595988816576731,
		  0.124628971255533872, 0.095158511682492785, 0.062253523938647892,
		  0.027152459411754095};

  static const double gQuadx32[16] = {0.048307665687738316, 0.144471961582796493, 0.239287362252137075,
		  0.331868602282127650,
		  0.421351276130635345, 0.506899908932229390, 0.587715757240762329,
		  0.663044266930215201,
		  0.732182118740289680, 0.794483795967942407, 0.849367613732569970,
		  0.896321155766052124,
		  0.934906075937739689, 0.964762255587506431, 0.985611511545268335,
		  0.997263861849481564};
  static const double gQuadw32[16] = {0.09654008851472780056, 0.09563872007927485942, 0.09384439908080456564,
		  0.09117387869576388471,
		  0.08765209300440381114, 0.08331192422694675522, 0.07819389578707030647,
		  0.07234579410884850625,
		  0.06582222277636184684, 0.05868409347853554714, 0.05099805926237617619,
		  0.04283589802222680057,
		  0.03427386291302143313, 0.02539206530926205956, 0.01627439473090567065,
		  0.00701861000947009660};
}

#endif /* KELECTROSTATICQUADRATURELINESEGMENTINTEGRATOR_DEF */
