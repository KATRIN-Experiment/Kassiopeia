#ifndef KZONALHARMONICSOURCEPOINT_DEF
#define KZONALHARMONICSOURCEPOINT_DEF

#include <vector>
#include <string>

namespace KEMField
{
  /**
   * @class KZonalHarmonicSourcepoint
   *
   * @brief KZonalHarmonicSourcePoint contains the source point location and the
   * coefficients of the zonal harmonic expansion.
   *
   * Used for axially symmetric magnetic and electric field calculations.
   *
   * @author T.J. Corona
   */

  class KZonalHarmonicSourcePoint
  {
  public:
    KZonalHarmonicSourcePoint() {}

    static std::string Name() { return "ZonalHarmonicSourcePoint"; }

    void SetValues(double z0,
		   double rho,
		   std::vector<double>& coeffs);

    virtual ~KZonalHarmonicSourcePoint() { fCoeffVec.clear(); }

    void SetZ0(const double& d)  { fZ0 = d; }
    void SetRho(const double& d) { fRho = d; }

    int    GetNCoeffs()      const { return (int)fCoeffVec.size(); }
    double GetZ0()           const { return fZ0; }
    double GetRho()          const { return fRho; }
    double GetCoeff(int i) const { return fCoeffVec.at(i); }

  private:

    double fZ0;                    ///< Z-coordinate for source point.
    double fRho;                   ///< Rho values for source point.
    std::vector<double> fCoeffVec; ///< Vector of coefficients.

  public:
    template <typename Stream>
    friend Stream& operator>>(Stream& s,KZonalHarmonicSourcePoint& sp)
    {
      s.PreStreamInAction(sp);
      s >> sp.fZ0 >> sp.fRho;
      unsigned int nCoeffs;
      s >> nCoeffs;
      sp.fCoeffVec.clear();
      double coeff;
      for (unsigned int i=0;i<nCoeffs;i++)
      {
	s >> coeff;
	sp.fCoeffVec.push_back(coeff);
      }
      s.PostStreamInAction(sp);
      return s;
    }

    template <typename Stream>
    friend Stream& operator<<(Stream& s,const KZonalHarmonicSourcePoint& sp)
    {
      s.PreStreamOutAction(sp);
      s << sp.fZ0;
      s << sp.fRho;
      s << (unsigned int)(sp.fCoeffVec.size());
      for (unsigned int i=0;i<sp.fCoeffVec.size();i++)
	s << sp.fCoeffVec.at(i);
      s.PostStreamOutAction(sp);
      return s;
    }
  };

} // end namespace KEMField

#endif /* KZONALHARMONICSOURCEPOINT_DEF */
