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

    void SetZ0(const double& d)  { fZ0 = d; fFloatZ0 = (float)d; }
    void SetRho(const double& d) { fRho = d; fRhosquared = (float)fRho*(float)fRho; f1overRhosquared=1./fRhosquared; }

    int    GetNCoeffs()      const { return (int)fCoeffVec.size(); }
    double GetZ0()           const { return fZ0; }
    float GetFloatZ0()       const { return fZ0; }
    double GetRho()          const { return fRho; }
    float GetRhosquared()      const { return fRhosquared; }
    float Get1overRhosquared() const { return f1overRhosquared; }
    double GetCoeff(int i) const { return fCoeffVec[i]; }

    const double* GetRawPointerToCoeff() const {return &(fCoeffVec[0]);};

  private:

    double fZ0;                    ///< Z-coordinate for source point.
    float fFloatZ0;
    double fRho;                   ///< Rho values for source point.
    float fRhosquared;
    float f1overRhosquared;
    std::vector<double> fCoeffVec; ///< Vector of coefficients.

  public:
    template <typename Stream>
    friend Stream& operator>>(Stream& s,KZonalHarmonicSourcePoint& sp)
    {
      s.PreStreamInAction(sp);
      s >> sp.fZ0 >> sp.fRho;
      sp.SetZ0( sp.fZ0 );
      sp.SetRho( sp.fRho );
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
