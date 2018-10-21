#ifndef KELECTROMAGNETICBASIS_DEF
#define KELECTROMAGNETICBASIS_DEF

#include <string>
#include <complex>

#include "KBasis.hh"

namespace KEMField
{

/**
* @class KElectromagneticBasis
*
* @brief A Basis policy for electromagnetics.
*
* KElectromagneticBasis defines the basis of the BEM to be a four-valued complex
* number, corresponding to the face-based equivalent of the RWG basis functions.
*
* @author T.J. Corona
*/

  class KElectromagneticBasis : public KBasisType<std::complex<double>,4>
  {
  public:
    KElectromagneticBasis() : KBasisType<std::complex<double>,4>() {}
    virtual ~KElectromagneticBasis() {}

    static std::string Name() { return "ElectromagneticBasis"; }
  };

  template <typename Stream>
  Stream& operator>>(Stream& s,KElectromagneticBasis& b)
  {
    s.PreStreamInAction(b);
    s >> static_cast<KBasisType<std::complex<double>,4>&>(b);
    s.PostStreamInAction(b);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KElectromagneticBasis& b)
  {
    s.PreStreamOutAction(b);
    s << static_cast<const KBasisType<std::complex<double>,4>&>(b);
    s.PostStreamOutAction(b);
    return s;
  }
}

#endif /* KELECTROMAGNETICBASIS_DEF */
