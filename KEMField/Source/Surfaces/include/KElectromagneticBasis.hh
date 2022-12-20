#ifndef KELECTROMAGNETICBASIS_DEF
#define KELECTROMAGNETICBASIS_DEF

#include "KBasis.hh"

#include <complex>
#include <string>

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

class KElectromagneticBasis : public KBasisType<std::complex<double>, 4>
{
  public:
    KElectromagneticBasis() : KBasisType<std::complex<double>, 4>() {}
    ~KElectromagneticBasis() override = default;

    static std::string Name()
    {
        return "ElectromagneticBasis";
    }
};

template<typename Stream> Stream& operator>>(Stream& s, KElectromagneticBasis& b)
{
    s.PreStreamInAction(b);
    s >> static_cast<KBasisType<std::complex<double>, 4>&>(b);
    s.PostStreamInAction(b);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KElectromagneticBasis& b)
{
    s.PreStreamOutAction(b);
    s << static_cast<const KBasisType<std::complex<double>, 4>&>(b);
    s.PostStreamOutAction(b);
    return s;
}
}  // namespace KEMField

#endif /* KELECTROMAGNETICBASIS_DEF */
