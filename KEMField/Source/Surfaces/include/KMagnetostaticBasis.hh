#ifndef KMAGNETOSTATICBASIS_DEF
#define KMAGNETOSTATICBASIS_DEF

#include "../../../Surfaces/include/KBasis.hh"

#include <string>

namespace KEMField
{

/**
* @class KMagnetostaticBasis
*
* @brief A Basis policy for magnetostatics.
*
* KMagnetostaticBasis defines the basis of the BEM to be a double-valued real
* number, corresponding to two orthogonal constant current densities across a
* surface.
*
* @author T.J. Corona
*/

class KMagnetostaticBasis : public KBasisType<double, 2>
{
  public:
    KMagnetostaticBasis() : KBasisType<double, 2>() {}
    ~KMagnetostaticBasis() override {}

    static std::string Name()
    {
        return "MagnetostaticBasis";
    }
};

template<typename Stream> Stream& operator>>(Stream& s, KMagnetostaticBasis& b)
{
    s.PreStreamInAction(b);
    s >> static_cast<KBasisType<double, 2>&>(b);
    s.PostStreamInAction(b);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KMagnetostaticBasis& b)
{
    s.PreStreamOutAction(b);
    s << static_cast<const KBasisType<double, 2>&>(b);
    s.PostStreamOutAction(b);
    return s;
}
}  // namespace KEMField

#endif /* KMAGNETOSTATICBASIS_DEF */
