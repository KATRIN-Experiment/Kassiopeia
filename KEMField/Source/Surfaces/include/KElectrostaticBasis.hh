#ifndef KELECTROSTATICBASIS_DEF
#define KELECTROSTATICBASIS_DEF

#include <string>

#include "../../../Surfaces/include/KBasis.hh"

namespace KEMField
{

/**
* @class KElectrostaticBasis
*
* @brief A Basis policy for electrostatics.
*
* KElectrostaticBasis defines the basis of the BEM to be a single-valued real
* number, corresponding to a constant charge density across a surface.
*
* @author T.J. Corona
*/

  class KElectrostaticBasis : public KBasisType<double,1>
  {
  public:
    KElectrostaticBasis() : KBasisType<double,1>() {}
    virtual ~KElectrostaticBasis() {}

    static std::string Name() { return "ElectrostaticBasis"; }
  };

  template <typename Stream>
  Stream& operator>>(Stream& s,KElectrostaticBasis& b)
  {
    s.PreStreamInAction(b);
    s >> static_cast<KBasisType<double,1>&>(b);
    s.PostStreamInAction(b);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KElectrostaticBasis& b)
  {
    s.PreStreamOutAction(b);
    s << static_cast<const KBasisType<double,1>&>(b);
    s.PostStreamOutAction(b);
    return s;
  }
}

#endif /* KELECTROSTATICBASIS_DEF */
