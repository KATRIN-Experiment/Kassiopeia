#ifndef KBASIS_DEF
#define KBASIS_DEF

#include "KComplexStreamer.hh"

#include <vector>

namespace KEMField
{

/**
* @class KBasis
*
* @brief Base class for Bases.
*
* KBasis is a base class for all classes used as a Basis policy.  The Basis
* policy is used to describe the the number of degrees of freedom and value type
* intrinsic to the PDE of interest.
*
* @author T.J. Corona
*/

class KBasis
{
  protected:
    KBasis() = default;
    virtual ~KBasis() = default;
};

// Template for bases
template<typename Type, unsigned int Dim> class KBasisType : public KBasis
{
  public:
    typedef Type ValueType;
    enum
    {
        Dimension = Dim
    };

  protected:
    KBasisType() : fSolution(Dim, 0.) {}
    ~KBasisType() override = default;

  public:
    void SetSolution(ValueType v)
    {
        fSolution[0] = v;
    }
    void SetSolution(unsigned int i, ValueType v)
    {
        if (i < Dim)
            fSolution[i] = v;
    }
    ValueType GetSolution(unsigned int i = 0) const
    {
        return (i < Dim ? fSolution[i] : 0.);
    }
    ValueType& GetSolution(unsigned int i = 0)
    {
        return (i < Dim ? fSolution[i] : fSolution[0]);
    }

  protected:
    std::vector<ValueType> fSolution;
};

template<typename Type, unsigned int Dim, typename Stream> Stream& operator>>(Stream& s, KBasisType<Type, Dim>& b)
{
    s.PreStreamInAction(b);
    Type value;
    for (unsigned int i = 0; i < Dim; i++) {
        s >> value;
        b.SetSolution(i, value);
    }
    s.PostStreamInAction(b);
    return s;
}

template<typename Type, unsigned int Dim, typename Stream> Stream& operator<<(Stream& s, const KBasisType<Type, Dim>& b)
{
    s.PreStreamOutAction(b);
    for (unsigned int i = 0; i < Dim; i++)
        s << b.GetSolution(i);
    s.PostStreamOutAction(b);
    return s;
}

}  // namespace KEMField

#endif /* KBASIS_DEF */
