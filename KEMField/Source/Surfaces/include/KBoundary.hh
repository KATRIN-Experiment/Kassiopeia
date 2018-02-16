#ifndef KBOUNDARY_DEF
#define KBOUNDARY_DEF

#include <vector>

#include "KDataComparator.hh"
#include "KComplexStreamer.hh"

namespace KEMField
{

  struct KDirichletBoundary
  {
    static std::string Name() { return "DirichletBoundary"; }
  };
  struct KNeumannBoundary
  {
    static std::string Name() { return "NeumannBoundary"; }
  };
  struct KCauchyBoundary
  {
    static std::string Name() { return "CauchyBoundary"; }
  };
  struct KRobinBoundary
  {
    static std::string Name() { return "RobinBoundary"; }
  };
  struct KIsolatedBoundary
  {
    static std::string Name() { return "IsolatedBoundary"; }
  };

/**
* @class KBoundary
*
* @brief Base class for Boundaries.
*
* KBoundary is a base class for all classes used as a Boundary policy.  The
* Boundary policy is used to describe the the boundary condition on the PDE of
* interest.
*
* @author T.J. Corona
*/

  class KBoundary
  {
  protected:
    KBoundary() {}
    virtual ~KBoundary() {}

    virtual bool SameBoundaryAs(const KBoundary*) const = 0;

  public:

    friend bool operator== (const KBoundary& lhs,const KBoundary& rhs)
    {
      return lhs.SameBoundaryAs(&rhs);
    }

    friend bool operator!= (const KBoundary& lhs,const KBoundary& rhs)
    {
      return !(lhs.SameBoundaryAs(&rhs));
    }

  };

  // Template for boundary conditions that set values for the function at a
  // boundary (Dirichlet, Cauchy, Robin)
  template <typename ValueType, unsigned int Dim>
  class KBoundaryValue
  {
  protected:
    KBoundaryValue() : fBoundaryValue(Dim,0.) {}
    virtual ~KBoundaryValue() {}

  public:
    void SetBoundaryValue(ValueType v) { fBoundaryValue[0] = v; }
    void SetBoundaryValue(unsigned int i,ValueType v) { if (i<Dim) fBoundaryValue[i] = v; }
    ValueType GetBoundaryValue(unsigned int i=0) const { return (i<Dim ? fBoundaryValue[i] : 0.); }

  protected:
    std::vector<ValueType> fBoundaryValue;
  };

  template <typename ValueType, unsigned int Dim, typename Stream>
  Stream& operator>>(Stream& s,KBoundaryValue<ValueType,Dim>& b)
  {
    s.PreStreamInAction(b);
    ValueType value;
    for (unsigned int i=0;i<Dim;i++)
    {
      s >> value;
      b.SetBoundaryValue(i,value);
    }
    s.PostStreamInAction(b);
    return s;
  }

  template <typename ValueType, unsigned int Dim, typename Stream>
  Stream& operator<<(Stream& s,const KBoundaryValue<ValueType,Dim>& b)
  {
    s.PreStreamOutAction(b);
    for (unsigned int i=0;i<Dim;i++)
      s << b.GetBoundaryValue(i);
    s.PostStreamOutAction(b);
    return s;
  }

  // Template for boundary conditions that set values for the normal flux at a
  // boundary (Neumann, Cauchy, Robin)
  template <typename ValueType, unsigned int Dim>
  class KNormalBoundaryFlux
  {
  protected:
    KNormalBoundaryFlux() : fNormalBoundaryFlux(Dim,0.) {}
    virtual ~KNormalBoundaryFlux() {}

  public:
    void SetNormalBoundaryFlux(ValueType v) { fNormalBoundaryFlux[0] = v; }
    void SetNormalBoundaryFlux(unsigned int i,ValueType v) { if (i<Dim) fNormalBoundaryFlux[i] = v; }
    ValueType GetNormalBoundaryFlux(unsigned int i=0) const { return (i<Dim ? fNormalBoundaryFlux[i] : 0.); }

  protected:
    std::vector<ValueType> fNormalBoundaryFlux;
  };

  template <typename ValueType, unsigned int Dim, typename Stream>
  Stream& operator>>(Stream& s,KNormalBoundaryFlux<ValueType,Dim>& b)
  {
    s.PreStreamInAction(b);
    ValueType value;
    for (unsigned int i=0;i<Dim;i++)
    {
      s >> value;
      b.SetNormalBoundaryFlux(i,value);
    }
    s.PostStreamInAction(b);
    return s;
  }

  template <typename ValueType, unsigned int Dim, typename Stream>
  Stream& operator<<(Stream& s,const KNormalBoundaryFlux<ValueType,Dim>& b)
  {
    s.PreStreamOutAction(b);
    for (unsigned int i=0;i<Dim;i++)
      s << b.GetNormalBoundaryFlux(i);
    s.PostStreamOutAction(b);
    return s;
  }

  // General template class description for all boundary types
  template <class BasisPolicy, class BoundaryCondition>
  class KBoundaryType;

  // Partial specialization for Dirichlet boundary
  template <class BasisPolicy>
  class KBoundaryType<BasisPolicy, KDirichletBoundary> :
    public KBoundary,
    public KDirichletBoundary,
    public KBoundaryValue<typename BasisPolicy::ValueType,BasisPolicy::Dimension>
  {
  public:
    typedef KBoundaryType<BasisPolicy, KDirichletBoundary> SelfType;

  protected:
    KBoundaryType() : KBoundary(), KDirichletBoundary(), KBoundaryValue<typename BasisPolicy::ValueType,BasisPolicy::Dimension>() {}
    virtual ~KBoundaryType() {}

    bool SameBoundaryAs(const KBoundary*) const;

  public:
    static std::string Name() { return KDirichletBoundary::Name(); }
  };

  template <class BasisPolicy>
  bool KBoundaryType<BasisPolicy,KDirichletBoundary>::SameBoundaryAs(const KBoundary* rhs) const
  {
    if (const KBoundaryType<BasisPolicy,KDirichletBoundary>* b = dynamic_cast<const KBoundaryType<BasisPolicy,KDirichletBoundary>*>(rhs))
    {
      static KDataComparator dC;
      return dC.Compare(*this,*b);
    }
    else
    return false;
  }

  template <class BasisPolicy, typename Stream>
  Stream& operator>>(Stream& s,KBoundaryType<BasisPolicy,KDirichletBoundary>& b)
  {
    s.PreStreamInAction(b);
    s >> static_cast<KBoundaryValue<typename BasisPolicy::ValueType,BasisPolicy::Dimension>&>(b);
    s.PostStreamInAction(b);
    return s;
  }

  template <class BasisPolicy, typename Stream>
  Stream& operator<<(Stream& s,const KBoundaryType<BasisPolicy,KDirichletBoundary>& b)
  {
    s.PreStreamOutAction(b);
    s << static_cast<const KBoundaryValue<typename BasisPolicy::ValueType,BasisPolicy::Dimension>&>(b);
    s.PostStreamOutAction(b);
    return s;
  }

  // Partial specialization for Neumann boundary
  template <class BasisPolicy>
  class KBoundaryType<BasisPolicy, KNeumannBoundary> :
    public KBoundary,
    public KNeumannBoundary,
    public KNormalBoundaryFlux<typename BasisPolicy::ValueType,BasisPolicy::Dimension>
  {
  public:
    typedef KBoundaryType<BasisPolicy, KNeumannBoundary> SelfType;

  protected:
    KBoundaryType() : KBoundary(), KNeumannBoundary(), KNormalBoundaryFlux<typename BasisPolicy::ValueType,BasisPolicy::Dimension>() {}
    virtual ~KBoundaryType() {}

    bool SameBoundaryAs(const KBoundary*) const;

  public:
    static std::string Name() { return KNeumannBoundary::Name(); }
  };

  template <class BasisPolicy>
  bool KBoundaryType<BasisPolicy,KNeumannBoundary>::SameBoundaryAs(const KBoundary* rhs) const
  {
    if (const KBoundaryType<BasisPolicy,KNeumannBoundary>* b = dynamic_cast<const KBoundaryType<BasisPolicy,KNeumannBoundary>*>(rhs))
    {
      static KDataComparator dC;
      return dC.Compare(*this,*b);
    }
    else
    return false;
  }

  template <class BasisPolicy, typename Stream>
  Stream& operator>>(Stream& s,KBoundaryType<BasisPolicy,KNeumannBoundary>& b)
  {
    s.PreStreamInAction(b);
    s >> static_cast<KNormalBoundaryFlux<typename BasisPolicy::ValueType,BasisPolicy::Dimension>&>(b);
    s.PostStreamInAction(b);
    return s;
  }

  template <class BasisPolicy, typename Stream>
  Stream& operator<<(Stream& s,const KBoundaryType<BasisPolicy,KNeumannBoundary>& b)
  {
    s.PreStreamOutAction(b);
    s << static_cast<const KNormalBoundaryFlux<typename BasisPolicy::ValueType,BasisPolicy::Dimension>&>(b);
    s.PostStreamOutAction(b);
    return s;
  }

  // Partial specialization for Cauchy boundary
  template <class BasisPolicy>
  class KBoundaryType<BasisPolicy, KCauchyBoundary> :
    public KBoundary,
    public KCauchyBoundary,
    public KBoundaryValue<typename BasisPolicy::ValueType,BasisPolicy::Dimension>,
    public KNormalBoundaryFlux<typename BasisPolicy::ValueType,BasisPolicy::Dimension>
  {
  public:
    typedef KBoundaryType<BasisPolicy, KCauchyBoundary> SelfType;

  protected:
    KBoundaryType() : KBoundary(), KCauchyBoundary(), KBoundaryValue<typename BasisPolicy::ValueType,BasisPolicy::Dimension>(),
		      KNormalBoundaryFlux<typename BasisPolicy::ValueType,BasisPolicy::Dimension>() {}
    virtual ~KBoundaryType() {}

    bool SameBoundaryAs(const KBoundary*) const;

  public:
    static std::string Name() { return KCauchyBoundary::Name(); }
  };

  template <class BasisPolicy>
  bool KBoundaryType<BasisPolicy,KCauchyBoundary>::SameBoundaryAs(const KBoundary* rhs) const
  {
    if (const KBoundaryType<BasisPolicy,KCauchyBoundary>* b = dynamic_cast<const KBoundaryType<BasisPolicy,KCauchyBoundary>*>(rhs))
    {
      static KDataComparator dC;
      return dC.Compare(*this,*b);
    }
    else
    return false;
  }

  template <class BasisPolicy, typename Stream>
  Stream& operator>>(Stream& s,KBoundaryType<BasisPolicy,KCauchyBoundary>& b)
  {
    s.PreStreamInAction(b);
    s >> static_cast<KBoundaryValue<typename BasisPolicy::ValueType,BasisPolicy::Dimension>&>(b)
      >> static_cast<KNormalBoundaryFlux<typename BasisPolicy::ValueType,BasisPolicy::Dimension>&>(b);
    s.PostStreamInAction(b);
    return s;
  }

  template <class BasisPolicy, typename Stream>
  Stream& operator<<(Stream& s,const KBoundaryType<BasisPolicy,KCauchyBoundary>& b)
  {
    s.PreStreamOutAction(b);
    s << static_cast<const KBoundaryValue<typename BasisPolicy::ValueType,BasisPolicy::Dimension>&>(b)
      << static_cast<const KNormalBoundaryFlux<typename BasisPolicy::ValueType,BasisPolicy::Dimension>&>(b);
    s.PostStreamOutAction(b);
    return s;
  }

  // Partial specialization for Robin boundary
  template <class BasisPolicy>
  class KBoundaryType<BasisPolicy, KRobinBoundary> :
    public KBoundary,
    public KRobinBoundary,
    public KBoundaryValue<typename BasisPolicy::ValueType,BasisPolicy::Dimension>,
    public KNormalBoundaryFlux<typename BasisPolicy::ValueType,BasisPolicy::Dimension>
  {
  protected:
  public:
    typedef KBoundaryType<BasisPolicy, KRobinBoundary> SelfType;

    KBoundaryType() : KBoundary(), KRobinBoundary(), KBoundaryValue<typename BasisPolicy::ValueType,BasisPolicy::Dimension>(),
		      KNormalBoundaryFlux<typename BasisPolicy::ValueType,BasisPolicy::Dimension>() {}
    virtual ~KBoundaryType() {}

    bool SameBoundaryAs(const KBoundary*) const;

  public:
    static std::string Name() { return KRobinBoundary::Name(); }
  };

  template <class BasisPolicy>
  bool KBoundaryType<BasisPolicy,KRobinBoundary>::SameBoundaryAs(const KBoundary* rhs) const
  {
    if (const KBoundaryType<BasisPolicy,KRobinBoundary>* b = dynamic_cast<const KBoundaryType<BasisPolicy,KRobinBoundary>*>(rhs))
    {
      static KDataComparator dC;
      return dC.Compare(*this,*b);
    }
    else
    return false;
  }

  template <class BasisPolicy, typename Stream>
  Stream& operator>>(Stream& s,KBoundaryType<BasisPolicy,KRobinBoundary>& b)
  {
    s.PreStreamInAction(b);
    s >> static_cast<KBoundaryValue<typename BasisPolicy::ValueType,BasisPolicy::Dimension>&>(b)
      >> static_cast<KNormalBoundaryFlux<typename BasisPolicy::ValueType,BasisPolicy::Dimension>&>(b);
    s.PostStreamInAction(b);
    return s;
  }

  template <class BasisPolicy, typename Stream>
  Stream& operator<<(Stream& s,const KBoundaryType<BasisPolicy,KRobinBoundary>& b)
  {
    s.PreStreamOutAction(b);
    s << static_cast<const KBoundaryValue<typename BasisPolicy::ValueType,BasisPolicy::Dimension>&>(b)
      << static_cast<const KNormalBoundaryFlux<typename BasisPolicy::ValueType,BasisPolicy::Dimension>&>(b);
    s.PostStreamOutAction(b);
    return s;
  }

  // Partial specialization for Isolated boundary
  template <class BasisPolicy>
  class KBoundaryType<BasisPolicy, KIsolatedBoundary> :
    public KBoundary,
    public KIsolatedBoundary
  {
  public:
    typedef KBoundaryType<BasisPolicy, KIsolatedBoundary> SelfType;

  protected:
    KBoundaryType() : KBoundary(), fBoundaryIndex(0) {}
    virtual ~KBoundaryType() {}

    bool SameBoundaryAs(const KBoundary*) const;

    unsigned int fBoundaryIndex;

  public:
    static std::string Name() { return KIsolatedBoundary::Name(); }

    void SetBoundaryIndex(unsigned int i) { fBoundaryIndex = i; }
    unsigned int GetBoundaryIndex() const { return fBoundaryIndex; }
  };

  template <class BasisPolicy>
  bool KBoundaryType<BasisPolicy,KIsolatedBoundary>::SameBoundaryAs(const KBoundary* rhs) const
  {
    if (const KBoundaryType<BasisPolicy,KIsolatedBoundary>* b = dynamic_cast<const KBoundaryType<BasisPolicy,KIsolatedBoundary>*>(rhs))
    {
      static KDataComparator dC;
      return dC.Compare(*this,*b);
    }
    else
    return false;
  }

  template <class BasisPolicy, typename Stream>
  Stream& operator>>(Stream& s,KBoundaryType<BasisPolicy,KIsolatedBoundary>& b)
  {
    s.PreStreamInAction(b);
    unsigned int boundaryIndex;
    s >> boundaryIndex;
    b.SetBoundaryIndex(boundaryIndex);
    s.PostStreamInAction(b);
    return s;
  }

  template <class BasisPolicy, typename Stream>
  Stream& operator<<(Stream& s,const KBoundaryType<BasisPolicy,KIsolatedBoundary>& b)
  {
    s.PreStreamOutAction(b);
    s << b.GetBoundaryIndex();
    s.PostStreamOutAction(b);
    return s;
  }
}
#endif /* KBOUNDARY_DEF */
