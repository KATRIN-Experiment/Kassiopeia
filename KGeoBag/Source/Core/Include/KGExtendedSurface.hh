#ifndef KGCORE_HH_
#error "do not include KGExtendedSurface.hh directly; include KGCore.hh instead."
#else

namespace KGeoBag
{

template<class XExtension>
class KGExtendedSurface : public katrin::KTagged, public XExtension::Surface, public KGExtensibleSurface
{
  public:
    friend class KGExtendedSpace<XExtension>;

  public:
    KGExtendedSurface(KGSurface* aSurface);
    KGExtendedSurface(KGSurface* aSurface, const typename XExtension::Surface&);
    ~KGExtendedSurface() override;

    static std::string Name()
    {
        return "extended_surface";
    }

  private:
    KGExtendedSurface();
    KGExtendedSurface(const KGExtendedSurface&);

    //********
    //clonable
    //********

  protected:
    KGExtensibleSurface* Clone(KGSurface* aParent = nullptr) const override;

    //*********
    //visitable
    //*********

  public:
    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitExtendedSurface(KGExtendedSurface<XExtension>*) = 0;
    };

    void Accept(KGVisitor* aVisitor) override;

    //**********
    //extensible
    //**********

  public:
    template<class XOtherExtension> bool HasExtension() const;

    template<class XOtherExtension> const KGExtendedSurface<XOtherExtension>* AsExtension() const;

    template<class XOtherExtension> KGExtendedSurface<XOtherExtension>* AsExtension();

    template<class XOtherExtension> KGExtendedSurface<XOtherExtension>* MakeExtension();

    const KGSurface* AsBase() const
    {
        return fSurface;
    }
    KGSurface* AsBase()
    {
        return fSurface;
    }

  private:
    KGSurface* fSurface;

    //************
    //structurable
    //************

  public:
    void Orphan();

    const KGExtendedSpace<XExtension>* GetParent() const;

  private:
    KGExtendedSpace<XExtension>* fParent;

    //*************
    //transformable
    //*************

  public:
    void Transform(const KTransformation* aTransformation);

    const KGeoBag::KThreeVector& GetOrigin() const;
    const KGeoBag::KThreeVector& GetXAxis() const;
    const KGeoBag::KThreeVector& GetYAxis() const;
    const KGeoBag::KThreeVector& GetZAxis() const;

    //*********
    //navigable
    //*********

  public:
    KGeoBag::KThreeVector Point(const KGeoBag::KThreeVector& aPoint) const;
    KGeoBag::KThreeVector Normal(const KGeoBag::KThreeVector& aPoint) const;
    bool Above(const KGeoBag::KThreeVector& aPoint) const;
};

}  // namespace KGeoBag

#endif
