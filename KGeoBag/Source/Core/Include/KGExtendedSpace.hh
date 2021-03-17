#ifndef KGCORE_HH_
#error "do not include KGExtendedSpace.hh directly; include KGCore.hh instead."
#else

namespace KGeoBag
{

template<class XExtension>
class KGExtendedSpace : public katrin::KTagged, public KGExtensibleSpace, public XExtension::Space
{
  public:
    friend class KGExtendedSurface<XExtension>;

  public:
    KGExtendedSpace(KGSpace* aSpace);
    KGExtendedSpace(KGSpace* aSpace, const typename XExtension::Space&);
    ~KGExtendedSpace() override;

    static std::string Name()
    {
        return "extended_space";
    }

  private:
    KGExtendedSpace();
    KGExtendedSpace(const KGExtendedSpace&);

    //********
    //clonable
    //********

  protected:
    KGExtensibleSpace* Clone(KGSpace* aParent = nullptr) const override;

    //*********
    //visitable
    //*********

  public:
    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitExtendedSpace(KGExtendedSpace<XExtension>*) = 0;
    };

    void Accept(KGVisitor* aVisitor) override;

    //**********
    //extensible
    //**********

  public:
    template<class XOtherExtension> bool HasExtension() const;

    template<class XOtherExtension> const KGExtendedSpace<XOtherExtension>* AsExtension() const;

    template<class XOtherExtension> KGExtendedSpace<XOtherExtension>* AsExtension();

    template<class XOtherExtension> KGExtendedSpace<XOtherExtension>* MakeExtension();

    const KGSpace* AsBase() const
    {
        return fSpace;
    }
    KGSpace* AsBase()
    {
        return fSpace;
    }

  private:
    KGSpace* fSpace;

    //************
    //structurable
    //************

  public:
    void Orphan();
    void AddBoundary(KGExtendedSurface<XExtension>* aBoundary);
    void AddChildSurface(KGExtendedSurface<XExtension>* aSurface);
    void AddChildSpace(KGExtendedSpace<XExtension>* aSpace);

    const KGExtendedSpace<XExtension>* GetParent() const;
    const std::vector<KGExtendedSurface<XExtension>*>* GetBoundaries() const;
    const std::vector<KGExtendedSurface<XExtension>*>* GetChildSurfaces() const;
    const std::vector<KGExtendedSpace<XExtension>*>* GetChildSpaces() const;

  private:
    KGExtendedSpace<XExtension>* fParent;
    std::vector<KGExtendedSurface<XExtension>*> fBoundaries;
    std::vector<KGExtendedSurface<XExtension>*> fChildSurfaces;
    std::vector<KGExtendedSpace<XExtension>*> fChildSpaces;

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
    bool Outside(const KGeoBag::KThreeVector& aPoint) const;
};

}  // namespace KGeoBag

#endif
