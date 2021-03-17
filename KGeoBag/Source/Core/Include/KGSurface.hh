#ifndef KGCORE_HH_
#error "do not include KGSurface.hh directly; include KGCore.hh instead."
#else

namespace KGeoBag
{

class KGSurface : public katrin::KTagged
{
  public:
    friend class KGSpace;

    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitSurface(KGSurface* aSurface) = 0;
    };

  public:
    KGSurface();
    KGSurface(KGArea* anArea);
    ~KGSurface() override;

    static std::string Name()
    {
        return "surface";
    }

    //**********
    //extensible
    //**********

  public:
    template<class XExtension> bool HasExtension() const;

    template<class XExtension> const KGExtendedSurface<XExtension>* AsExtension() const;

    template<class XExtension> KGExtendedSurface<XExtension>* AsExtension();

    template<class XExtension> KGExtendedSurface<XExtension>* MakeExtension();

    template<class XExtension> KGExtendedSurface<XExtension>* MakeExtension(const typename XExtension::Surface& aCopy);

  protected:
    std::vector<KGExtensibleSurface*> fExtensions;

    //************
    //structurable
    //************

  public:
    void Orphan();

    const KGSpace* GetParent() const;
    std::string GetPath() const;

  protected:
    KGSpace* fParent;

    //*************
    //transformable
    //*************

  public:
    void Transform(const KTransformation* aTransformation);

    const KGeoBag::KThreeVector& GetOrigin() const;
    const KGeoBag::KThreeVector& GetXAxis() const;
    const KGeoBag::KThreeVector& GetYAxis() const;
    const KGeoBag::KThreeVector& GetZAxis() const;

  protected:
    KGeoBag::KThreeVector fOrigin;
    KGeoBag::KThreeVector fXAxis;
    KGeoBag::KThreeVector fYAxis;
    KGeoBag::KThreeVector fZAxis;

    //********
    //clonable
    //********

  public:
    KGSurface* CloneNode() const;

    //*********
    //visitable
    //*********

  public:
    void AcceptNode(KGVisitor* aVisitor);

    //*********
    //navigable
    //*********

  public:
    void Area(const std::shared_ptr<KGArea>& anArea);
    const std::shared_ptr<KGArea>& Area() const;

    bool Above(const KGeoBag::KThreeVector& aPoint) const;
    KGeoBag::KThreeVector Point(const KGeoBag::KThreeVector& aPoint) const;
    KGeoBag::KThreeVector Normal(const KGeoBag::KThreeVector& aPoint) const;

  private:
    std::shared_ptr<KGArea> fArea;
};

}  // namespace KGeoBag

#endif
