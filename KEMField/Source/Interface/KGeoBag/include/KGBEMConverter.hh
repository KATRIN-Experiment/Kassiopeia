#ifndef KGBEMCONVERTER_DEF
#define KGBEMCONVERTER_DEF

#include "KGAxialMesh.hh"
#include "KGBEM.hh"
#include "KGDiscreteRotationalMesh.hh"
#include "KGMesh.hh"

#include "KAxis.hh"
#include "KThreeVector.hh"

#include <vector>

namespace KGeoBag
{

template<template<class, class> class XNode, class XListOne, class XListTwo> class KGDualHierarchy;

template<template<class, class> class XNode, class XTypeOne, class XHeadTwo, class XTailTwo>
class KGDualHierarchy<XNode, XTypeOne, KEMField::KTypelist<XHeadTwo, XTailTwo>> :
    public KGDualHierarchy<XNode, XTypeOne, XTailTwo>,
    public XNode<XTypeOne, XHeadTwo>
{};

template<template<class, class> class XNode, class XTypeOne, class XHeadTwo>
class KGDualHierarchy<XNode, XTypeOne, KEMField::KTypelist<XHeadTwo, KEMField::KNullType>> : public XNode<XTypeOne, XHeadTwo>
{};

template<template<class, class> class XNode, class XHeadOne, class XTailOne, class XHeadTwo, class XTailTwo>
class KGDualHierarchy<XNode, KEMField::KTypelist<XHeadOne, XTailOne>, KEMField::KTypelist<XHeadTwo, XTailTwo>> :
    public KGDualHierarchy<XNode, XHeadOne, KEMField::KTypelist<XHeadTwo, XTailTwo>>,
    public KGDualHierarchy<XNode, XTailOne, KEMField::KTypelist<XHeadTwo, XTailTwo>>
{};

template<template<class, class> class XNode, class XHeadOne, class XHeadTwo, class XTailTwo>
class KGDualHierarchy<XNode, KEMField::KTypelist<XHeadOne, KEMField::KNullType>, KEMField::KTypelist<XHeadTwo, XTailTwo>> :
    public KGDualHierarchy<XNode, XHeadOne, KEMField::KTypelist<XHeadTwo, XTailTwo>>
{};

}  // namespace KGeoBag

namespace KGeoBag
{

class KGBEMConverter : public KGVisitor, public KGSurface::Visitor, public KGSpace::Visitor
{
  protected:
    KGBEMConverter();

  public:
    ~KGBEMConverter() override;

  public:
    void SetSurfaceContainer(KEMField::KSurfaceContainer* aContainer)
    {
        fSurfaceContainer = aContainer;
        return;
    }
    void SetMinimumArea(double aMinimumArea)
    {
        fMinimumArea = aMinimumArea;
        return;
    }

    void SetMaximumAspectRatio(double aMaximumRatio)
    {
        fMaximumAspectRatio = aMaximumRatio;
        return;
    }

    void SetVerbosity(int verbosity)
    {
        fVerbosity = verbosity;
    }

  protected:
    KEMField::KSurfaceContainer* fSurfaceContainer;
    double fMinimumArea;
    double fMaximumAspectRatio;
    int fVerbosity;

    class Triangle : public KEMField::KTriangle
    {
      public:
        using ShapePolicy = KEMField::KTriangle;

        Triangle() = default;
        ~Triangle() override = default;
    };

    class Rectangle : public KEMField::KRectangle
    {
      public:
        using ShapePolicy = KEMField::KRectangle;

        Rectangle() = default;
        ~Rectangle() override = default;
    };

    class LineSegment : public KEMField::KLineSegment
    {
      public:
        using ShapePolicy = KEMField::KLineSegment;

        LineSegment() = default;
        ~LineSegment() override = default;
    };

    class ConicSection : public KEMField::KConicSection
    {
      public:
        using ShapePolicy = KEMField::KConicSection;

        ConicSection() = default;
        ~ConicSection() override = default;
    };

    class Ring : public KEMField::KRing
    {
      public:
        using ShapePolicy = KEMField::KRing;

        Ring() = default;
        ~Ring() override = default;
    };

    class SymmetricTriangle : public KEMField::KSymmetryGroup<KEMField::KTriangle>
    {
      public:
        using ShapePolicy = KEMField::KSymmetryGroup<KEMField::KTriangle>;

        SymmetricTriangle() = default;
        ~SymmetricTriangle() override = default;
    };

    class SymmetricRectangle : public KEMField::KSymmetryGroup<KEMField::KRectangle>
    {
      public:
        using ShapePolicy = KEMField::KSymmetryGroup<KEMField::KRectangle>;

        SymmetricRectangle() = default;
        ~SymmetricRectangle() override = default;
    };

    class SymmetricLineSegment : public KEMField::KSymmetryGroup<KEMField::KLineSegment>
    {
      public:
        using ShapePolicy = KEMField::KSymmetryGroup<KEMField::KLineSegment>;

        SymmetricLineSegment() = default;
        ~SymmetricLineSegment() override = default;
    };

    class SymmetricConicSection : public KEMField::KSymmetryGroup<KEMField::KConicSection>
    {
      public:
        using ShapePolicy = KEMField::KSymmetryGroup<KEMField::KConicSection>;

        SymmetricConicSection() = default;
        ~SymmetricConicSection() override = default;
    };

    class SymmetricRing : public KEMField::KSymmetryGroup<KEMField::KRing>
    {
      public:
        using ShapePolicy = KEMField::KSymmetryGroup<KEMField::KRing>;

        SymmetricRing() = default;
        ~SymmetricRing() override = default;
    };

    void Clear();

    std::vector<Triangle*> fTriangles;
    std::vector<Rectangle*> fRectangles;
    std::vector<LineSegment*> fLineSegments;
    std::vector<ConicSection*> fConicSections;
    std::vector<Ring*> fRings;
    std::vector<SymmetricTriangle*> fSymmetricTriangles;
    std::vector<SymmetricRectangle*> fSymmetricRectangles;
    std::vector<SymmetricLineSegment*> fSymmetricLineSegments;
    std::vector<SymmetricConicSection*> fSymmetricConicSections;
    std::vector<SymmetricRing*> fSymmetricRings;

  public:
    void SetSystem(const katrin::KThreeVector& anOrigin, const katrin::KThreeVector& anXAxis, const katrin::KThreeVector& aYAxis,
                   const katrin::KThreeVector& aZAxis);
    const katrin::KThreeVector& GetOrigin() const;
    const katrin::KThreeVector& GetXAxis() const;
    const katrin::KThreeVector& GetYAxis() const;
    const katrin::KThreeVector& GetZAxis() const;
    const katrin::KAxis& GetAxis() const;

    katrin::KThreeVector GlobalToInternalPosition(const katrin::KThreeVector& aPosition);
    katrin::KThreeVector GlobalToInternalVector(const katrin::KThreeVector& aVector);
    katrin::KThreeVector InternalToGlobalPosition(const katrin::KThreeVector& aVector);
    katrin::KThreeVector InternalToGlobalVector(const katrin::KThreeVector& aVector);

    void VisitSurface(KGSurface* aSurface) override;
    void VisitSpace(KGSpace* aSpace) override;

  protected:
    KEMField::KPosition LocalToInternal(const katrin::KThreeVector& aVector);
    KEMField::KPosition LocalToInternal(const katrin::KTwoVector& aVector);

    virtual void DispatchSurface(KGSurface* aSurface) = 0;
    virtual void DispatchSpace(KGSpace* aSpace) = 0;

  protected:
    katrin::KThreeVector fOrigin;
    katrin::KThreeVector fXAxis;
    katrin::KThreeVector fYAxis;
    katrin::KThreeVector fZAxis;
    katrin::KAxis fAxis;

    katrin::KThreeVector fCurrentOrigin;
    katrin::KThreeVector fCurrentXAxis;
    katrin::KThreeVector fCurrentYAxis;
    katrin::KThreeVector fCurrentZAxis;
    katrin::KAxis fCurrentAxis;

    katrin::KTagged* fCurrentElement;
};

template<class XBasisPolicy, class XBoundaryPolicy>
class KGBEMConverterNode :
    virtual public KGBEMConverter,
    public KGExtendedSurface<KGBEM<XBasisPolicy, XBoundaryPolicy>>::Visitor,
    public KGExtendedSpace<KGBEM<XBasisPolicy, XBoundaryPolicy>>::Visitor
{
  public:
    KGBEMConverterNode() : KGBEMConverter() {}
    ~KGBEMConverterNode() override = default;

  public:
    void VisitExtendedSurface(KGExtendedSurface<KGBEM<XBasisPolicy, XBoundaryPolicy>>* aSurface) override
    {
        Add(aSurface);
        return;
    }

    void VisitExtendedSpace(KGExtendedSpace<KGBEM<XBasisPolicy, XBoundaryPolicy>>* aSpace) override
    {
        Add(aSpace);
        return;
    }

  private:
    void Add(KGBEMData<XBasisPolicy, XBoundaryPolicy>* aBEM)
    {
        using namespace KEMField;

        //cout << "adding bem surface of type < " << XBasisPolicy::Name() << ", " << XBoundaryPolicy::Name() << " >..." << endl;

        for (auto tTriangleIt = fTriangles.begin(); tTriangleIt != fTriangles.end(); tTriangleIt++) {
            fSurfaceContainer->push_back(
                new KSurface<XBasisPolicy, XBoundaryPolicy, KTriangle>(*aBEM, *aBEM, **tTriangleIt));
        }
        for (auto tRectangleIt = fRectangles.begin(); tRectangleIt != fRectangles.end(); tRectangleIt++) {
            fSurfaceContainer->push_back(
                new KSurface<XBasisPolicy, XBoundaryPolicy, KRectangle>(*aBEM, *aBEM, **tRectangleIt));
        }
        for (auto tLineSegmentIt = fLineSegments.begin(); tLineSegmentIt != fLineSegments.end(); tLineSegmentIt++) {
            fSurfaceContainer->push_back(
                new KSurface<XBasisPolicy, XBoundaryPolicy, KLineSegment>(*aBEM, *aBEM, **tLineSegmentIt));
        }
        for (auto tConicSectionIt = fConicSections.begin(); tConicSectionIt != fConicSections.end();
             tConicSectionIt++) {
            fSurfaceContainer->push_back(
                new KSurface<XBasisPolicy, XBoundaryPolicy, KConicSection>(*aBEM, *aBEM, **tConicSectionIt));
        }
        for (auto tRingIt = fRings.begin(); tRingIt != fRings.end(); tRingIt++) {
            fSurfaceContainer->push_back(new KSurface<XBasisPolicy, XBoundaryPolicy, KRing>(*aBEM, *aBEM, **tRingIt));
        }
        for (auto tTriangleIt = fSymmetricTriangles.begin(); tTriangleIt != fSymmetricTriangles.end(); tTriangleIt++) {
            fSurfaceContainer->push_back(
                new KSurface<XBasisPolicy, XBoundaryPolicy, KSymmetryGroup<KTriangle>>(*aBEM, *aBEM, **tTriangleIt));
        }
        for (auto tRectangleIt = fSymmetricRectangles.begin(); tRectangleIt != fSymmetricRectangles.end();
             tRectangleIt++) {
            fSurfaceContainer->push_back(
                new KSurface<XBasisPolicy, XBoundaryPolicy, KSymmetryGroup<KRectangle>>(*aBEM, *aBEM, **tRectangleIt));
        }
        for (auto tLineSegmentIt = fSymmetricLineSegments.begin(); tLineSegmentIt != fSymmetricLineSegments.end();
             tLineSegmentIt++) {
            fSurfaceContainer->push_back(
                new KSurface<XBasisPolicy, XBoundaryPolicy, KSymmetryGroup<KLineSegment>>(*aBEM,
                                                                                          *aBEM,
                                                                                          **tLineSegmentIt));
        }
        for (auto tConicSectionIt = fSymmetricConicSections.begin(); tConicSectionIt != fSymmetricConicSections.end();
             tConicSectionIt++) {
            fSurfaceContainer->push_back(
                new KSurface<XBasisPolicy, XBoundaryPolicy, KSymmetryGroup<KConicSection>>(*aBEM,
                                                                                           *aBEM,
                                                                                           **tConicSectionIt));
        }
        for (auto tRingIt = fSymmetricRings.begin(); tRingIt != fSymmetricRings.end(); tRingIt++) {
            fSurfaceContainer->push_back(
                new KSurface<XBasisPolicy, XBoundaryPolicy, KSymmetryGroup<KRing>>(*aBEM, *aBEM, **tRingIt));
        }

        //cout << "...surface container has <" << fSurfaceContainer->size() << "> elements." << endl;
        return;
    }
};

class KGBEMMeshConverter : public KGDualHierarchy<KGBEMConverterNode, KEMField::KBasisTypes, KEMField::KBoundaryTypes>
{
  public:
    KGBEMMeshConverter();
    KGBEMMeshConverter(KEMField::KSurfaceContainer& aContainer);
    ~KGBEMMeshConverter() override;

  protected:
    void DispatchSurface(KGSurface* aSurface) override;
    void DispatchSpace(KGSpace* aSpace) override;

  private:
    bool Add(KGMeshData* aData);
};

class KGBEMAxialMeshConverter : public KGDualHierarchy<KGBEMConverterNode, KEMField::KBasisTypes, KEMField::KBoundaryTypes>
{
  public:
    KGBEMAxialMeshConverter();
    KGBEMAxialMeshConverter(KEMField::KSurfaceContainer& aContainer);
    ~KGBEMAxialMeshConverter() override;

  protected:
    void DispatchSurface(KGSurface* aSurface) override;
    void DispatchSpace(KGSpace* aSpace) override;

  private:
    bool Add(KGAxialMeshData* aData);
};

class KGBEMDiscreteRotationalMeshConverter : public KGDualHierarchy<KGBEMConverterNode, KEMField::KBasisTypes, KEMField::KBoundaryTypes>
{
  public:
    KGBEMDiscreteRotationalMeshConverter();
    KGBEMDiscreteRotationalMeshConverter(KEMField::KSurfaceContainer& aContainer);
    ~KGBEMDiscreteRotationalMeshConverter() override;

  protected:
    void DispatchSurface(KGSurface* aSurface) override;
    void DispatchSpace(KGSpace* aSpace) override;

  private:
    bool Add(KGDiscreteRotationalMeshData* aData);
};

}  // namespace KGeoBag

#endif
