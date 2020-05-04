#ifndef KGBEMCONVERTER_DEF
#define KGBEMCONVERTER_DEF

#include "KAxis.hh"
#include "KGAxialMesh.hh"
#include "KGBEM.hh"
#include "KGDiscreteRotationalMesh.hh"
#include "KGMesh.hh"
#include "KThreeVector.hh"

#include <vector>

namespace KGeoBag
{

template<template<class, class> class XNode, class XListOne, class XListTwo> class KGDualHierarchy;

template<template<class, class> class XNode, class XTypeOne, class XHeadTwo, class XTailTwo>
class KGDualHierarchy<XNode, XTypeOne, KTypelist<XHeadTwo, XTailTwo>> :
    public KGDualHierarchy<XNode, XTypeOne, XTailTwo>,
    public XNode<XTypeOne, XHeadTwo>
{};

template<template<class, class> class XNode, class XTypeOne, class XHeadTwo>
class KGDualHierarchy<XNode, XTypeOne, KTypelist<XHeadTwo, KNullType>> : public XNode<XTypeOne, XHeadTwo>
{};

template<template<class, class> class XNode, class XHeadOne, class XTailOne, class XHeadTwo, class XTailTwo>
class KGDualHierarchy<XNode, KTypelist<XHeadOne, XTailOne>, KTypelist<XHeadTwo, XTailTwo>> :
    public KGDualHierarchy<XNode, XHeadOne, KTypelist<XHeadTwo, XTailTwo>>,
    public KGDualHierarchy<XNode, XTailOne, KTypelist<XHeadTwo, XTailTwo>>
{};

template<template<class, class> class XNode, class XHeadOne, class XHeadTwo, class XTailTwo>
class KGDualHierarchy<XNode, KTypelist<XHeadOne, KNullType>, KTypelist<XHeadTwo, XTailTwo>> :
    public KGDualHierarchy<XNode, XHeadOne, KTypelist<XHeadTwo, XTailTwo>>
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
    void SetSurfaceContainer(KSurfaceContainer* aContainer)
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
    KSurfaceContainer* fSurfaceContainer;
    double fMinimumArea;
    double fMaximumAspectRatio;
    int fVerbosity;

    class Triangle : public KEMField::KTriangle
    {
      public:
        typedef KEMField::KTriangle ShapePolicy;

        Triangle() {}
        ~Triangle() override {}
    };

    class Rectangle : public KEMField::KRectangle
    {
      public:
        typedef KEMField::KRectangle ShapePolicy;

        Rectangle() {}
        ~Rectangle() override {}
    };

    class LineSegment : public KEMField::KLineSegment
    {
      public:
        typedef KEMField::KLineSegment ShapePolicy;

        LineSegment() {}
        ~LineSegment() override {}
    };

    class ConicSection : public KEMField::KConicSection
    {
      public:
        typedef KEMField::KConicSection ShapePolicy;

        ConicSection() {}
        ~ConicSection() override {}
    };

    class Ring : public KEMField::KRing
    {
      public:
        typedef KEMField::KRing ShapePolicy;

        Ring() {}
        ~Ring() override {}
    };

    class SymmetricTriangle : public KEMField::KSymmetryGroup<KEMField::KTriangle>
    {
      public:
        typedef KEMField::KSymmetryGroup<KEMField::KTriangle> ShapePolicy;

        SymmetricTriangle() {}
        ~SymmetricTriangle() override {}
    };

    class SymmetricRectangle : public KEMField::KSymmetryGroup<KEMField::KRectangle>
    {
      public:
        typedef KEMField::KSymmetryGroup<KEMField::KRectangle> ShapePolicy;

        SymmetricRectangle() {}
        ~SymmetricRectangle() override {}
    };

    class SymmetricLineSegment : public KEMField::KSymmetryGroup<KEMField::KLineSegment>
    {
      public:
        typedef KEMField::KSymmetryGroup<KEMField::KLineSegment> ShapePolicy;

        SymmetricLineSegment() {}
        ~SymmetricLineSegment() override {}
    };

    class SymmetricConicSection : public KEMField::KSymmetryGroup<KEMField::KConicSection>
    {
      public:
        typedef KEMField::KSymmetryGroup<KEMField::KConicSection> ShapePolicy;

        SymmetricConicSection() {}
        ~SymmetricConicSection() override {}
    };

    class SymmetricRing : public KEMField::KSymmetryGroup<KEMField::KRing>
    {
      public:
        typedef KEMField::KSymmetryGroup<KEMField::KRing> ShapePolicy;

        SymmetricRing() {}
        ~SymmetricRing() override {}
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
    void SetSystem(const KThreeVector& anOrigin, const KThreeVector& anXAxis, const KThreeVector& aYAxis,
                   const KThreeVector& aZAxis);
    const KThreeVector& GetOrigin() const;
    const KThreeVector& GetXAxis() const;
    const KThreeVector& GetYAxis() const;
    const KThreeVector& GetZAxis() const;
    const KAxis& GetAxis() const;

    KThreeVector GlobalToInternalPosition(const KThreeVector& aPosition);
    KThreeVector GlobalToInternalVector(const KThreeVector& aVector);
    KThreeVector InternalToGlobalPosition(const KThreeVector& aVector);
    KThreeVector InternalToGlobalVector(const KThreeVector& aVector);

    void VisitSurface(KGSurface* aSurface) override;
    void VisitSpace(KGSpace* aSpace) override;

  protected:
    KPosition LocalToInternal(const KThreeVector& aVector);
    KPosition LocalToInternal(const KTwoVector& aVector);

    virtual void DispatchSurface(KGSurface* aSurface) = 0;
    virtual void DispatchSpace(KGSpace* aSpace) = 0;

  protected:
    KThreeVector fOrigin;
    KThreeVector fXAxis;
    KThreeVector fYAxis;
    KThreeVector fZAxis;
    KAxis fAxis;

    KThreeVector fCurrentOrigin;
    KThreeVector fCurrentXAxis;
    KThreeVector fCurrentYAxis;
    KThreeVector fCurrentZAxis;
    KAxis fCurrentAxis;
};

template<class XBasisPolicy, class XBoundaryPolicy>
class KGBEMConverterNode :
    virtual public KGBEMConverter,
    public KGExtendedSurface<KGBEM<XBasisPolicy, XBoundaryPolicy>>::Visitor,
    public KGExtendedSpace<KGBEM<XBasisPolicy, XBoundaryPolicy>>::Visitor
{
  public:
    KGBEMConverterNode() : KGBEMConverter() {}
    ~KGBEMConverterNode() override {}

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

class KGBEMMeshConverter : public KGDualHierarchy<KGBEMConverterNode, KBasisTypes, KBoundaryTypes>
{
  public:
    KGBEMMeshConverter();
    KGBEMMeshConverter(KSurfaceContainer& aContainer);
    ~KGBEMMeshConverter() override;

  protected:
    void DispatchSurface(KGSurface* aSurface) override;
    void DispatchSpace(KGSpace* aSpace) override;

  private:
    void Add(KGMeshData* aData);
};

class KGBEMAxialMeshConverter : public KGDualHierarchy<KGBEMConverterNode, KBasisTypes, KBoundaryTypes>
{
  public:
    KGBEMAxialMeshConverter();
    KGBEMAxialMeshConverter(KSurfaceContainer& aContainer);
    ~KGBEMAxialMeshConverter() override;

  protected:
    void DispatchSurface(KGSurface* aSurface) override;
    void DispatchSpace(KGSpace* aSpace) override;

  private:
    void Add(KGAxialMeshData* aData);
};

class KGBEMDiscreteRotationalMeshConverter : public KGDualHierarchy<KGBEMConverterNode, KBasisTypes, KBoundaryTypes>
{
  public:
    KGBEMDiscreteRotationalMeshConverter();
    KGBEMDiscreteRotationalMeshConverter(KSurfaceContainer& aContainer);
    ~KGBEMDiscreteRotationalMeshConverter() override;

  protected:
    void DispatchSurface(KGSurface* aSurface) override;
    void DispatchSpace(KGSpace* aSpace) override;

  private:
    void Add(KGDiscreteRotationalMeshData* aData);
};

}  // namespace KGeoBag

#endif
