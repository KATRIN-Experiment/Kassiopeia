#ifndef KGEXTRUDEDCLOSEDPATHSPACE_HH_
#define KGEXTRUDEDCLOSEDPATHSPACE_HH_

#include "KGExtrudedPathSurface.hh"
#include "KGFlattenedClosedPathSurface.hh"
#include "KGPlanarClosedPath.hh"
#include "KGShapeMessage.hh"
#include "KGVolume.hh"

namespace KGeoBag
{

template<class XPathType> class KGExtrudedClosedPathSpace : public KGVolume
{
  public:
    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitExtrudedClosedPathSpace(KGExtrudedClosedPathSpace* aExtrudedClosedPathSpace) = 0;
    };

  public:
    KGExtrudedClosedPathSpace() :
        KGVolume(),
        fPath(new XPathType()),
        fZMin(0.),
        fZMax(0.),
        fExtrudedMeshCount(1),
        fExtrudedMeshPower(1.),
        fFlattenedMeshCount(1),
        fFlattenedMeshPower(1.)
    {
        CompilerCheck();
    }
    KGExtrudedClosedPathSpace(const KGExtrudedClosedPathSpace<XPathType>& aCopy) :
        KGVolume(aCopy),
        fPath(aCopy.fPath->Clone()),
        fZMin(aCopy.fZMin),
        fZMax(aCopy.fZMax),
        fExtrudedMeshCount(aCopy.fExtrudedMeshCount),
        fExtrudedMeshPower(aCopy.fExtrudedMeshPower),
        fFlattenedMeshCount(aCopy.fFlattenedMeshCount),
        fFlattenedMeshPower(aCopy.fFlattenedMeshPower)
    {}
    KGExtrudedClosedPathSpace(const std::shared_ptr<XPathType>& aPath) :
        KGVolume(),
        fPath(aPath),
        fZMin(0.),
        fZMax(0.),
        fExtrudedMeshCount(1),
        fExtrudedMeshPower(1.),
        fFlattenedMeshCount(1),
        fFlattenedMeshPower(1.)
    {}
    ~KGExtrudedClosedPathSpace() override = default;

    static std::string Name()
    {
        return "extruded_" + XPathType::Name() + "_space";
    }

  public:
    std::shared_ptr<XPathType>& Path()
    {
        return fPath;
    }
    const std::shared_ptr<XPathType>& Path() const
    {
        return fPath;
    }

    void ZMin(const double& aZMin)
    {
        fZMin = aZMin;
        return;
    }
    const double& ZMin() const
    {
        return fZMin;
    }

    void ZMax(const double& aZMax)
    {
        fZMax = aZMax;
        return;
    }
    const double& ZMax() const
    {
        return fZMax;
    }

    void ExtrudedMeshCount(const unsigned int& aCount)
    {
        fExtrudedMeshCount = aCount;
        return;
    }
    const unsigned int& ExtrudedMeshCount() const
    {
        return fExtrudedMeshCount;
    }

    void ExtrudedMeshPower(const double& aPower)
    {
        fExtrudedMeshPower = aPower;
        return;
    }
    const double& ExtrudedMeshPower() const
    {
        return fExtrudedMeshPower;
    }

    void FlattenedMeshCount(const unsigned int& aCount)
    {
        fFlattenedMeshCount = aCount;
        return;
    }
    const unsigned int& FlattenedMeshCount() const
    {
        return fFlattenedMeshCount;
    }

    void FlattenedMeshPower(const double& aPower)
    {
        fFlattenedMeshPower = aPower;
        return;
    }
    const double& FlattenedMeshPower() const
    {
        return fFlattenedMeshPower;
    }

  protected:
    mutable std::shared_ptr<XPathType> fPath;
    mutable double fZMin;
    mutable double fZMax;
    mutable unsigned int fExtrudedMeshCount;
    mutable double fExtrudedMeshPower;
    mutable unsigned int fFlattenedMeshCount;
    mutable double fFlattenedMeshPower;

  public:
    void VolumeInitialize(BoundaryContainer& aBoundaryContainer) const override
    {
        auto tTop = std::make_shared<KGFlattenedClosedPathSurface<XPathType>>(fPath);
        tTop->Z(fZMax);
        tTop->FlattenedMeshCount(fFlattenedMeshCount);
        tTop->FlattenedMeshPower(fFlattenedMeshPower);
        tTop->SetName("top");
        aBoundaryContainer.push_back(tTop);

        auto tJacket = std::make_shared<KGExtrudedPathSurface<XPathType>>(fPath);
        tJacket->ZMax(fZMax);
        tJacket->ZMin(fZMin);
        tJacket->ExtrudedMeshCount(fExtrudedMeshCount);
        tJacket->ExtrudedMeshPower(fExtrudedMeshPower);
        tJacket->SetName("jacket");
        aBoundaryContainer.push_back(tJacket);

        auto tBottom = std::make_shared<KGFlattenedClosedPathSurface<XPathType>>(fPath);
        tBottom->Z(fZMin);
        tBottom->FlattenedMeshCount(fFlattenedMeshCount);
        tBottom->FlattenedMeshPower(fFlattenedMeshPower);
        tBottom->SetName("bottom");
        aBoundaryContainer.push_back(tBottom);

        return;
    }
    void VolumeAccept(KGVisitor* aVisitor) override
    {
        shapemsg_debug("extruded closed path volume named <" << GetName() << "> is receiving a visitor" << eom);
        auto* tExtrudedClosedPathSpaceVisitor = dynamic_cast<typename KGExtrudedClosedPathSpace::Visitor*>(aVisitor);
        if (tExtrudedClosedPathSpaceVisitor != nullptr) {
            shapemsg_debug("extruded closed path volume named <" << GetName() << "> is accepting a visitor" << eom);
            tExtrudedClosedPathSpaceVisitor->VisitExtrudedClosedPathSpace(this);
            return;
        }
        KGVolume::VolumeAccept(aVisitor);
        return;
    }
    bool VolumeOutside(const KGeoBag::KThreeVector& aPoint) const override
    {
        KTwoVector tXYPoint = aPoint.ProjectXY();
        double tZPoint = aPoint.Z();

        KTwoVector tJacketPoint = fPath->Point(tXYPoint) - tXYPoint;

        KTwoVector tCapPoint(0., 0.);
        if (fPath->Above(tXYPoint) == true) {
            tCapPoint = tJacketPoint;
        }

        double tJacketZ = 0.;
        if (tZPoint > fZMax) {
            tJacketZ = fZMax - tZPoint;
        }
        if (tZPoint < fZMin) {
            tJacketZ = fZMin - tZPoint;
        }

        double tTopZ = tZPoint - fZMax;
        double tBottomZ = fZMin - tZPoint;

        double tJacketDistanceSquared = tJacketPoint.MagnitudeSquared() + tJacketZ * tJacketZ;
        double tTopDistanceSquared = tCapPoint.MagnitudeSquared() + tTopZ * tTopZ;
        double tBottomDistanceSquared = tCapPoint.MagnitudeSquared() + tBottomZ * tBottomZ;

        if (tTopDistanceSquared < tJacketDistanceSquared) {
            if (tTopDistanceSquared < tBottomDistanceSquared) {
                if (tTopZ > 0.) {
                    return true;
                }
                return false;
            }
        }
        if (tBottomDistanceSquared < tJacketDistanceSquared) {
            if (tBottomDistanceSquared < tTopDistanceSquared) {
                if (tBottomZ > 0.) {
                    return true;
                }
                return false;
            }
        }
        return fPath->Above(tXYPoint);
    }
    KGeoBag::KThreeVector VolumePoint(const KGeoBag::KThreeVector& aPoint) const override
    {
        KTwoVector tXYPoint = aPoint.ProjectXY();
        double tZPoint = aPoint.Z();

        KTwoVector tJacketPoint = fPath->Point(tXYPoint) - tXYPoint;

        KTwoVector tCapPoint(0., 0.);
        if (fPath->Above(tXYPoint) == true) {
            tCapPoint = tJacketPoint;
        }

        double tJacketZ = 0.;
        if (tZPoint > fZMax) {
            tJacketZ = fZMax - tZPoint;
        }
        if (tZPoint < fZMin) {
            tJacketZ = fZMin - tZPoint;
        }

        double tTopZ = tZPoint - fZMax;
        double tBottomZ = fZMin - tZPoint;

        double tJacketDistanceSquared = tJacketPoint.MagnitudeSquared() + tJacketZ * tJacketZ;
        double tTopDistanceSquared = tCapPoint.MagnitudeSquared() + tTopZ * tTopZ;
        double tBottomDistanceSquared = tCapPoint.MagnitudeSquared() + tBottomZ * tBottomZ;

        KTwoVector tXYNearest;
        double tZNearest;
        if (tTopDistanceSquared < tJacketDistanceSquared) {
            if (tTopDistanceSquared < tBottomDistanceSquared) {
                tXYNearest = tXYPoint + tCapPoint;
                tZNearest = tZPoint + tTopZ;
                return KGeoBag::KThreeVector(tXYNearest.X(), tXYNearest.Y(), tZNearest);
            }
        }
        if (tBottomDistanceSquared < tJacketDistanceSquared) {
            if (tBottomDistanceSquared < tTopDistanceSquared) {
                tXYNearest = tXYPoint + tCapPoint;
                tZNearest = tZPoint + tBottomZ;
                return KGeoBag::KThreeVector(tXYNearest.X(), tXYNearest.Y(), tZNearest);
            }
        }
        tXYNearest = tXYPoint + tJacketPoint;
        tZNearest = tZPoint + tJacketZ;
        return KGeoBag::KThreeVector(tXYNearest.X(), tXYNearest.Y(), tZNearest);
    }
    KGeoBag::KThreeVector VolumeNormal(const KGeoBag::KThreeVector& aPoint) const override
    {
        KTwoVector tXYPoint = aPoint.ProjectXY();
        double tZPoint = aPoint.Z();

        KTwoVector tJacketPoint = fPath->Point(tXYPoint) - tXYPoint;

        KTwoVector tCapPoint(0., 0.);
        if (fPath->Above(tXYPoint) == true) {
            tCapPoint = tJacketPoint;
        }

        double tJacketZ = 0.;
        if (tZPoint > fZMax) {
            tJacketZ = fZMax - tZPoint;
        }
        if (tZPoint < fZMin) {
            tJacketZ = fZMin - tZPoint;
        }

        double tTopZ = tZPoint - fZMax;
        double tBottomZ = fZMin - tZPoint;

        double tJacketDistanceSquared = tJacketPoint.MagnitudeSquared() + tJacketZ * tJacketZ;
        double tTopDistanceSquared = tCapPoint.MagnitudeSquared() + tTopZ * tTopZ;
        double tBottomDistanceSquared = tCapPoint.MagnitudeSquared() + tBottomZ * tBottomZ;

        KTwoVector tXYNormal(0., 0.);
        double tZNormal = 0.;
        if (tTopDistanceSquared < tJacketDistanceSquared) {
            if (tTopDistanceSquared < tBottomDistanceSquared) {
                tZNormal = 1.;
                return KGeoBag::KThreeVector(tXYNormal.X(), tXYNormal.Y(), tZNormal);
            }
        }
        if (tBottomDistanceSquared < tJacketDistanceSquared) {
            if (tBottomDistanceSquared < tTopDistanceSquared) {
                tZNormal = 1.;
                return KGeoBag::KThreeVector(tXYNormal.X(), tXYNormal.Y(), tZNormal);
            }
        }
        tXYNormal = fPath->Normal(tXYPoint);
        return KGeoBag::KThreeVector(tXYNormal.X(), tXYNormal.Y(), tZNormal);
    }

  private:
    static KGPlanarClosedPath* CompilerCheck()
    {
        XPathType* tPath = nullptr;
        return tPath;
    }
};

}  // namespace KGeoBag

#endif
