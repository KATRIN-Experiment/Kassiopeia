#ifndef KGROTATEDOPENPATHSPACE_HH_
#define KGROTATEDOPENPATHSPACE_HH_

#include "KGFlattenedCircleSurface.hh"
#include "KGPlanarOpenPath.hh"
#include "KGRotatedPathSurface.hh"
#include "KGShapeMessage.hh"
#include "KGVolume.hh"

#include <iomanip>
#include <memory>

namespace KGeoBag
{

template<class XPathType> class KGRotatedOpenPathSpace : public KGVolume
{
  public:
    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitRotatedOpenPathSpace(KGRotatedOpenPathSpace* aRotatedOpenPathSpace) = 0;
    };

  public:
    KGRotatedOpenPathSpace() :
        KGVolume(),
        fPath(new XPathType()),
        fTopPath(new KGPlanarCircle()),
        fBottomPath(new KGPlanarCircle()),
        fSign(1.),
        fRotatedMeshCount(64),
        fFlattenedMeshCount(8),
        fFlattenedMeshPower(1.)
    {
        CompilerCheck();
    }
    KGRotatedOpenPathSpace(const KGRotatedOpenPathSpace<XPathType>& aCopy) :
        KGVolume(aCopy),
        fPath(aCopy.fPath),
        fTopPath(aCopy.fTopPath),
        fBottomPath(aCopy.fBottomPath),
        fSign(aCopy.fSign),
        fRotatedMeshCount(aCopy.fRotatedMeshCount),
        fFlattenedMeshCount(aCopy.fFlattenedMeshCount),
        fFlattenedMeshPower(aCopy.fFlattenedMeshPower)
    {}
    KGRotatedOpenPathSpace(const std::shared_ptr<XPathType>& aPath) :
        KGVolume(),
        fPath(aPath),
        fTopPath(new KGPlanarCircle()),
        fBottomPath(new KGPlanarCircle()),
        fSign(1.),
        fRotatedMeshCount(64),
        fFlattenedMeshCount(8),
        fFlattenedMeshPower(1.)
    {}
    ~KGRotatedOpenPathSpace() override = default;

    static std::string Name()
    {
        return "rotated_" + XPathType::Name() + "_space";
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

    const std::shared_ptr<KGPlanarCircle>& StartPath() const
    {
        return fTopPath;
    }
    const std::shared_ptr<KGPlanarCircle>& EndPath() const
    {
        return fBottomPath;
    }

    void Sign(const double& aSign)
    {
        fSign = aSign / fabs(aSign);
        return;
    }
    const double& Sign() const
    {
        return fSign;
    }

    void RotatedMeshCount(const unsigned int& aCount)
    {
        fRotatedMeshCount = aCount;
        return;
    }
    const unsigned int& RotatedMeshCount() const
    {
        return fRotatedMeshCount;
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
    mutable std::shared_ptr<KGPlanarCircle> fTopPath;
    mutable std::shared_ptr<KGPlanarCircle> fBottomPath;
    mutable double fSign;
    mutable unsigned int fRotatedMeshCount;
    mutable unsigned int fFlattenedMeshCount;
    mutable double fFlattenedMeshPower;

  public:
    void VolumeInitialize(BoundaryContainer& aBoundaryContainer) const override
    {
        fTopPath->X(0.);
        fTopPath->Y(0.);
        fTopPath->Radius(fPath->Start().Y());
        fTopPath->MeshCount(fRotatedMeshCount);

        auto tTop = std::make_shared<KGFlattenedCircleSurface>(fTopPath);
        tTop->Sign(1.);
        tTop->Z(fPath->Start().X());
        tTop->FlattenedMeshCount(fFlattenedMeshCount);
        tTop->FlattenedMeshPower(fFlattenedMeshPower);
        tTop->SetName("top");
        aBoundaryContainer.push_back(tTop);

        auto tJacket = std::make_shared<KGRotatedPathSurface<XPathType>>(fPath);
        tJacket->RotatedMeshCount(fRotatedMeshCount);
        tJacket->SetName("jacket");
        aBoundaryContainer.push_back(tJacket);

        fBottomPath->X(0.);
        fBottomPath->Y(0.);
        fBottomPath->Radius(fPath->End().Y());
        fBottomPath->MeshCount(fRotatedMeshCount);

        auto tBottom = std::make_shared<KGFlattenedCircleSurface>(fBottomPath);
        tBottom->Sign(-1.);
        tBottom->Z(fPath->End().X());
        tBottom->FlattenedMeshCount(fFlattenedMeshCount);
        tBottom->FlattenedMeshPower(fFlattenedMeshPower);
        tBottom->SetName("bottom");
        aBoundaryContainer.push_back(tBottom);

        return;
    }
    void VolumeAccept(KGVisitor* aVisitor) override
    {
        shapemsg_debug("rotated open path volume named <" << GetName() << "> is receiving a visitor" << eom);
        auto* tRotatedOpenPathSpaceVisitor = dynamic_cast<typename KGRotatedOpenPathSpace::Visitor*>(aVisitor);
        if (tRotatedOpenPathSpaceVisitor != nullptr) {
            shapemsg_debug("rotated open path volume named <" << GetName() << "> is accepting a visitor" << eom);
            tRotatedOpenPathSpaceVisitor->VisitRotatedOpenPathSpace(this);
            return;
        }
        KGVolume::VolumeAccept(aVisitor);
        return;
    }
    bool VolumeOutside(const KGeoBag::KThreeVector& aQuery) const override
    {
        KGeoBag::KThreeVector tPoint = VolumePoint(aQuery);
        KGeoBag::KThreeVector tNormal = VolumeNormal(aQuery);

        if (tNormal.Dot(aQuery - tPoint) > 0.) {
            return true;
        }

        return false;
    }
    KGeoBag::KThreeVector VolumePoint(const KGeoBag::KThreeVector& aQuery) const override
    {
        KGFlattenedCircleSurface tTop = KGFlattenedCircleSurface(fTopPath);
        tTop.Sign(1.);
        tTop.Z(fPath->Start().X());
        KGeoBag::KThreeVector tTopPoint = tTop.Point(aQuery);
        double tTopDistanceSquared = (aQuery - tTopPoint).MagnitudeSquared();

        KGFlattenedCircleSurface tBottom = KGFlattenedCircleSurface(fBottomPath);
        tBottom.Sign(-1.);
        tBottom.Z(fPath->End().X());
        KGeoBag::KThreeVector tBottomPoint = tBottom.Point(aQuery);
        double tBottomDistanceSquared = (aQuery - tBottomPoint).MagnitudeSquared();

        KTwoVector tZRPoint = aQuery.ProjectZR();
        double tAngle = aQuery.AzimuthalAngle();

        KTwoVector tJacketZRPoint = fPath->Point(tZRPoint);
        KGeoBag::KThreeVector tJacketPoint(cos(tAngle) * tJacketZRPoint.R(),
                                           sin(tAngle) * tJacketZRPoint.R(),
                                           tJacketZRPoint.Z());
        double tJacketDistanceSquared = (aQuery - tJacketPoint).MagnitudeSquared();

        if (tTopDistanceSquared < tJacketDistanceSquared) {
            if (tTopDistanceSquared < tBottomDistanceSquared) {
                return tTopPoint;
            }
            else {
                return tBottomPoint;
            }
        }
        else if (tBottomDistanceSquared < tJacketDistanceSquared) {
            return tBottomPoint;
        }
        else {
            return tJacketPoint;
        }
    }
    KGeoBag::KThreeVector VolumeNormal(const KGeoBag::KThreeVector& aQuery) const override
    {
        KTwoVector tZRPoint = aQuery.ProjectZR();
        double tAngle = aQuery.AzimuthalAngle();

        KGFlattenedCircleSurface tTop = KGFlattenedCircleSurface(fTopPath);
        tTop.Sign(1.);
        tTop.Z(fPath->Start().X());
        KGeoBag::KThreeVector tTopPoint = tTop.Point(aQuery);
        KGeoBag::KThreeVector tTopNormal = tTop.Normal(aQuery);
        double tTopDistanceSquared = (aQuery - tTopPoint).MagnitudeSquared();

        KGFlattenedCircleSurface tBottom = KGFlattenedCircleSurface(fBottomPath);
        tBottom.Sign(-1.);
        tBottom.Z(fPath->End().X());
        KGeoBag::KThreeVector tBottomPoint = tBottom.Point(aQuery);
        KGeoBag::KThreeVector tBottomNormal = tBottom.Normal(aQuery);
        double tBottomDistanceSquared = (aQuery - tBottomPoint).MagnitudeSquared();

        KTwoVector tJacketZRPoint = fPath->Point(tZRPoint);
        KGeoBag::KThreeVector tJacketPoint(cos(tAngle) * tJacketZRPoint.R(),
                                           sin(tAngle) * tJacketZRPoint.R(),
                                           tJacketZRPoint.Z());
        KTwoVector tJacketZRNormal = fPath->Normal(tZRPoint);
        KGeoBag::KThreeVector tJacketNormal(cos(tAngle) * tJacketZRNormal.R(),
                                            sin(tAngle) * tJacketZRNormal.R(),
                                            tJacketZRNormal.Z());
        double tJacketDistanceSquared = (aQuery - tJacketPoint).MagnitudeSquared();

        KGeoBag::KThreeVector tAveragePoint;
        KGeoBag::KThreeVector tAverageNormal;

        if (tTopDistanceSquared < tBottomDistanceSquared) {
            tAveragePoint = .5 * (tTopPoint + tJacketPoint);
            tAverageNormal = (tTopNormal + tJacketNormal).Unit();

            if ((.5 * (tTopPoint - tJacketPoint).Magnitude() / tAveragePoint.Magnitude()) < 1.e-12) {
                if (tAverageNormal.Dot(aQuery - tAveragePoint) > 0.) {
                    return (1. * (aQuery - tAveragePoint).Unit());
                }
                else {
                    return (-1. * (aQuery - tAveragePoint).Unit());
                }
            }

            if (tTopDistanceSquared < tJacketDistanceSquared) {
                return tTopNormal;
            }
            else {
                return tJacketNormal;
            }
        }
        else {
            tAveragePoint = .5 * (tBottomPoint + tJacketPoint);
            tAverageNormal = (tBottomNormal + tJacketNormal).Unit();

            if ((.5 * (tBottomPoint - tJacketPoint).Magnitude() / tAveragePoint.Magnitude()) < 1.e-12) {
                if (tAverageNormal.Dot(aQuery - tAveragePoint) > 0.) {
                    return (1. * (aQuery - tAveragePoint).Unit());
                }
                else {
                    return (-1. * (aQuery - tAveragePoint).Unit());
                }
            }

            if (tBottomDistanceSquared < tJacketDistanceSquared) {
                return tBottomNormal;
            }
            else {
                return tJacketNormal;
            }
        }
    }

  private:
    static KGPlanarOpenPath* CompilerCheck()
    {
        XPathType* tPath = nullptr;
        return tPath;
    }
};

}  // namespace KGeoBag

#endif
