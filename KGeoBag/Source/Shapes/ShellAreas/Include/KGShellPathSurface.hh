#ifndef KGSHELLPATHSURFACE_HH_
#define KGSHELLPATHSURFACE_HH_

#include "KGArea.hh"
#include "KGPlanarPath.hh"
#include "KGShapeMessage.hh"

#include <memory>

namespace KGeoBag
{

template<class XPathType> class KGShellPathSurface : public KGArea
{
  public:
    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitShellPathSurface(KGShellPathSurface* aShellPathSurface) = 0;
    };

  public:
    KGShellPathSurface() :
        KGArea(),
        fPath(new XPathType()),
        fSign(1.),
        fShellMeshCount(64),
        fShellMeshPower(1.),
        fAngleStart(0),
        fAngleStop(360)
    {
        CompilerCheck();
    }
    KGShellPathSurface(const KGShellPathSurface<XPathType>& aCopy) :
        KGArea(aCopy),
        fPath(aCopy.fPath),
        fSign(1.),
        fShellMeshCount(aCopy.fShellMeshCount),
        fShellMeshPower(aCopy.fShellMeshPower),
        fAngleStart(aCopy.fAngleStart),
        fAngleStop(aCopy.fAngleStop)
    {}
    KGShellPathSurface(std::shared_ptr<XPathType> aPath) :
        KGArea(),
        fPath(aPath),
        fSign(1.),
        fShellMeshCount(64),
        fShellMeshPower(1.),
        fAngleStart(0),
        fAngleStop(360)
    {}
    ~KGShellPathSurface() override = default;

    static std::string Name()
    {
        return "shell_" + XPathType::Name() + "_surface";
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

    void Sign(const double& aSign)
    {
        fSign = aSign / fabs(aSign);
        return;
    }
    const double& Sign() const
    {
        return fSign;
    }

    void ShellMeshCount(const unsigned int& aCount)
    {
        fShellMeshCount = aCount;
    }
    const unsigned int& ShellMeshCount() const
    {
        return fShellMeshCount;
    }

    void ShellMeshPower(const double& aPower)
    {
        fShellMeshPower = aPower;
        return;
    }
    const double& ShellMeshPower() const
    {
        return fShellMeshPower;
    }

    void AngleStart(const double& aAngle)
    {
        fAngleStart = aAngle;
    }
    const double& AngleStart() const
    {
        return fAngleStart;
    }

    void AngleStop(const double& aAngle)
    {
        fAngleStop = aAngle;
    }
    const double& AngleStop() const
    {
        return fAngleStop;
    }

  protected:
    mutable std::shared_ptr<XPathType> fPath;
    mutable double fSign;
    mutable unsigned int fShellMeshCount;
    mutable double fShellMeshPower;
    mutable double fAngleStart;
    mutable double fAngleStop;

  public:
    void AreaInitialize() const override
    {
        return;
    }
    void AreaAccept(KGVisitor* aVisitor) override
    {
        shapemsg_debug("shell path area named <" << GetName() << "> is receiving a visitor" << eom);
        auto* tShellPathSurfaceVisitor = dynamic_cast<typename KGShellPathSurface<XPathType>::Visitor*>(aVisitor);
        if (tShellPathSurfaceVisitor != nullptr) {
            shapemsg_debug("shell path area named <" << GetName() << "> is accepting a visitor" << eom);
            tShellPathSurfaceVisitor->VisitShellPathSurface(this);
        }
        return;
    }
    bool AreaAbove(const KGeoBag::KThreeVector& aPoint) const override
    {
        KTwoVector tZRPoint = aPoint.ProjectZR();
        bool tZRAbove = fPath->Above(tZRPoint);
        if ((tZRAbove == true) && (fSign > 0.)) {
            return true;
        }
        else {
            return false;
        }
    }
    KGeoBag::KThreeVector AreaPoint(const KGeoBag::KThreeVector& aPoint) const override
    {
        NearestPointInfo info = CalculateNearestPointInfo(aPoint);
        KTwoVector tZRNearest = fPath->Point(info.tZRPoint);
        return KGeoBag::KThreeVector(cos(info.tAngleNearest) * tZRNearest.R(),
                                     sin(info.tAngleNearest) * tZRNearest.R(),
                                     tZRNearest.Z());
    }
    KGeoBag::KThreeVector AreaNormal(const KGeoBag::KThreeVector& aPoint) const override
    {
        NearestPointInfo info = CalculateNearestPointInfo(aPoint);
        KTwoVector tZRNormal = fPath->Normal(info.tZRPoint);
        return fSign * KGeoBag::KThreeVector(cos(info.tAngleNearest) * tZRNormal.R(),
                                             sin(info.tAngleNearest) * tZRNormal.R(),
                                             tZRNormal.Z());
    }

  private:
    struct NearestPointInfo
    {
        double tAngleNearest;
        KTwoVector tZRPoint;
    };

    NearestPointInfo CalculateNearestPointInfo(const KGeoBag::KThreeVector& aPoint) const
    {
        NearestPointInfo info;
        double tAnglePoint;
        double tAngleStart = fAngleStart / 180. * M_PI;                  //angle of start edge of shell
        double tAngleClosed = (fAngleStop - fAngleStart) / 180. * M_PI;  //angle width of shell closed
        double tAngleStop = tAngleStart + tAngleClosed;                  //angle of stop edge of shell
        tAnglePoint = tAngleStart + fmod(aPoint.AzimuthalAngle() - tAngleStart,
                                         2 * M_PI);  //angle of query point must be within 2Pi of start edge of shell
        if (tAnglePoint - tAngleStart < 0.)
            tAnglePoint += 2 * M_PI;  //angle of query point must be greater angle of start edge of shell

        if (tAngleStart <= tAnglePoint && tAnglePoint <= tAngleStop)  //if query point lies in angle of closed shell
            info.tAngleNearest = tAnglePoint;                         //choose angle of the query point
        else                                                          //if query point lies in angle of shell gap
        {
            double tAngleDistanceToStop =
                tAnglePoint - tAngleStop;  //calculate angle distance from stop edge of shell to query point
            double tAngleOpen = 2 * M_PI - tAngleClosed;  //calculate width of shell gap
            if (tAngleDistanceToStop < tAngleOpen / 2.)   //choose angle of closer shell edge
                info.tAngleNearest = tAngleStop;
            else
                info.tAngleNearest = tAngleStart;
        }

        info.tZRPoint = aPoint.ProjectZR();  //find (z,r) for query point using r^2=x^2+y^2
        info.tZRPoint.R() *=
            cos(info.tAngleNearest - tAnglePoint);  //project query point on zr-plane at phi of nearest point
        return info;
    }


    static KGPlanarPath* CompilerCheck()
    {
        XPathType* tPath = nullptr;
        return tPath;
    }
};

}  // namespace KGeoBag

#endif
