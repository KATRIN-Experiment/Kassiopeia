#ifndef KGEXTRUDEDOBJECT_DEF
#define KGEXTRUDEDOBJECT_DEF

#include "KGBoundary.hh"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace KGeoBag
{
class KGExtrudedObject : public KGBoundary
{
  public:
    KGExtrudedObject() :
        fZMin(0.),
        fZMax(0.),
        fNDisc(0),
        fDiscretizationPower(2.),
        fNInnerSegments(0),
        fNOuterSegments(0),
        fClosedLoops(false),
        fBackwards(false)
    {}
    KGExtrudedObject(double zMin, double zMax, int nDisc, bool closedLoops) :
        fZMin(zMin),
        fZMax(zMax),
        fNDisc(nDisc),
        fDiscretizationPower(2.),
        fNInnerSegments(0),
        fNOuterSegments(0),
        fClosedLoops(closedLoops),
        fBackwards(false)
    {}

    ~KGExtrudedObject() override;

    static std::string Name()
    {
        return "extruded_object";
    }

    virtual void Initialize() const {}
    void AreaInitialize() const override
    {
        Initialize();
    }

    virtual KGExtrudedObject* Clone() const;

    bool ContainsPoint(const double* P) const;
    double DistanceTo(const double* P, double* P_in = nullptr, double* P_norm = nullptr) const;

    void SetZMin(double zmin)
    {
        fZMin = zmin;
    }
    void SetZMax(double zmax)
    {
        fZMax = zmax;
    }
    void SetDiscretizationPower(double d)
    {
        fDiscretizationPower = d;
    }
    void SetNDisc(int ndisc)
    {
        fNDisc = ndisc;
    }
    void Open()
    {
        fClosedLoops = false;
    }
    void Close()
    {
        fClosedLoops = true;
    }
    void Forwards()
    {
        fBackwards = false;
    }
    void Backwards()
    {
        fBackwards = true;
    }

    void AddInnerLine(double p1[2], double p2[2]);
    void AddOuterLine(double p1[2], double p2[2]);
    void AddInnerArc(double p1[2], double p2[2], double radius, bool positiveOrientation = true);
    void AddOuterArc(double p1[2], double p2[2], double radius, bool positiveOrientation = true);

    double GetZMin() const
    {
        return fZMin;
    }
    double GetZMax() const
    {
        return fZMax;
    }
    int GetNDisc() const
    {
        return fNDisc;
    }
    double GetDiscretizationPower() const
    {
        return fDiscretizationPower;
    }
    bool ClosedLoops() const
    {
        return fClosedLoops;
    }
    bool IsBackwards() const
    {
        return fBackwards;
    }

    static double Theta(const double x, const double y);

    class Line
    {
      public:
        Line() = default;
        Line(KGExtrudedObject* eS, const double p1[2], const double p2[2]);

        virtual ~Line() = default;

        virtual Line* Clone(KGExtrudedObject* eO) const;

        virtual void Initialize() const;

        virtual double DistanceTo(const double* P, double* P_in = nullptr, double* P_norm = nullptr) const;

        void SetOrder(int i)
        {
            fOrder = i;
        }
        int GetOrder() const
        {
            return fOrder;
        }

        void SetNDisc(int i)
        {
            fNDisc = i;
        }
        int GetNDisc() const
        {
            return fNDisc;
        }

        void SetP1(double d[2])
        {
            for (int i = 0; i < 2; i++)
                fP1[i] = d[i];
        }
        void SetP2(double d[2])
        {
            for (int i = 0; i < 2; i++)
                fP2[i] = d[i];
        }

        double GetP1(unsigned int i) const
        {
            return (i < 2 ? fP1[i] : 0.);
        }
        double GetP2(unsigned int i) const
        {
            return (i < 2 ? fP2[i] : 0.);
        }

        virtual double GetLength() const
        {
            return fLength;
        }

        virtual bool IsArc()
        {
            return false;
        }

        static bool comp(const KGExtrudedObject::Line* left, const KGExtrudedObject::Line* right)
        {
            return (left->GetOrder() < right->GetOrder()) ? true : false;
        }

        void SetExtruded(KGExtrudedObject* e)
        {
            fExtruded = e;
        }

      protected:
        // for a KGExtrudedObject, this number represents the position of the
        // segment when traveling from KGExtrudedObject::fP1 to
        // KGExtrudedObject::fP2
        int fOrder;

        // the (x,z) coordinate of the start point (as seen on a plane that
        // intersects the z-axis)
        double fP1[2];

        // the (x,z) coordinate of the end point (as seen on a plane that
        // intersects the z-axis)
        double fP2[2];

        // length of the line segment
        mutable double fLength;

        // the number of elements along the line
        int fNDisc;

        // pointer to container class
        KGExtrudedObject* fExtruded;
    };

    class Arc : public Line
    {
      public:
        Arc() = default;
        Arc(KGExtrudedObject* eS, double p1[2], double p2[2], double radius, bool positiveOrientation = true);

        ~Arc() override = default;

        void Initialize() const override;

        Arc* Clone(KGExtrudedObject* eO) const override;

        double DistanceTo(const double* P, double* P_in = nullptr, double* P_norm = nullptr) const override;

        void FindCenter() const;

        void SetRadius(double d)
        {
            fRadius = d;
        }

        double GetCenter(unsigned int i) const
        {
            return (i < 2 ? fCenter[i] : 0.);
        };
        double GetRadius() const
        {
            return fRadius;
        }
        double GetPhiStart() const
        {
            return fPhiStart;
        }
        double GetPhiEnd() const
        {
            return fPhiEnd;
        }

        double GetLength() const override;

        static double NormalizeAngle(double angle);

        double GetAngularSpread() const;

        void IsPositivelyOriented(bool b)
        {
            fPositiveOrientation = b;
        }
        bool IsPositivelyOriented() const
        {
            return fPositiveOrientation;
        }

        bool IsArc() override
        {
            return true;
        }

        static bool AngleIsWithinRange(double phi_test, double phi_min, double phi_max, bool positiveOrientation);

      private:
        void ComputeAngles() const;

        // the radius of the arc
        double fRadius;

        // the center of the circle from which the arc is formed
        mutable double fCenter[2];

        // starting phi corresponding to fP1
        mutable double fPhiStart;
        // ending phi corresponding to fP2
        mutable double fPhiEnd;
        // phi that splits distance-finding regions
        mutable double fPhiBoundary;

        // reverses the concavity of the arc
        bool fPositiveOrientation;
    };

    void AddInnerSegment(KGExtrudedObject::Line*);
    void AddOuterSegment(KGExtrudedObject::Line*);

    unsigned int GetNOuterSegments() const
    {
        return fOuterSegments.size();
    }
    unsigned int GetNInnerSegments() const
    {
        return fInnerSegments.size();
    }

    const KGExtrudedObject::Line* GetOuterSegment(unsigned int i) const
    {
        return fOuterSegments.at(i);
    }
    const KGExtrudedObject::Line* GetInnerSegment(unsigned int i) const
    {
        return fInnerSegments.at(i);
    }

    static bool CompareTheta(std::vector<double> p1, std::vector<double> p2);

    static bool RayIntersectsLineSeg(const std::vector<double>& p0, const std::vector<double>& s1,
                                     const std::vector<double>& s2, std::vector<double>& p_int);

    static bool PointIsInPolygon(std::vector<double>& p1, const std::vector<std::vector<double>>& vertices,
                                 unsigned int vertexStart, unsigned int nVertices);

  protected:
    // Upstream z position of the surface
    double fZMin;
    // Downstream z position of the surface
    double fZMax;
    // Number of discretizations in the z-direction
    int fNDisc;
    // Power of discretization in the z-direction
    double fDiscretizationPower;

    // # of segments that comprise the 2-D image
    int fNInnerSegments;
    // # of segments that comprise the 2-D image
    int fNOuterSegments;

    // parameter to determine whether the inner and outer segments are closed
    // forms
    bool fClosedLoops;

    // test variable to flip order of coordinates
    bool fBackwards;

    // list of segments that comprise the inner boundary of the 2-D image
    std::vector<KGExtrudedObject::Line*> fInnerSegments;

    // list of segments that comprise the outer boundary of the 2-D image
    std::vector<KGExtrudedObject::Line*> fOuterSegments;
};
}  // namespace KGeoBag

#endif /* KGEXTRUDEDOBJECT_DEF */
