#ifndef KGPORTHOUSING_DEF
#define KGPORTHOUSING_DEF

#include "KGBoundary.hh"
#include "KGCoordinateTransform.hh"

#include <vector>

namespace KGeoBag
{
class KGPortHousing : public KGBoundary
{
  public:
    KGPortHousing() : fCoordTransform(nullptr) {}
    KGPortHousing(double Amain[3], double Bmain[3], double rmain);

    ~KGPortHousing() override;

    static std::string Name()
    {
        return "port_housing";
    }

    void AddCircularPort(double asub[3], double rsub);
    void AddRectangularPort(double asub[3], double length, double width);

    virtual KGPortHousing* Clone() const;

    virtual void Initialize() const;
    virtual void AreaInitialize() const override
    {
        Initialize();
    }

    bool ContainsPoint(const double* P) const;
    double DistanceTo(const double* P, double* P_in = nullptr, double* P_norm = nullptr) const;

    void SetAMain(double d[3])
    {
        for (int i = 0; i < 3; i++)
            fAMain[i] = d[i];
    }
    void SetBMain(double d[3])
    {
        for (int i = 0; i < 3; i++)
            fBMain[i] = d[i];
    }
    void SetRMain(double d)
    {
        fRMain = d;
    }
    void SetNumDiscMain(int i) const
    {
        fNumDiscMain = i;
    }
    void SetPolyMain(int i) const
    {
        fPolyMain = i;
    }

    const double* GetAMain() const
    {
        return fAMain;
    }
    const double* GetBMain() const
    {
        return fBMain;
    }
    double GetAMain(int i) const
    {
        return (i < 3) ? fAMain[i] : 0;
    }
    double GetBMain(int i) const
    {
        return (i < 3) ? fBMain[i] : 0;
    }
    double GetRMain() const
    {
        return fRMain;
    }
    int GetNumDiscMain() const
    {
        return fNumDiscMain;
    }
    int GetPolyMain() const
    {
        return fPolyMain;
    }

    const KGCoordinateTransform* GetCoordinateTransform() const
    {
        return const_cast<const KGCoordinateTransform*>(fCoordTransform);
    }

    class Port
    {
      public:
        Port() {}
        Port(KGPortHousing* portHousing) : fPortHousing(portHousing) {}

        virtual ~Port() {}

        virtual Port* Clone(KGPortHousing*) const = 0;

        virtual void Initialize() {}

        KGPortHousing* GetPortHousing() const
        {
            return fPortHousing;
        }
        void SetPortHousing(KGPortHousing* p)
        {
            fPortHousing = p;
        }

        virtual void ComputeLocalFrame(double* cen, double* x, double* y, double* z) const = 0;

        virtual double GetBoxLength() const = 0;
        virtual double GetBoxWidth() const = 0;

        virtual bool ContainsPoint(const double* P) const = 0;
        virtual double DistanceTo(const double* P, double* P_in, double* P_norm) const = 0;

        const KGCoordinateTransform* GetCoordinateTransform() const
        {
            return const_cast<const KGCoordinateTransform*>(fCoordTransform);
        }

      protected:
        // pointer to port that owns this valve
        KGPortHousing* fPortHousing;
        // This could be changed to be handled by an external library
        KGCoordinateTransform* fCoordTransform;

        // local origin and x,y,z unit vectors in the global frame
        double fCen[3];
        double fX_loc[3];
        double fY_loc[3];
        double fZ_loc[3];
        double fNorm[3];
    };

    class RectangularPort : public KGPortHousing::Port
    {
      public:
        RectangularPort() {}
        RectangularPort(KGPortHousing* portHousing, double asub[3], double length, double width);

        ~RectangularPort() override;

        RectangularPort* Clone(KGPortHousing*) const override;

        void Initialize() override;

        bool ContainsPoint(const double* P) const override;
        double DistanceTo(const double* P, double* P_in = nullptr, double* P_norm = nullptr) const override;

        void ComputeLocalFrame(double* cen, double* x, double* y, double* z) const override;

        // intrinsic characteristics of the port
        void SetASub(double d[3])
        {
            for (int i = 0; i < 3; i++)
                fASub[i] = d[i];
        }
        void SetLength(double d)
        {
            fLength = d;
        }
        void SetWidth(double d)
        {
            fWidth = d;
        }

        double GetASub(int i) const
        {
            return (i < 3) ? fASub[i] : 0;
        }
        double GetLength() const
        {
            return fLength;
        }
        double GetWidth() const
        {
            return fWidth;
        }
        double GetPortLength() const
        {
            return fPortLength;
        }

        // discretization parameters
        void SetXDisc(int i) const
        {
            fXDisc = i;
        }
        void SetNumDiscSub(int i) const
        {
            fNumDiscSub = i;
        }
        void SetLengthDisc(int i) const
        {
            fLengthDisc = i;
        }
        void SetWidthDisc(int i) const
        {
            fWidthDisc = i;
        }
        void SetBoxLength(double d) const
        {
            fBoxLength = d;
        }
        void SetBoxWidth(double d) const
        {
            fBoxWidth = d;
        }

        int GetXDisc() const
        {
            return fXDisc;
        }
        int GetNumDiscSub() const
        {
            return fNumDiscSub;
        }
        int GetLengthDisc() const
        {
            return fLengthDisc;
        }
        int GetWidthDisc() const
        {
            return fWidthDisc;
        }
        double GetBoxLength() const override
        {
            return fBoxLength;
        }
        double GetBoxWidth() const override
        {
            return fBoxWidth;
        }

      private:
        // point describing the middle of the free edge of the port
        double fASub[3];
        // dimension of the port transverse to main cylinder
        double fLength;
        // dimension of the port parallel to the main cylinder
        double fWidth;

        // parameter for discretizing the box surrounding the
        // hole in the main cylinder
        mutable int fXDisc;

        // Determines the number of subelements created from the
        // port during discretization
        mutable int fNumDiscSub;

        // the # of discretizations along fLength
        mutable int fLengthDisc;
        // the # of discretizations along fWidth
        mutable int fWidthDisc;

        // trans. dim. of box surrounding the hole
        mutable double fBoxLength;
        // long. dim. of box surrounding the hole
        mutable double fBoxWidth;

        // Length of the port from the center of the housing
        double fPortLength;
    };

    class CircularPort : public KGPortHousing::Port
    {
      public:
        CircularPort() {}
        CircularPort(KGPortHousing* portHousing, double asub[3], double rsub);

        ~CircularPort() override;

        CircularPort* Clone(KGPortHousing*) const override;

        void Initialize() override;

        bool ContainsPoint(const double* P) const override;
        double DistanceTo(const double* P, double* P_in = nullptr, double* P_norm = nullptr) const override;

        void ComputeLocalFrame(double* cen, double* x, double* y, double* z) const override;

        // intrinsic characteristics of the port:
        void SetASub(double d[3])
        {
            for (int i = 0; i < 3; i++)
                fASub[i] = d[i];
        }
        void SetRSub(double d)
        {
            fRSub = d;
        }

        double GetASub(int i) const
        {
            return (i < 3) ? fASub[i] : 0;
        }
        double GetRSub() const
        {
            return fRSub;
        }
        double GetLength() const
        {
            return fLength;
        }
        double GetNorm(int i) const
        {
            return (i < 3) ? fNorm[i] : 0;
        }

        // discretization parameters
        void SetXDisc(int i) const
        {
            fXDisc = i;
        }
        void SetNumDiscSub(int i) const
        {
            fNumDiscSub = i;
        }
        void SetPolySub(int i) const
        {
            fPolySub = i;
        }
        void SetBoxLength(double d) const
        {
            fBoxLength = d;
        }

        int GetXDisc() const
        {
            return fXDisc;
        }
        int GetNumDiscSub() const
        {
            return fNumDiscSub;
        }
        int GetPolySub() const
        {
            return fPolySub;
        }
        double GetBoxLength() const override
        {
            return fBoxLength;
        }
        double GetBoxWidth() const override
        {
            return fBoxLength;
        }

      private:
        // point describing the free end of the subordinate cyl.
        double fASub[3];
        // radius of the subordinate cylinder
        double fRSub;

        // Length of the port from the center of the housing
        double fLength;
        // square of fLength (no need to continuously recompute!)
        double fLengthSq;
        // unit vector (global coords) pointing from fASub to fCen
        double fNorm[3];

        // parameter for discretizing the box surrounding the hole in the main
        // cylinder
        mutable int fXDisc;

        // Determines the number of subelements created from the subordinate
        // cylinder during discretization
        mutable int fNumDiscSub;

        // the # of sides of the polygon formed by the cross-section of the
        // subordinate cylinder when it is discretized
        mutable int fPolySub;

        // Length of box surrounding the hole in the main cyl.
        mutable double fBoxLength;
    };

    void AddPort(KGPortHousing::Port*);

    unsigned int GetNPorts() const
    {
        return fPorts.size();
    }
    const KGPortHousing::Port* GetPort(int i) const
    {
        return const_cast<const KGPortHousing::Port*>(fPorts.at(i));
    }

  private:
    // point describing one end of the main cylinder
    double fAMain[3];
    // point describing the other end of the main cylinder
    double fBMain[3];
    // radius of the main cylinder
    double fRMain;
    mutable double fD[3];
    mutable double fLength;
    mutable double fLengthSq;
    mutable double fRSq;
    mutable double fNorm[3];

    // Determines the number of subelements created from the main cylinder
    // during discretization
    mutable int fNumDiscMain;

    // the # of sides of the polygon formed by the cross-section of the main
    // cyl. when it is discretized
    mutable int fPolyMain;

    // vector of port valves associated with this port
    std::vector<KGPortHousing::Port*> fPorts;

    // This could be changed to be handled by an external library
    mutable KGCoordinateTransform* fCoordTransform;
};

}  // namespace KGeoBag

#endif /* PORTHOUSING_DEF */
