#ifndef KGCONICSECTPORTHOUSING_DEF
#define KGCONICSECTPORTHOUSING_DEF

#include <vector>

#include "KGCoordinateTransform.hh"

namespace KGeoBag
{
  class KGConicSectPortHousing
  {
  public:
    KGConicSectPortHousing() {}
    KGConicSectPortHousing(double zA,
			   double rA,
			   double zB,
			   double rB);

    virtual ~KGConicSectPortHousing();

    static std::string Name() { return "conic_section_port_housing"; }

    virtual KGConicSectPortHousing* Clone() const;

    virtual void Initialize() const;

    bool   ContainsPoint(const double* P) const;
    double DistanceTo(const double* P,double* P_in=NULL,double* P_norm=NULL) const;

    void AddParaxialPort(double asub[3],
			 double rsub);

    void AddOrthogonalPort(double asub[3],
			   double rsub);

    void SetZAMain(double d) { fzAMain = d; }
    void SetZBMain(double d) { fzBMain = d; }
    void SetRAMain(double d) { frAMain = d; }
    void SetRBMain(double d) { frBMain = d; }
    void SetNumDiscMain(int i) const { fNumDiscMain = i; }
    void SetPolyMain(int i) const { fPolyMain = i; }

    double GetZAMain()      const { return fzAMain; }
    double GetZBMain()      const { return fzBMain; }
    double GetRAMain()      const { return frAMain; }
    double GetRBMain()      const { return frBMain; }
    int    GetNumDiscMain() const { return fNumDiscMain; }
    int    GetPolyMain()    const { return fPolyMain; }

    double GetRAlongConicSect(double z) const;
    double GetZAlongConicSect(double r) const;

    double DistanceToConicSect(const double* P) const;
    bool   ContainedByConicSect(const double* P) const;

    void RayConicSectIntersection(const std::vector<double>& p0,
				  const std::vector<double>& n1,
				  std::vector<double>& p_int) const;

    double DistanceBetweenLines(const std::vector<double>& s1,
				const std::vector<double>& s2,
				const std::vector<double>& p1,
				const std::vector<double>& p2) const;

    class Port
    {
    public:
      Port() {}
      Port(KGConicSectPortHousing* portHousing,
	   double asub[3],
	   double r);

      virtual ~Port();

      virtual Port* Clone(KGConicSectPortHousing*) const = 0;

      virtual void Initialize() {}

      virtual bool   ContainsPoint(const double* P) const = 0;
      virtual double DistanceTo(const double* P,double* P_in,double* P_norm) const = 0;

      void     SetASub(double d[3])   { for (int i=0;i<3;i++) fASub[i]=d[i]; }
      void     SetRSub(double d)      { fRSub = d; }

      double GetASub(int i)  const { return (i<3) ? fASub[i] : 0; }
      double GetRSub()         const { return fRSub; }

      void     SetBoxRInner(double d) const { fBoxRInner = d; }
      void     SetBoxROuter(double d) const { fBoxROuter = d; }
      void     SetBoxAngle(double d)  const { fBoxAngle = d; }
      void     SetBoxTheta(double d)  const { fBoxTheta = d; }

      double GetBoxRInner()    const { return fBoxRInner; }
      double GetBoxROuter()    const { return fBoxROuter; }
      double GetBoxAngle()     const { return fBoxAngle; }
      double GetBoxTheta()     const { return fBoxTheta; }

      const KGConicSectPortHousing* GetPortHousing() const { return fPortHousing; }
      void SetPortHousing(KGConicSectPortHousing* p) { fPortHousing = p; }

    protected:

      // pointer to port that owns this valve
      KGConicSectPortHousing* fPortHousing;

      // point describing the free end of the subordinate cyl.
      double fASub[3];
      // radius of the subordinate cylinder
      double fRSub;

      // the bounding box for the port is described as a piece of the conic
      // section, and is uniquely defined by the inner radius, outer radius and
      // angle defining the box
      mutable double fBoxRInner;
      mutable double fBoxROuter;
      mutable double fBoxAngle;
      mutable double fBoxTheta;
    };

    class OrthogonalPort : public KGConicSectPortHousing::Port
    {
    public:
      OrthogonalPort() {}
      OrthogonalPort(KGConicSectPortHousing* portHousing,
		     double asub[3],
		     double rsub);

      virtual ~OrthogonalPort();

      virtual OrthogonalPort* Clone(KGConicSectPortHousing*) const;

      virtual void Initialize();

      bool   ContainsPoint(const double* P) const;
      double DistanceTo(const double* P,double* P_in,double* P_norm) const;

      void     SetXDisc(int i)        const { fXDisc = i; }
      void     SetCylDisc(int i)      const { fCylDisc = i; }
      void     SetAlphaPolySub(int i) const { fAlphaPolySub = i; }
      void     SetPolySub(int i)      const { fPolySub = i; }

      int    GetXDisc()         const { return fXDisc; }
      int    GetCylDisc()       const { return fCylDisc; }
      int    GetAlphaPolySub()  const { return fAlphaPolySub; }
      int    GetPolySub()       const { return fPolySub; }
      double GetCen(unsigned int i)   const { return (i<3 ? fCen[i] : 0); }
      double GetX_loc(unsigned int i) const { return (i<3 ? fX_loc[i] : 0); }
      double GetY_loc(unsigned int i) const { return (i<3 ? fY_loc[i] : 0); }
      double GetZ_loc(unsigned int i) const { return (i<3 ? fZ_loc[i] : 0); }
      double GetLength()        const { return fLength; }

      const KGCoordinateTransform* GetCoordinateTransform() const { return const_cast<const KGCoordinateTransform*>(fCoordTransform); }

    private:
      void ComputeLocalFrame(double *cen,
			     double *x,
			     double *y,
			     double *z);

      // Length of the port from the center of the housing
      double fLength;
      // Length of the port from the axis of the conic section
      double fAugmentedLength;

      // parameter for discretizing the box surrounding the hole in the main
      // cylinder
      mutable int fXDisc;

      // parameter for discretizing the sheath of the cylinder
      mutable int fCylDisc;

      // the # of sides of the curve on the inner and outer edges of the
      // bounding box
      mutable int fAlphaPolySub;

      // the # of sides of the polygon formed by the cross-section of the
      // subordinate cylinder when it is discretized (must be an even number)
      mutable int fPolySub;

      KGCoordinateTransform* fCoordTransform;

      // local origin and x,y,z unit vectors in the global frame
      double fCen[3];
      double fX_loc[3];
      double fY_loc[3];
      double fZ_loc[3];

      // unit vector pointing from the z-axis to the center point
      double fNorm[3];

      // distance beyond which a point must map to the intersection
      double fSafeHeight;
    };

    class ParaxialPort : public KGConicSectPortHousing::Port
    {
    public:
      ParaxialPort() {}
      ParaxialPort(KGConicSectPortHousing* portHousing,
		   double asub[3],
		   double rsub);

      virtual ~ParaxialPort();

      virtual ParaxialPort* Clone(KGConicSectPortHousing*) const;

      virtual void Initialize();

      bool   ContainsPoint(const double* P) const;
      double DistanceTo(const double* P,double* P_in,double* P_norm) const;

      void     SetXDisc(int i)        const { fXDisc = i; }
      void     SetCylDisc(int i)      const { fCylDisc = i; }
      void     SetAlphaPolySub(int i) const { fAlphaPolySub = i; }
      void     SetPolySub(int i)      const { fPolySub = i; }

      int    GetXDisc()        const { return fXDisc; }
      int    GetCylDisc()      const { return fCylDisc; }
      int    GetAlphaPolySub() const { return fAlphaPolySub; }
      int    GetPolySub()      const { return fPolySub; }

      double GetSymmetricLength() const { return fSymmetricLength; }
      double GetAsymmetricLength() const { return fAsymmetricLength; }

      bool IsUpstream() const { return fIsUpstream; }

    private:

      void ComputeNorm();

      // total length of the port
      mutable double fLength;

      // parameter for discretizing the box surrounding the hole in the main
      // cylinder
      mutable int fXDisc;

      // parameter for discretizing the sheath of the cylinder
      mutable int fCylDisc;

      // the # of sides of the curve on the inner and outer edges of the
      // bounding box
      mutable int fAlphaPolySub;

      // the # of sides of the polygon formed by the cross-section of the
      // subordinate cylinder when it is discretized (must be an even number)
      mutable int fPolySub;

      // the lengths of the rotationally symmetric and asymmetric parts of the
      // port
      mutable double fSymmetricLength;
      mutable double fAsymmetricLength;

      // flag to determine in which direction the port points
      bool fIsUpstream;

      // unit vector normal to the cone's surface and coplanar with the port
      // axis
      double fNorm[3];
    };

    void AddPort(KGConicSectPortHousing::Port*);

    unsigned int GetNPorts() const { return fPorts.size(); }
    const KGConicSectPortHousing::Port* GetPort(int i) const { return fPorts.at(i); }

  private:

    // z-coordinate of 1st endpoint of generating line (m)
    double fzAMain;
    // r-coordinate of 1st endpoint of generating line (m)
    double frAMain;
    // z-coordinate of 2nd endpoint of generating line (m)
    double fzBMain;
    // r-coordinate of 2nd endpoint of generating line (m)
    double frBMain;
    double fLength;

    // Determines the number of subelements created from the main conic section
    // during discretization
    mutable int fNumDiscMain;

    // the # of sides of the polygon formed by the cross-section of the main
    // c.s. when it is discretized
    mutable int fPolyMain;

    // vector of port valves associated with this port
    std::vector<KGConicSectPortHousing::Port*> fPorts;
  };
}

#endif /* KGCONICSECTPORTHOUSING_DEF */
