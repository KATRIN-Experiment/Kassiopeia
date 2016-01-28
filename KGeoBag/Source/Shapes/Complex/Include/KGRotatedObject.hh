#ifndef KGROTATEDOBJECT_DEF
#define KGROTATEDOBJECT_DEF

#include <stddef.h>
#include <vector>
#include <cmath>
#include <string>

namespace KGeoBag
{

  class KGRotatedObject
  {
  public:
    KGRotatedObject() : fNPolyBegin(0),
			fNPolyEnd(0),
			fNSegments(0),
			fDiscretizationPower(2.) {}

    KGRotatedObject(unsigned int nPolyBegin,
		    unsigned int nPolyEnd) : fNPolyBegin(nPolyBegin),
					     fNPolyEnd(nPolyEnd),
					     fNSegments(0),
					     fDiscretizationPower(2.) {}
    virtual ~KGRotatedObject();

    static std::string Name() { return "rotated_object"; }

    virtual void Initialize() const {}

    virtual KGRotatedObject* Clone() const;

    bool ContainsPoint(const double* P) const;
    double DistanceTo(const double* P,
		      double* P_in=NULL,
		      double* P_norm=NULL) const;
    double Area() const;
    double Volume() const;

    void AddLine(const double p1[2],const double p2[2]);
    void AddArc(const double p1[2],const double p2[2],const double radius,const bool positiveOrientation = true);

    unsigned int GetNPolyBegin() const {return fNPolyBegin;}
    unsigned int GetNPolyEnd() const { return fNPolyEnd;}
    double GetDiscretizationPower() const { return fDiscretizationPower;}

    void SetNPolyBegin(unsigned int i) { fNPolyBegin = i; }
    void SetNPolyEnd(unsigned int i) { fNPolyEnd = i; }
    void SetDiscretizationPower(double d) { fDiscretizationPower = d; }

    double GetStartPoint(unsigned int i) const { return (i < 2 ? fP1[i] : 0.); }
    double GetEndPoint(unsigned int i) const { return (i < 2 ? fP2[i] : 0.); }

    class Line
    {
    public:
      Line() {}
      Line(KGRotatedObject* rO,
           const double    p1[2],
           const double    p2[2]);

      virtual ~Line() {}

      virtual void Initialize() const;

      virtual Line* Clone(KGRotatedObject* rS) const;

      virtual bool ContainsPoint(const double* P) const;
      virtual double DistanceTo(const double* P,
				double* P_in=NULL,
				double* P_norm=NULL) const;

      double Area() const { return fArea; }
      double Volume() const { return fVolume; }

      void  SetP1(double d[2]) { for (int i=0;i<2;i++) fP1[i]=d[i]; }
      void  SetP1(unsigned int i,double d) { if (i<2) fP1[i] = d; }
      void  SetP2(double d[2]) { for (int i=0;i<2;i++) fP2[i]=d[i]; }
      void  SetP2(unsigned int i,double d) { if (i<2) fP2[i] = d; }
      void  SetNPolyBegin(unsigned int i)    { fNPolyBegin = i; }
      void  SetNPolyEnd(unsigned int i)      { fNPolyEnd = i; }
      void  SetOrder(unsigned int i)         { fOrder = i; }

      double GetP1(unsigned int i) const { return (i<2 ? fP1[i] : 0); }
      double GetP2(unsigned int i) const { return (i<2 ? fP2[i] : 0); }
      unsigned int   GetNPolyBegin() const { return fNPolyBegin; }
      unsigned int   GetNPolyEnd()   const { return fNPolyEnd; }
      unsigned int   GetOrder()      const { return fOrder; }

      double GetAlpha() const { return fAlpha; }
      double GetTheta() const { return fTheta; }
      bool OpensUp() const { return fOpensUp; }
      double GetZIntercept() const { return fZIntercept; }
      double GetUnrolledBoundingBox(unsigned int i) const
      { return (i < 4 ? fUnrolledBoundingBox[i] : 0.); }
      double GetUnrolledRadius1Squared() const
      { return fUnrolledRadius1Squared; }
      double GetUnrolledRadius2Squared() const
      { return fUnrolledRadius2Squared; }

      KGRotatedObject* GetRotated() const { return fRotated; }
      void SetRotated(KGRotatedObject* r) { fRotated = r; }

      virtual double GetLength() const { return fLength; }

      static bool comp(const KGRotatedObject::Line* left, const KGRotatedObject::Line* right)
      { return (left->GetOrder()<right->GetOrder()) ? true : false; }

      virtual bool IsArc() const { return false; }

    protected:

      // for a Rotated, this number represents the position of the segment
      // when traveling from KGRotatedObject::fP1 to KGRotatedObject::fP2
      unsigned int fOrder;

      // the (x,z) coordinate of the start point (as seen by on a plane that
      // intersects the y=0 axis)
      double fP1[2];
      // the (x,z) coordinate of the end point (as seen by on a plane that
      // intersects the y=0 axis)
      double fP2[2];

      // discretization number about the z-axis for the first opening
      unsigned int fNPolyBegin;
      // discretization number about the z-axis for the second opening
      unsigned int fNPolyEnd;

      // half the opening angle
      mutable double fAlpha;
      // the angle subtended by the unrolled conic section
      mutable double fTheta;
      // min & max x & y of box that obunds unrolled conic section
      mutable double fUnrolledBoundingBox[4];
      // the radius of the outer boundary of the unrolled conic section
      mutable double fUnrolledRadius2;
      // the square of the radius of the outer boundary of the unrolled conic
      // section
      mutable double fUnrolledRadius2Squared;
      // the squared length of the generating line
      mutable double fLengthSquared;
      // the length of the generating line
      mutable double fLength;
      // the radius of the inner boundary of the unrolled conic section
      mutable double fUnrolledRadius1;
      // the square of the radius of the inner boundary of the unrolled conic
      // section
      mutable double fUnrolledRadius1Squared;
      // the z-position where the generating line reaches r=0
      mutable double fZIntercept;
      // bool that states whether the conic section opens up in z
      mutable bool   fOpensUp;
      // the volume of the conic section
      mutable double fVolume;
      // the area of the sheath of the conic section
      mutable double fArea;

      // pointer to container class
      KGRotatedObject* fRotated;
    };

    class Arc : public KGRotatedObject::Line
    {
    public:
      Arc() {}
      Arc(KGRotatedObject* aRotated,
          const double    p1[2],
          const double    p2[2],
          const double    radius,
          const bool      positiveOrientation=true);
      Arc(const KGRotatedObject::Arc& rA);

      virtual ~Arc() {}

      virtual void Initialize() const;

      virtual Arc* Clone(KGRotatedObject* rO) const;

      virtual bool ContainsPoint(const double* P) const;
      virtual double DistanceTo(const double* P,
				double* P_in=NULL,
				double* P_norm=NULL) const;

      void FindCenter() const;

      double GetLength() const;

      double NormalizeAngle(double angle) const;

      double GetRadius(double z) const;

      void SetRadius(double d)         { fRadius = d; }
      void SetOrientation(bool choice) { fPositiveOrientation = choice; }

      double GetRadius()      const { return fRadius; }
      bool   GetOrientation() const { return fPositiveOrientation; }

      double GetCenter(unsigned int i) const { return (i<2 ? fCenter[i] : 0.); }
      double GetPhiStart() const { return fPhiStart; }
      double GetPhiEnd() const { return fPhiEnd; }
      double GetPhiMid() const { return fPhiMid; }

      double GetRMax() const { return fRMax; }

      bool IsArc() const { return true; }

    private:

      bool AngleIsWithinRange(double phi_test,
			      double phi_min,
			      double phi_max,
			      bool   positiveOrientation) const;

      // the radius of the arc
      mutable double fRadius;
      // the center of the circle from which the arc is formed
      mutable double fCenter[2];

      // starting phi corresponding to fP1
      mutable double fPhiStart;
      // ending phi corresponding to fP2
      mutable double fPhiEnd;
      mutable double fPhiMid;
      // phi that splits distance-finding regions
      mutable double fPhiBoundary;

      // maximum radial value for surface of revolution
      mutable double fRMax;

      // reverses the concavity of the arc
      bool fPositiveOrientation;
    };

    void AddSegment(KGRotatedObject::Line*);

    unsigned int GetNSegments() const { return fSegments.size(); }

    const KGRotatedObject::Line* GetSegment(unsigned int i) const { return fSegments.at(i); }

  private:
    // discretization number about the z-axis for the first opening
    unsigned int fNPolyBegin;
    // discretization number about the z-axis for the second opening
    unsigned int fNPolyEnd; 

    // # of segments that comprise the 2-D image
    unsigned int fNSegments;

    // list of segments that comprise the 2-D image
    std::vector< KGRotatedObject::Line* > fSegments;

    // the (x,z) coordinate of the start point (as seen by on a plane that
    // intersects the y=0 axis)
    double fP1[2];
    // the (x,z) coordinate of the end point (as seen by on a plane that
    // intersects the y=0 axis)
    double fP2[2];

    double fDiscretizationPower;

  };

}

#endif
