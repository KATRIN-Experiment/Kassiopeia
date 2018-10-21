#ifndef KSURFACE_DEF
#define KSURFACE_DEF

#include <iostream>
#include <sstream>

#include "KSurfaceID.hh"
#include "KSurfacePrimitive.hh"
#include "KSurfaceVisitors.hh"

namespace KEMField
{

/**
* @class KSurface
*
* @brief A policy-based class for defining BEM surfaces.
*
* KSurface is a description of the fundamental element of the BEM.  It is a
* policy-based class (see "Modern C++ Design: Generic Programming and Design
* Patterns Applied" by Andrei Alexandrescu, 2001) designed to separate surface
* elements into three policies: Shape (describing the physical characteristics
* of the surface), Boundary (describing its boundary condition on the PDE of
* interest), and Basis (describing the number of degrees of freedom and value
* type intrinsic to the PDE of interest).
*
* @author T.J. Corona
*/

  template <class BasisPolicy,
	    class BoundaryPolicy,
	    class ShapePolicy>
  class KSurface : public KSurfacePrimitive,
		   public BasisPolicy,
		   public KBoundaryType<BasisPolicy,BoundaryPolicy>,
		   public ShapePolicy
  {
  public:
    typedef BasisPolicy Basis;
    typedef KBoundaryType<BasisPolicy,BoundaryPolicy> Boundary;
    typedef ShapePolicy Shape;

    KSurface() : KSurfacePrimitive(),Basis(),Boundary(),Shape() {}
    KSurface(const Basis& basis,const Boundary& boundary,const Shape& shape) :
      KSurfacePrimitive(),Basis(basis), Boundary(boundary), Shape(shape) {}
    ~KSurface() {}

    KSurface<BasisPolicy,BoundaryPolicy,ShapePolicy>* Clone() const
    { return new KSurface<BasisPolicy,BoundaryPolicy,ShapePolicy>(*this); }

    Basis* GetBasis() { return this; }
    Boundary* GetBoundary() { return this; }
    Shape* GetShape() { return this; }

    std::string GetName() const { return Name(); }
    KSurfaceID& GetID() const { return ID(); }

    static std::string Name()
    {
      std::stringstream s;
      s<<Basis::Name()<<"_"<<Boundary::Name()<<"_"<<Shape::Name();
      return s.str();
    }

    static KSurfaceID& ID() { return fID; }

    void Accept(KBasisVisitor& visitor) { visitor.Visit(*this); }
    void Accept(KBoundaryVisitor& visitor) { visitor.Visit(*this); }
    void Accept(KShapeVisitor& visitor) { visitor.Visit(*this); }

  private:
    static KSurfaceID fID;
  };

  template <typename BasisPolicy,
	    typename BoundaryPolicy,
	    typename ShapePolicy>
  KSurfaceID KSurface<BasisPolicy,BoundaryPolicy,ShapePolicy>::fID = KSurfaceID(IndexOf<KBasisTypes,BasisPolicy>::value,IndexOf<KBoundaryTypes,BoundaryPolicy>::value,IndexOf<KShapeTypes,ShapePolicy>::value);

  template <typename BasisPolicy,
	    typename BoundaryPolicy,
	    typename ShapePolicy,
  	    typename Stream>
  Stream& operator>>(Stream& s,KSurface<BasisPolicy,BoundaryPolicy,ShapePolicy>& e)
  {
    typedef KSurface<BasisPolicy,BoundaryPolicy,ShapePolicy> Surface;
    s.PreStreamInAction(e);
    s >> static_cast<typename Surface::Shape&>(e)
      >> static_cast<typename Surface::Boundary&>(e)
      >> static_cast<typename Surface::Basis&>(e);
    s.PostStreamInAction(e);
    return s;
  }

  template <typename BasisPolicy,
	    typename BoundaryPolicy,
	    typename ShapePolicy,
  	    typename Stream>
  Stream& operator<<(Stream& s,const KSurface<BasisPolicy,BoundaryPolicy,ShapePolicy>& e)
  {
    typedef KSurface<BasisPolicy,BoundaryPolicy,ShapePolicy> Surface;
    s.PreStreamOutAction(e);
    s << static_cast<const typename Surface::Shape&>(e)
      << static_cast<const typename Surface::Boundary&>(e)
      << static_cast<const typename Surface::Basis&>(e);
    s.PostStreamOutAction(e);
    return s;
  }
}

#include "KSurfaceAction.hh"

namespace KEMField
{
  template <typename Stream>
  Stream& operator>>(Stream& s,KSurfacePrimitive& sP)
  {
    return KSurfaceStreamer<Stream,false>::StreamSurface(s,sP);
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KSurfacePrimitive& sP)
  {
    return KSurfaceStreamer<Stream,true>::StreamSurface(s,sP);
  }
}

#endif /* KSURFACE_DEF */
