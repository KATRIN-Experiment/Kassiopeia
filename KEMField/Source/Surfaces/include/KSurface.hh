#ifndef KSURFACE_DEF
#define KSURFACE_DEF

#include "KSurfaceID.hh"
#include "KSurfacePrimitive.hh"
#include "KSurfaceVisitors.hh"

#include <iostream>
#include <sstream>

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

template<class BasisPolicy, class BoundaryPolicy, class ShapePolicy>
class KSurface :
    public KSurfacePrimitive,
    public BasisPolicy,
    public KBoundaryType<BasisPolicy, BoundaryPolicy>,
    public ShapePolicy
{
  public:
    using Basis = BasisPolicy;
    using Boundary = KBoundaryType<BasisPolicy, BoundaryPolicy>;
    using Shape = ShapePolicy;

    KSurface() : KSurfacePrimitive(), Basis(), Boundary(), Shape() {}
    KSurface(const Basis& basis, const Boundary& boundary, const Shape& shape) :
        KSurfacePrimitive(),
        Basis(basis),
        Boundary(boundary),
        Shape(shape)
    {}
    ~KSurface() override = default;

    KSurface<BasisPolicy, BoundaryPolicy, ShapePolicy>* Clone() const override
    {
        return new KSurface<BasisPolicy, BoundaryPolicy, ShapePolicy>(*this);
    }

    Basis* GetBasis() override
    {
        return this;
    }
    Boundary* GetBoundary() override
    {
        return this;
    }
    Shape* GetShape() override
    {
        return this;
    }

    std::string GetName() const override
    {
        return Name();
    }
    KSurfaceID& GetID() const override
    {
        return ID();
    }

    static std::string Name()
    {
        std::stringstream s;
        s << Basis::Name() << "_" << Boundary::Name() << "_" << Shape::Name();
        return s.str();
    }

    static KSurfaceID& ID()
    {
        return fID;
    }

    void Accept(KBasisVisitor& visitor) override
    {
        visitor.Visit(*this);
    }
    void Accept(KBoundaryVisitor& visitor) override
    {
        visitor.Visit(*this);
    }
    void Accept(KShapeVisitor& visitor) override
    {
        visitor.Visit(*this);
    }

  private:
    static KSurfaceID fID;
};

template<typename BasisPolicy, typename BoundaryPolicy, typename ShapePolicy>
KSurfaceID
    KSurface<BasisPolicy, BoundaryPolicy, ShapePolicy>::fID = KSurfaceID(IndexOf<KBasisTypes, BasisPolicy>::value,
                                                                         IndexOf<KBoundaryTypes, BoundaryPolicy>::value,
                                                                         IndexOf<KShapeTypes, ShapePolicy>::value);

template<typename BasisPolicy, typename BoundaryPolicy, typename ShapePolicy, typename Stream>
Stream& operator>>(Stream& s, KSurface<BasisPolicy, BoundaryPolicy, ShapePolicy>& e)
{
    using Surface = KSurface<BasisPolicy, BoundaryPolicy, ShapePolicy>;
    s.PreStreamInAction(e);
    s >> static_cast<typename Surface::Shape&>(e) >> static_cast<typename Surface::Boundary&>(e) >>
        static_cast<typename Surface::Basis&>(e);
    s.PostStreamInAction(e);
    return s;
}

template<typename BasisPolicy, typename BoundaryPolicy, typename ShapePolicy, typename Stream>
Stream& operator<<(Stream& s, const KSurface<BasisPolicy, BoundaryPolicy, ShapePolicy>& e)
{
    using Surface = KSurface<BasisPolicy, BoundaryPolicy, ShapePolicy>;
    s.PreStreamOutAction(e);
    s << static_cast<const typename Surface::Shape&>(e) << static_cast<const typename Surface::Boundary&>(e)
      << static_cast<const typename Surface::Basis&>(e);
    s.PostStreamOutAction(e);
    return s;
}
}  // namespace KEMField

#include "KSurfaceAction.hh"

namespace KEMField
{
template<typename Stream> Stream& operator>>(Stream& s, KSurfacePrimitive& sP)
{
    return KSurfaceStreamer<Stream, false>::StreamSurface(s, sP);
}

template<typename Stream> Stream& operator<<(Stream& s, const KSurfacePrimitive& sP)
{
    return KSurfaceStreamer<Stream, true>::StreamSurface(s, sP);
}
}  // namespace KEMField

#endif /* KSURFACE_DEF */
