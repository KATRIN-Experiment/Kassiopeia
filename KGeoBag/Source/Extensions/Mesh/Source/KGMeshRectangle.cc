#include "KGMeshRectangle.hh"

#include <math.h>

namespace KGeoBag
{
    KGMeshRectangle::KGMeshRectangle( const double& a, const double& b, const KThreeVector& p0, const KThreeVector& n1, const KThreeVector& n2 ) :
            KGMeshElement(),
            fA( a ),
            fB( b ),
            fP0( p0 ),
            fN1( n1 ),
            fN2( n2 )
    {
    }
    KGMeshRectangle::KGMeshRectangle( const KThreeVector& p0, const KThreeVector& p1, const KThreeVector& /*p2*/, const KThreeVector& p3 ) :
            KGMeshElement()
    {
        fP0 = p0;
        fN1 = p1 - p0;
        fA = fN1.Magnitude();
        fN1 = fN1.Unit();
        fN2 = p3 - p0;
        fB = fN2.Magnitude();
        fN2 = fN2.Unit();
    }
    KGMeshRectangle::~KGMeshRectangle()
    {
    }

    double KGMeshRectangle::Area() const
    {
        return fA * fB;
    }
    double KGMeshRectangle::Aspect() const
    {
        if( fA > fB )
        {
            return fA / fB;
        }
        else
        {
            return fB / fA;
        }
    }

    void KGMeshRectangle::Transform( const KTransformation& transform )
    {
        transform.Apply( fP0 );
        transform.ApplyRotation( fN1 );
        transform.ApplyRotation( fN2 );
    }
}
