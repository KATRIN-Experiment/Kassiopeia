#include "KGMeshWire.hh"

#include <math.h>

namespace KGeoBag
{
    KGMeshWire::KGMeshWire( const KThreeVector& p0, const KThreeVector& p1, const double& diameter ) :
            KGMeshElement(),
            fP0( p0 ),
            fP1( p1 ),
            fDiameter( diameter )
    {
    }
    KGMeshWire::~KGMeshWire()
    {
    }

    double KGMeshWire::Area() const
    {
        return (.5 * M_PI * fDiameter + (fP1 - fP0).Magnitude()) * M_PI * fDiameter;
    }
    double KGMeshWire::Aspect() const
    {
        return ((fP1 - fP0).Magnitude()) / fDiameter;
    }

    void KGMeshWire::Transform( const KTransformation& transform )
    {
        transform.Apply( fP0 );
        transform.Apply( fP1 );
    }
}
