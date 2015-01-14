#include "KFrame.hh"

namespace KGeoBag
{

    KFrame::KFrame() :
        fOrigin( 0., 0., 0. ),
        fXAxis( 1., 0., 0. ),
        fYAxis( 0., 1., 0. ),
        fZAxis( 0., 0., 1. )
    {
    }
    KFrame::KFrame( const KFrame& aFrame ) :
        fOrigin( aFrame.fOrigin ),
        fXAxis( aFrame.fXAxis ),
        fYAxis( aFrame.fYAxis ),
        fZAxis( aFrame.fZAxis )
    {
    }
    KFrame::~KFrame()
    {
    }

    void KFrame::Transform( const KTransformation& aTransformation )
    {
        aTransformation.ApplyDisplacement( fOrigin );
        aTransformation.ApplyRotation( fXAxis );
        aTransformation.ApplyRotation( fYAxis );
        aTransformation.ApplyRotation( fZAxis );
        return;
    }

    void KFrame::SetOrigin( const KThreeVector& anOrigin )
    {
        fOrigin = anOrigin;
        return;
    }
    const KThreeVector& KFrame::GetOrigin() const
    {
        return fOrigin;
    }
    void KFrame::SetXAxis( const KThreeVector& anXAxis )
    {
        fXAxis = anXAxis;
        return;
    }
    const KThreeVector& KFrame::GetXAxis() const
    {
        return fXAxis;
    }
    void KFrame::SetYAxis( const KThreeVector& aYAxis )
    {
        fYAxis = aYAxis;
        return;
    }
    const KThreeVector& KFrame::GetYAxis() const
    {
        return fYAxis;
    }
    void KFrame::SetZAxis( const KThreeVector& aZAxis )
    {
        fZAxis = aZAxis;
        return;
    }
    const KThreeVector& KFrame::GetZAxis() const
    {
        return fZAxis;
    }

}
