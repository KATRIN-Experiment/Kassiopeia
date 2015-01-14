#ifndef KGCORE_HH_
#error "do not include KGExtendedSurfaceFunctions.hh directly; include KGCore.hh instead."
#else

namespace KGeoBag
{

    template< class XExtension >
    inline KGExtendedSurface< XExtension >::KGExtendedSurface( KGSurface* aSurface ) :
    XExtension::Surface( aSurface ),
    KGExtensibleSurface(),
    fSurface( aSurface ),
    fParent( NULL )
    {
        SetName( fSurface->GetName() );
    }
    template< class XExtension >
    inline KGExtendedSurface< XExtension >::KGExtendedSurface( KGSurface* aSurface, const typename XExtension::Surface& anExtension ) :
    XExtension::Surface( aSurface, anExtension ),
    KGExtensibleSurface(),
    fSurface( aSurface ),
    fParent( NULL )
    {
        SetName( fSurface->GetName() );
    }
    template< class XExtension >
    inline KGExtendedSurface< XExtension >::~KGExtendedSurface()
    {
    }

    //********
    //clonable
    //********

    template< class XExtension >
    KGExtensibleSurface* KGExtendedSurface< XExtension >::Clone( KGSurface* aSurface ) const
    {
        KGExtendedSurface< XExtension >* tClonedSurface = new KGExtendedSurface< XExtension >( aSurface, *this );
        return tClonedSurface;
    }

    //*********
    //visitable
    //*********

    template< class XExtension >
    KGExtendedSurface< XExtension >::Visitor::Visitor()
    {
    }

    template< class XExtension >
    KGExtendedSurface< XExtension >::Visitor::~Visitor()
    {
    }

    template< class XExtension >
    void KGExtendedSurface< XExtension >::Accept( KGVisitor* aVisitor )
    {
        coremsg_debug( "extended surface named <" << GetName() << "> is receiving a visitor" << eom )

        //visit this extension
        typename KGExtendedSurface< XExtension >::Visitor* MyVisitor = dynamic_cast< typename KGExtendedSurface< XExtension >::Visitor* >( aVisitor );
        if( MyVisitor != NULL )
        {
            coremsg_debug( "extended surface named <" << GetName() << "> is accepting a visitor" << eom )
            MyVisitor->VisitExtendedSurface( this );
        }

        return;
    }

    //**********
    //extensible
    //**********

    template< class XExtension >
    template< class XOtherExtension >
    inline bool KGExtendedSurface< XExtension >::HasExtension() const
    {
        return fSurface->HasExtension< XOtherExtension >();
    }

    template< class XExtension >
    template< class XOtherExtension >
    inline const KGExtendedSurface< XOtherExtension >* KGExtendedSurface< XExtension >::AsExtension() const
    {
        return fSurface->AsExtension< XOtherExtension >();
    }

    template< class XExtension >
    template< class XOtherExtension >
    inline KGExtendedSurface< XOtherExtension >* KGExtendedSurface< XExtension >::AsExtension()
    {
        return fSurface->AsExtension< XOtherExtension >();
    }

    template< class XExtension >
    template< class XOtherExtension >
    inline KGExtendedSurface< XOtherExtension >* KGExtendedSurface< XExtension >::MakeExtension()
    {
        return fSurface->MakeExtension< XOtherExtension >();
    }

    //************
    //structurable
    //************

    template< class XExtension >
    void KGExtendedSurface< XExtension >::Orphan()
    {
        if( fParent != NULL )
        {
            typename vector< KGExtendedSurface< XExtension >* >::iterator tIt;
            for( tIt = fParent->fBoundaries.begin(); tIt != fParent->fBoundaries.end(); tIt++ )
            {
                if( (*tIt) == this )
                {
                    fParent->fBoundaries.erase( tIt );
                    fParent = NULL;
                    return;
                }
            }
            for( tIt = fParent->fChildSurfaces.begin(); tIt != fParent->fChildSurfaces.end(); tIt++ )
            {
                if( (*tIt) == this )
                {
                    fParent->fChildSurfaces.erase( tIt );
                    fParent = NULL;
                    return;
                }
            }
        }
    }

    template< class XExtension >
    const KGExtendedSpace< XExtension >* KGExtendedSurface< XExtension >::GetParent() const
    {
        return fParent;
    }

    //*************
    //transformable
    //*************

    template< class XExtension >
    void KGExtendedSurface< XExtension >::Transform( const KTransformation* aTransformation )
    {
        return fSurface->Transform( aTransformation );
    }

    template< class XExtension >
    const KThreeVector& KGExtendedSurface< XExtension >::GetOrigin() const
    {
        return fSurface->GetOrigin();
    }

    template< class XExtension >
    const KThreeVector& KGExtendedSurface< XExtension >::GetXAxis() const
    {
        return fSurface->GetXAxis();
    }

    template< class XExtension >
    const KThreeVector& KGExtendedSurface< XExtension >::GetYAxis() const
    {
        return fSurface->GetYAxis();
    }

    template< class XExtension >
    const KThreeVector& KGExtendedSurface< XExtension >::GetZAxis() const
    {
        return fSurface->GetZAxis();
    }

    //*********
    //navigable
    //*********

    template< class XExtension >
    KThreeVector KGExtendedSurface< XExtension >::Point( const KThreeVector& aPoint ) const
    {
        return fSurface->Point( aPoint );
    }

    template< class XExtension >
    KThreeVector KGExtendedSurface< XExtension >::Normal( const KThreeVector& aPoint ) const
    {
        return fSurface->Normal( aPoint );
    }

    template< class XExtension >
    bool KGExtendedSurface< XExtension >::Above( const KThreeVector& aPoint ) const
    {
        return fSurface->Above( aPoint );
    }

}

#endif
