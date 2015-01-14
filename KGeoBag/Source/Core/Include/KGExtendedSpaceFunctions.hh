#ifndef KGCORE_HH_
#error "do not include KGExtendedSppaceFunctions.hh directly; include KGCore.hh instead."
#else

namespace KGeoBag
{

    template< class XExtension >
    inline KGExtendedSpace< XExtension >::KGExtendedSpace( KGSpace* aSpace ) :
            KGExtensibleSpace(),
            XExtension::Space( aSpace ),
            fSpace( aSpace ),
            fParent( NULL )
    {
        SetName( fSpace->GetName() );
    }
    template< class XExtension >
    inline KGExtendedSpace< XExtension >::KGExtendedSpace( KGSpace* aSpace, const typename XExtension::Space& anExtension ) :
            KGExtensibleSpace(),
            XExtension::Space( aSpace, anExtension ),
            fSpace( aSpace ),
            fParent( NULL )
    {
        SetName( fSpace->GetName() );
    }
    template< class XExtension >
    inline KGExtendedSpace< XExtension >::~KGExtendedSpace()
    {
    }

    //********
    //clonable
    //********

    template< class XExtension >
    KGExtensibleSpace* KGExtendedSpace< XExtension >::Clone( KGSpace* aSpace ) const
    {
        KGExtendedSpace< XExtension >* tClonedSpace = new KGExtendedSpace< XExtension >( aSpace, *this );
        return tClonedSpace;
    }

    //*********
    //visitable
    //*********

    template< class XExtension >
    KGExtendedSpace< XExtension >::Visitor::Visitor()
    {
    }

    template< class XExtension >
    KGExtendedSpace< XExtension >::Visitor::~Visitor()
    {
    }

    template< class XExtension >
    void KGExtendedSpace< XExtension >::Accept( KGVisitor* aVisitor )
    {
        coremsg_debug( "extended space named <" << GetName() << "> is receiving a visitor" << eom )

        //visit this extension
        typename KGExtendedSpace< XExtension >::Visitor * MyVisitor = dynamic_cast< typename KGExtendedSpace< XExtension >::Visitor* >( aVisitor );
        if( MyVisitor != NULL )
        {
            coremsg_debug( "extended space named <" << GetName() << "> is accepting a visitor" << eom )
            MyVisitor->VisitExtendedSpace( this );
        }

        return;
    }

    //**********
    //extensible
    //**********

    template< class XExtension >
    template< class XOtherExtension >
    inline bool KGExtendedSpace< XExtension >::HasExtension() const
    {
        return fSpace->HasExtension< XOtherExtension >();
    }

    template< class XExtension >
    template< class XOtherExtension >
    inline const KGExtendedSpace< XOtherExtension >* KGExtendedSpace< XExtension >::AsExtension() const
    {
        return fSpace->AsExtension< XOtherExtension >();
    }

    template< class XExtension >
    template< class XOtherExtension >
    inline KGExtendedSpace< XOtherExtension >* KGExtendedSpace< XExtension >::AsExtension()
    {
        return fSpace->AsExtension< XOtherExtension >();
    }

    template< class XExtension >
    template< class XOtherExtension >
    inline KGExtendedSpace< XOtherExtension >* KGExtendedSpace< XExtension >::MakeExtension()
    {
        return fSpace->MakeExtension< XOtherExtension >();
    }

    //************
    //structurable
    //************

    template< class XExtension >
    void KGExtendedSpace< XExtension >::Orphan()
    {
        if( fParent != NULL )
        {
            typename vector< KGExtendedSpace< XExtension >* >::iterator tIt;
            for( tIt = fParent->fChildSpaces.begin(); tIt != fParent->fChildSpaces.end(); tIt++ )
            {
                if( (*tIt) == this )
                {
                    fParent->fChildSpaces.erase( tIt );
                    fParent = NULL;
                    return;
                }
            }
        }
        return;
    }

    template< class XExtension >
    void KGExtendedSpace< XExtension >::AddBoundary( KGExtendedSurface< XExtension >* aBoundary )
    {
        aBoundary->Orphan();
        aBoundary->fParent = this;
        this->fBoundaries.push_back( aBoundary );
        return;
    }

    template< class XExtension >
    void KGExtendedSpace< XExtension >::AddChildSurface( KGExtendedSurface< XExtension >* aSurface )
    {
        aSurface->Orphan();
        aSurface->fParent = this;
        this->fChildSurfaces.push_back( aSurface );
        return;
    }

    template< class XExtension >
    void KGExtendedSpace< XExtension >::AddChildSpace( KGExtendedSpace< XExtension >* aSpace )
    {
        aSpace->Orphan();
        aSpace->fParent = this;
        this->fChildSpaces.push_back( aSpace );
        return;
    }

    template< class XExtension >
    const KGExtendedSpace< XExtension >* KGExtendedSpace< XExtension >::GetParent() const
    {
        return fParent;
    }

    template< class XExtension >
    const vector< KGExtendedSurface< XExtension >* >* KGExtendedSpace< XExtension >::GetBoundaries() const
    {
        return &fBoundaries;
    }

    template< class XExtension >
    const vector< KGExtendedSurface< XExtension >* >* KGExtendedSpace< XExtension >::GetChildSurfaces() const
    {
        return &fChildSurfaces;
    }

    template< class XExtension >
    const vector< KGExtendedSpace< XExtension >* >* KGExtendedSpace< XExtension >::GetChildSpaces() const
    {
        return &fChildSpaces;
    }

    //*************
    //transformable
    //*************

    template< class XExtension >
    void KGExtendedSpace< XExtension >::Transform( const KTransformation* aTransformation )
    {
        fSpace->Transform( aTransformation );
    }

    template< class XExtension >
    const KThreeVector& KGExtendedSpace< XExtension >::GetOrigin() const
    {
        return fSpace->GetOrigin();
    }

    template< class XExtension >
    const KThreeVector& KGExtendedSpace< XExtension >::GetXAxis() const
    {
        return fSpace->GetXAxis();
    }

    template< class XExtension >
    const KThreeVector& KGExtendedSpace< XExtension >::GetYAxis() const
    {
        return fSpace->GetYAxis();
    }

    template< class XExtension >
    const KThreeVector& KGExtendedSpace< XExtension >::GetZAxis() const
    {
        return fSpace->GetZAxis();
    }

    //*********
    //navigable
    //*********

    template< class XExtension >
    KThreeVector KGExtendedSpace< XExtension >::Point( const KThreeVector& aPoint ) const
    {
        return fSpace->Point( aPoint );
    }

    template< class XExtension >
    KThreeVector KGExtendedSpace< XExtension >::Normal( const KThreeVector& aPoint ) const
    {
        return fSpace->Normal( aPoint );
    }

    template< class XExtension >
    bool KGExtendedSpace< XExtension >::Outside( const KThreeVector& aPoint ) const
    {
        return fSpace->Outside( aPoint );
    }

}

#endif
