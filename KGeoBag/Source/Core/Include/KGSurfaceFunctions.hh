#ifndef KGCORE_HH_
#error "do not include KGSurfaceFunctions.hh directly; include KGCore.hh instead."
#else

namespace KGeoBag
{

    //**********
    //extensible
    //**********

    template< class XExtension >
    inline bool KGSurface::HasExtension() const
    {
        KGExtensibleSurface* tExtension;
        KGExtendedSurface< XExtension >* tOtherExtension;
        std::vector< KGExtensibleSurface* >::const_iterator tIt;
        for( tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++ )
        {
            tExtension = *tIt;
            tOtherExtension = dynamic_cast< KGExtendedSurface< XExtension >* >( tExtension );
            if( tOtherExtension != NULL )
            {
                return true;
            }
        }
        return false;
    }

    template< class XExtension >
    inline const KGExtendedSurface< XExtension >* KGSurface::AsExtension() const
    {
        KGExtensibleSurface* tExtension;
        KGExtendedSurface< XExtension >* tOtherExtension;
        std::vector< KGExtensibleSurface* >::const_iterator tIt;
        for( tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++ )
        {
            tExtension = *tIt;
            tOtherExtension = dynamic_cast< KGExtendedSurface< XExtension >* >( tExtension );
            if( tOtherExtension != NULL )
            {
                return tOtherExtension;
            }
        }
        return NULL;
    }

    template< class XExtension >
    inline KGExtendedSurface< XExtension >* KGSurface::AsExtension()
    {
        KGExtensibleSurface* tExtension;
        KGExtendedSurface< XExtension >* tOtherExtension;
        std::vector< KGExtensibleSurface* >::iterator tIt;
        for( tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++ )
        {
            tExtension = *tIt;
            tOtherExtension = dynamic_cast< KGExtendedSurface< XExtension >* >( tExtension );
            if( tOtherExtension != NULL )
            {
                return tOtherExtension;
            }
        }
        return NULL;
    }

    template< class XExtension >
    inline KGExtendedSurface< XExtension >* KGSurface::MakeExtension()
    {
        KGExtensibleSurface* tExtension;
        KGExtendedSurface< XExtension >* tOtherExtension;
        std::vector< KGExtensibleSurface* >::iterator tIt;
        for( tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++ )
        {
            tExtension = *tIt;
            tOtherExtension = dynamic_cast< KGExtendedSurface< XExtension >* >( tExtension );
            if( tOtherExtension != NULL )
            {
                delete tOtherExtension;
                fExtensions.erase( tIt );
                break;
            }
        }
        tOtherExtension = new KGExtendedSurface< XExtension >( this );
        fExtensions.push_back( tOtherExtension );
        return tOtherExtension;
    }

    template< class XExtension >
    inline KGExtendedSurface< XExtension >* KGSurface::MakeExtension( const typename XExtension::Surface& aCopy )
    {
        KGExtensibleSurface* tExtension;
        KGExtendedSurface< XExtension >* tOtherExtension;
        std::vector< KGExtensibleSurface* >::iterator tIt;
        for( tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++ )
        {
            tExtension = *tIt;
            tOtherExtension = dynamic_cast< KGExtendedSurface< XExtension >* >( tExtension );
            if( tOtherExtension != NULL )
            {
                delete tOtherExtension;
                fExtensions.erase( tIt );
                break;
            }
        }
        tOtherExtension = new KGExtendedSurface< XExtension >( this, aCopy );
        fExtensions.push_back( tOtherExtension );
        return tOtherExtension;
    }

}

#endif
