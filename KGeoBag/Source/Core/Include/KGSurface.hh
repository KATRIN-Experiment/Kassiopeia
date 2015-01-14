#ifndef KGCORE_HH_
#error "do not include KGSurface.hh directly; include KGCore.hh instead."
#else

namespace KGeoBag
{

    class KGSurface :
        public KTagged
    {
        public:
            friend class KGSpace;

            class Visitor
            {
                public:
                    Visitor();
                    virtual ~Visitor();

                    virtual void VisitSurface( KGSurface* aSurface ) = 0;
            };

        public:
            KGSurface();
            KGSurface( KGArea* anArea );
            virtual ~KGSurface();

            //**********
            //extensible
            //**********

        public:
            template< class XExtension >
            bool HasExtension() const;

            template< class XExtension >
            const KGExtendedSurface< XExtension >* AsExtension() const;

            template< class XExtension >
            KGExtendedSurface< XExtension >* AsExtension();

            template< class XExtension >
            KGExtendedSurface< XExtension >* MakeExtension();

            template< class XExtension >
            KGExtendedSurface< XExtension >* MakeExtension( const typename XExtension::Surface& aCopy );

        protected:
            vector< KGExtensibleSurface* > fExtensions;

            //************
            //structurable
            //************

        public:
            void Orphan();

            const KGSpace* GetParent() const;

        protected:
            KGSpace* fParent;

            //*************
            //transformable
            //*************

        public:
            void Transform( const KTransformation* aTransformation );

            const KThreeVector& GetOrigin() const;
            const KThreeVector& GetXAxis() const;
            const KThreeVector& GetYAxis() const;
            const KThreeVector& GetZAxis() const;

        protected:
            KThreeVector fOrigin;
            KThreeVector fXAxis;
            KThreeVector fYAxis;
            KThreeVector fZAxis;

            //********
            //clonable
            //********

        public:
            KGSurface* CloneNode() const;

            //*********
            //visitable
            //*********

        public:
            void AcceptNode( KGVisitor* aVisitor );

            //*********
            //navigable
            //*********

        public:
            void Area( const KSmartPointer< KGArea >& anArea );
            const KSmartPointer< KGArea >& Area() const;

            bool Above( const KThreeVector& aPoint ) const;
            KThreeVector Point( const KThreeVector& aPoint ) const;
            KThreeVector Normal( const KThreeVector& aPoint ) const;

        private:
            KSmartPointer< KGArea > fArea;
    };

}

#endif
