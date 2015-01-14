#ifndef KGCORE_HH_
#error "do not include KGExtendedSurface.hh directly; include KGCore.hh instead."
#else

namespace KGeoBag
{

    template< class XExtension >
    class KGExtendedSurface :
        public KTagged,
        public XExtension::Surface,
        public KGExtensibleSurface
    {
        public:
            friend class KGExtendedSpace< XExtension >;

        public:
            KGExtendedSurface( KGSurface* aSurface );
            KGExtendedSurface( KGSurface* aSurface, const typename XExtension::Surface& );
            virtual ~KGExtendedSurface();

        private:
            KGExtendedSurface();
            KGExtendedSurface( const KGExtendedSurface& );

            //********
            //clonable
            //********

        protected:
            KGExtensibleSurface* Clone( KGSurface* aParent = NULL ) const;

            //*********
            //visitable
            //*********

        public:
            class Visitor
            {
                public:
                    Visitor();
                    virtual ~Visitor();

                    virtual void VisitExtendedSurface( KGExtendedSurface< XExtension >* ) = 0;
            };

            void Accept( KGVisitor* aVisitor );

            //**********
            //extensible
            //**********

        public:
            template< class XOtherExtension >
            bool HasExtension() const;

            template< class XOtherExtension >
            const KGExtendedSurface< XOtherExtension >* AsExtension() const;

            template< class XOtherExtension >
            KGExtendedSurface< XOtherExtension >* AsExtension();

            template< class XOtherExtension >
            KGExtendedSurface< XOtherExtension >* MakeExtension();

           const KGSurface* AsBase() const { return fSurface; }
           KGSurface* AsBase() { return fSurface; }

        private:
            KGSurface* fSurface;

            //************
            //structurable
            //************

        public:
            void Orphan();

            const KGExtendedSpace< XExtension >* GetParent() const;

        private:
            KGExtendedSpace< XExtension >* fParent;

            //*************
            //transformable
            //*************

        public:
            void Transform( const KTransformation* aTransformation );

            const KThreeVector& GetOrigin() const;
            const KThreeVector& GetXAxis() const;
            const KThreeVector& GetYAxis() const;
            const KThreeVector& GetZAxis() const;

            //*********
            //navigable
            //*********

        public:
            KThreeVector Point( const KThreeVector& aPoint ) const;
            KThreeVector Normal( const KThreeVector& aPoint ) const;
            bool Above( const KThreeVector& aPoint ) const;

    };

}

#endif
