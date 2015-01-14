#ifndef KGCORE_HH_
#error "do not include KGExtendedSpace.hh directly; include KGCore.hh instead."
#else

namespace KGeoBag
{

    template< class XExtension >
    class KGExtendedSpace :
        public KTagged,
        public KGExtensibleSpace,
        public XExtension::Space
    {
        public:
            friend class KGExtendedSurface< XExtension > ;

        public:
            KGExtendedSpace( KGSpace* aSpace );
            KGExtendedSpace( KGSpace* aSpace, const typename XExtension::Space& );
            virtual ~KGExtendedSpace();

        private:
            KGExtendedSpace();
            KGExtendedSpace( const KGExtendedSpace& );

            //********
            //clonable
            //********

        protected:
            KGExtensibleSpace* Clone( KGSpace* aParent = NULL ) const;

            //*********
            //visitable
            //*********

        public:
            class Visitor
            {
                public:
                    Visitor();
                    virtual ~Visitor();

                    virtual void VisitExtendedSpace( KGExtendedSpace< XExtension >* ) = 0;
            };

            void Accept( KGVisitor* aVisitor );

            //**********
            //extensible
            //**********

        public:
            template< class XOtherExtension >
            bool HasExtension() const;

            template< class XOtherExtension >
            const KGExtendedSpace< XOtherExtension >* AsExtension() const;

            template< class XOtherExtension >
            KGExtendedSpace< XOtherExtension >* AsExtension();

            template< class XOtherExtension >
            KGExtendedSpace< XOtherExtension >* MakeExtension();

            const KGSpace* AsBase() const { return fSpace; }
            KGSpace* AsBase() { return fSpace; }

        private:
            KGSpace* fSpace;

            //************
            //structurable
            //************

        public:
            void Orphan();
            void AddBoundary( KGExtendedSurface< XExtension >* aBoundary );
            void AddChildSurface( KGExtendedSurface< XExtension >* aSurface );
            void AddChildSpace( KGExtendedSpace< XExtension >* aSpace );

            const KGExtendedSpace< XExtension >* GetParent() const;
            const vector< KGExtendedSurface< XExtension >* >* GetBoundaries() const;
            const vector< KGExtendedSurface< XExtension >* >* GetChildSurfaces() const;
            const vector< KGExtendedSpace< XExtension >* >* GetChildSpaces() const;

        private:
            KGExtendedSpace< XExtension >* fParent;
            vector< KGExtendedSurface< XExtension >* > fBoundaries;
            vector< KGExtendedSurface< XExtension >* > fChildSurfaces;
            vector< KGExtendedSpace< XExtension >* > fChildSpaces;

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
            bool Outside( const KThreeVector& aPoint ) const;

    };

}

#endif
