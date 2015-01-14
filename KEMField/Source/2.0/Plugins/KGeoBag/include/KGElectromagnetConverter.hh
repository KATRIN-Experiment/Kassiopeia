#ifndef KGELECTROMAGNETCONVERTER_DEF
#define KGELECTROMAGNETCONVERTER_DEF

#include "KTagged.h"
using katrin::KTag;

#include "KEMThreeMatrix.hh"
using KEMField::KEMThreeMatrix;

#include "KGElectromagnet.hh"

#include "KGRodSpace.hh"
#include "KGCylinderTubeSpace.hh"
#include "KGCylinderSurface.hh"


namespace KGeoBag
{
    class KGElectromagnetConverter :
        virtual public KGVisitor,
        virtual public KGSpace::Visitor,
        virtual public KGSurface::Visitor,
        virtual public KGExtendedSpace< KGElectromagnet >::Visitor,
        virtual public KGExtendedSurface< KGElectromagnet >::Visitor,
        virtual public KGRodSpace::Visitor,
        virtual public KGCylinderTubeSpace::Visitor,
        virtual public KGCylinderSurface::Visitor
    {
        public:
            KGElectromagnetConverter();
            virtual ~KGElectromagnetConverter();

        public:
            void SetElectromagnetContainer( KElectromagnetContainer* aContainer )
            {
                fElectromagnetContainer = aContainer;
                return;
            }

        protected:
            KElectromagnetContainer* fElectromagnetContainer;

        public:
            void SetSystem( const KThreeVector& anOrigin, const KThreeVector& anXAxis, const KThreeVector& aYAxis, const KThreeVector& aZAxis );
            const KThreeVector& GetOrigin() const;
            const KThreeVector& GetXAxis() const;
            const KThreeVector& GetYAxis() const;
            const KThreeVector& GetZAxis() const;

            KEMThreeVector GlobalToInternalPosition( const KThreeVector& aPosition );
            KEMThreeVector GlobalToInternalVector( const KThreeVector& aVector );
            KThreeVector InternalToGlobalPosition( const KEMThreeVector& aVector );
            KThreeVector InternalToGlobalVector( const KEMThreeVector& aVector );
            KThreeMatrix InternalTensorToGlobal( const KGradient& aGradient );

            void VisitSpace( KGSpace* aSpace );
            void VisitSurface( KGSurface* aSurface );

            void VisitExtendedSpace( KGExtendedSpace< KGElectromagnet >* electromagnetSpace );
            void VisitExtendedSurface( KGExtendedSurface< KGElectromagnet >* electromagnetSurface );

            void VisitWrappedSpace( KGRodSpace* rod );
            void VisitCylinderSurface( KGCylinderSurface* cylinder );
            void VisitCylinderTubeSpace( KGCylinderTubeSpace* cylinderTube );

        private:
            void Clear();

            KThreeVector fOrigin;
            KThreeVector fXAxis;
            KThreeVector fYAxis;
            KThreeVector fZAxis;

            KThreeVector fCurrentOrigin;
            KThreeVector fCurrentXAxis;
            KThreeVector fCurrentYAxis;
            KThreeVector fCurrentZAxis;

            KGExtendedSpace< KGElectromagnet >* fCurrentElectromagnetSpace;
            KGExtendedSurface< KGElectromagnet >* fCurrentElectromagnetSurface;
    };

}

#endif /* KGELECTROMAGNETCONVERTER_DEF */
