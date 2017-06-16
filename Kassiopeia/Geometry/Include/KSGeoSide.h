#ifndef Kassiopeia_KSGeoSide_h_
#define Kassiopeia_KSGeoSide_h_

#include "KSSide.h"

#include "KGCore.hh"
using namespace KGeoBag;

namespace Kassiopeia
{

    class KSGeoSpace;

    class KSGeoSide :
        public KSComponentTemplate< KSGeoSide, KSSide >
    {
        public:
            friend class KSGeoSpace;

        public:
            KSGeoSide();
            KSGeoSide( const KSGeoSide& aCopy );
            KSGeoSide* Clone() const;
            virtual ~KSGeoSide();

        public:
            void On() const;
            void Off() const;

            KThreeVector Point( const KThreeVector& aPoint ) const;
            KThreeVector Normal( const KThreeVector& aPoint ) const;

        public:
            void AddContent( KGSurface* aSurface );
            void RemoveContent( KGSurface* aSurface );
            std::vector< KGSurface* > GetContent();

            void AddCommand( KSCommand* anCommand );
            void RemoveCommand( KSCommand* anCommand );

        protected:
            void InitializeComponent();
            void DeinitializeComponent();

        private:
            KSGeoSpace* fOutsideParent;
            KSGeoSpace* fInsideParent;

            mutable std::vector< KGSurface* > fContents;
            mutable std::vector< KSCommand* > fCommands;
    };

}

#endif
