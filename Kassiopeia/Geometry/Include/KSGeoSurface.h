#ifndef Kassiopeia_KSGeoSurface_h_
#define Kassiopeia_KSGeoSurface_h_

#include "KSSurface.h"

#include "KGCore.hh"
using namespace KGeoBag;

namespace Kassiopeia
{

    class KSGeoSpace;

    class KSGeoSurface :
        public KSComponentTemplate< KSGeoSurface, KSSurface >
    {
        public:
            friend class KSGeoSpace;

        public:
            KSGeoSurface();
            KSGeoSurface( const KSGeoSurface& aCopy );
            KSGeoSurface* Clone() const;
            virtual ~KSGeoSurface();

        public:
            void On() const;
            void Off() const;

            KThreeVector Point( const KThreeVector& aPoint ) const;
            KThreeVector Normal( const KThreeVector& aPoint ) const;

        public:
            void AddContent( KGSurface* aSurface );
            void RemoveContent( KGSurface* aSurface );
            vector< KGSurface* > GetContent();

            void AddCommand( KSCommand* anCommand );
            void RemoveCommand( KSCommand* anCommand );

        protected:
            void InitializeComponent();
            void DeinitializeComponent();

        private:
            KSGeoSpace* fParent;

            mutable vector< KGSurface* > fContents;
            mutable vector< KSCommand* > fCommands;
    };

}

#endif
