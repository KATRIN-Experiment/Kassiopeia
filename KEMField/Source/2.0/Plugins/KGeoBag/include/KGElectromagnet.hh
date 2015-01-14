#ifndef KGELECTROMAGNET_DEF
#define KGELECTROMAGNET_DEF

#include "KSurfaceContainer.hh"
#include "KElectromagnetContainer.hh"

using KEMField::KMagnetostaticBasis;

using KEMField::KPosition;
using KEMField::KDirection;
using KEMField::KGradient;
using KEMField::KEMThreeVector;

using KEMField::KLineCurrent;
using KEMField::KCoil;
using KEMField::KSolenoid;

using KEMField::KElectromagnetContainer;

#include "KGCore.hh"

namespace KGeoBag
{
    class KGElectromagnetData
    {
        public:
            KGElectromagnetData(): fCurrent(0.) {}
            KGElectromagnetData(KGSpace*): fCurrent(0.) {}
            KGElectromagnetData(KGSurface*): fCurrent(0.) {}
            KGElectromagnetData(KGSpace*, const KGElectromagnetData& aCopy): fCurrent(aCopy.fCurrent) {}
			KGElectromagnetData(KGSurface*, const KGElectromagnetData& aCopy): fCurrent(aCopy.fCurrent) {}

            virtual ~KGElectromagnetData() {}

            void SetCurrent( double d );
            double GetCurrent() const;

        private:
            double fCurrent;

    };

    class KGElectromagnet
    {
        public:
            typedef KGElectromagnetData Surface;
            typedef KGElectromagnetData Space;
    };

    typedef KGExtendedSurface< KGElectromagnet > KGElectromagnetSurface;
    typedef KGExtendedSpace< KGElectromagnet > KGElectromagnetSpace;

}

#endif /* KGELECTROMAGNETDATA_DEF */
