#ifndef Kassiopeia_KSGenGeneratorSimulation_h_
#define Kassiopeia_KSGenGeneratorSimulation_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"
#include "KSGenerator.h"
#include "KField.h"
#include "../../Readers/Include/KSReadFileROOT.h"

#include "TFormula.h"

#include <vector>
using std::vector;

#include <string>
using std::string;

namespace Kassiopeia
{

    class KSGenGeneratorSimulation :
        public KSComponentTemplate< KSGenGeneratorSimulation, KSGenerator >
    {
        public:
            KSGenGeneratorSimulation();
            KSGenGeneratorSimulation( const KSGenGeneratorSimulation& aCopy );
            KSGenGeneratorSimulation* Clone() const;
            virtual ~KSGenGeneratorSimulation();

        public:
            virtual void ExecuteGeneration( KSParticleQueue& aPrimaries );

        public:
            ;K_SET_GET( string, Base );
            ;K_SET_GET( string, Path );
            ;K_SET_GET( string, PositionX );
            ;K_SET_GET( string, PositionY );
            ;K_SET_GET( string, PositionZ );
            ;K_SET_GET( string, DirectionX );
            ;K_SET_GET( string, DirectionY );
            ;K_SET_GET( string, DirectionZ );
            ;K_SET_GET( string, Energy );
            ;K_SET_GET( string, Time );
            ;K_SET_GET( string, Terminator );
            ;K_SET_GET( string, Generator );
            ;K_SET_GET( string, TrackGroupName );
            ;K_SET_GET( string, TerminatorName );
            ;K_SET_GET( string, GeneratorName );
            ;K_SET_GET( string, PositionName );
            ;K_SET_GET( string, MomentumName );
            ;K_SET_GET( string, KineticEnergyName );
            ;K_SET_GET( string, TimeName );
            ;K_SET_GET( string, PIDName );

        protected:
            void InitializeComponent();
            void DeinitializeComponent();

            void GenerateParticlesFromFile( KSParticleQueue &aParticleQueue );

        private:
            KRootFile* fRootFile;

            TFormula *fFormulaPositionX;
            TFormula *fFormulaPositionY;
            TFormula *fFormulaPositionZ;
            TFormula *fFormulaDirectionX;
            TFormula *fFormulaDirectionY;
            TFormula *fFormulaDirectionZ;
            TFormula *fFormulaEnergy;
            TFormula *fFormulaTime;
    };

}

#endif
