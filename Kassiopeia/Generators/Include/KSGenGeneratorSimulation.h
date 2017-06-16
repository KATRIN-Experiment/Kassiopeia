#ifndef Kassiopeia_KSGenGeneratorSimulation_h_
#define Kassiopeia_KSGenGeneratorSimulation_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"
#include "KSGenerator.h"
#include "KField.h"
#include "KSReadFileROOT.h"

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
            ;K_SET_GET( std::string, Base );
            ;K_SET_GET( std::string, Path );
            ;K_SET_GET( std::string, PositionX );
            ;K_SET_GET( std::string, PositionY );
            ;K_SET_GET( std::string, PositionZ );
            ;K_SET_GET( std::string, DirectionX );
            ;K_SET_GET( std::string, DirectionY );
            ;K_SET_GET( std::string, DirectionZ );
            ;K_SET_GET( std::string, Energy );
            ;K_SET_GET( std::string, Time );
            ;K_SET_GET( std::string, Terminator );
            ;K_SET_GET( std::string, Generator );
            ;K_SET_GET( std::string, TrackGroupName );
            ;K_SET_GET( std::string, TerminatorName );
            ;K_SET_GET( std::string, GeneratorName );
            ;K_SET_GET( std::string, PositionName );
            ;K_SET_GET( std::string, MomentumName );
            ;K_SET_GET( std::string, KineticEnergyName );
            ;K_SET_GET( std::string, TimeName );
            ;K_SET_GET( std::string, PIDName );

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
