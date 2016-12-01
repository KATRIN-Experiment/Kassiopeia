#include "KSElectricField.h"

namespace Kassiopeia
{

    KSElectricField::KSElectricField()
    {
    }

    KSElectricField::~KSElectricField()
    {
    }

    void KSElectricField::CalculateGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeMatrix& aGradient )
    {
        double tEpsilon = 1e-10;

        KThreeVector aSamplePointXPlus = aSamplePoint + KThreeVector(tEpsilon, 0.0, 0.0 );
        KThreeVector aSamplePointXMinus = aSamplePoint + KThreeVector(-tEpsilon, 0.0, 0.0 );
        KThreeVector aSamplePointYPlus = aSamplePoint + KThreeVector(0.0, tEpsilon, 0.0 );
        KThreeVector aSamplePointYMinus = aSamplePoint + KThreeVector(0.0, -tEpsilon, 0.0 );
        KThreeVector aSamplePointZPlus = aSamplePoint + KThreeVector(0.0, 0.0, tEpsilon );
        KThreeVector aSamplePointZMinus = aSamplePoint + KThreeVector(0.0, 0.0, -tEpsilon );

        KThreeVector tFieldPointXPlus;
        KThreeVector tFieldPointXMinus;
        KThreeVector tFieldPointYPlus;
        KThreeVector tFieldPointYMinus;
        KThreeVector tFieldPointZPlus;
        KThreeVector tFieldPointZMinus;

        CalculateField( aSamplePointXPlus, aSampleTime, tFieldPointXPlus );
        CalculateField( aSamplePointXMinus, aSampleTime, tFieldPointXMinus );
        CalculateField( aSamplePointYPlus, aSampleTime, tFieldPointYPlus );
        CalculateField( aSamplePointYMinus, aSampleTime, tFieldPointYMinus );
        CalculateField( aSamplePointZPlus, aSampleTime, tFieldPointZPlus );
        CalculateField( aSamplePointZMinus, aSampleTime, tFieldPointZMinus );

        double tdExdx = ( tFieldPointXPlus.X()-tFieldPointXMinus.X() ) / ( 2.0 * tEpsilon );
        double tdExdy = ( tFieldPointYPlus.X()-tFieldPointYMinus.X() ) / ( 2.0 * tEpsilon );
        double tdExdz = ( tFieldPointZPlus.X()-tFieldPointZMinus.X() ) / ( 2.0 * tEpsilon );

        double tdEydx = ( tFieldPointXPlus.Y()-tFieldPointXMinus.Y() ) / ( 2.0 * tEpsilon );
        double tdEydy = ( tFieldPointYPlus.Y()-tFieldPointYMinus.Y() ) / ( 2.0 * tEpsilon );
        double tdEydz = ( tFieldPointZPlus.Y()-tFieldPointZMinus.Y() ) / ( 2.0 * tEpsilon );

        double tdEzdx = ( tFieldPointXPlus.Z()-tFieldPointXMinus.Z() ) / ( 2.0 * tEpsilon );
        double tdEzdy = ( tFieldPointYPlus.Z()-tFieldPointYMinus.Z() ) / ( 2.0 * tEpsilon );
        double tdEzdz = ( tFieldPointZPlus.Z()-tFieldPointZMinus.Z() ) / ( 2.0 * tEpsilon );

        aGradient.SetComponents(tdExdx, tdExdy, tdExdz,
                                tdEydx, tdEydy, tdEydz,
                                tdEzdx, tdEzdy, tdEzdz);

        return;
    }


}
