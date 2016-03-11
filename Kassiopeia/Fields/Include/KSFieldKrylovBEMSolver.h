/*
 * KSFieldFastMultipoleBEMSolver.h
 *
 *  Created on: 27.04.2015
 *      Author: gosda
 */

#ifndef KSFIELDFASTMULTIPOLEBEMSOLVER_H_
#define KSFIELDFASTMULTIPOLEBEMSOLVER_H_

#include "KSFieldElectrostatic.h"

namespace KEMField {

            class KrylovBEMSolver :
                public Kassiopeia::KSFieldElectrostatic::BEMSolver
            {
                public:
                    KrylovBEMSolver();
                    virtual ~KrylovBEMSolver();

                    KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration*
                    		GetSolverConfig() {return &fKrylovConfig;}

                    void Initialize( KSurfaceContainer& container );

                private:

                    virtual bool FindSolution( double threshold, KSurfaceContainer& container );

                    KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration
                    	fKrylovConfig;
                    KFMElectrostaticFastMultipoleBoundaryValueSolver* fSolver;
            };

}
#endif /* KSFIELDFASTMULTIPOLEBEMSOLVER_H_ */
