/*
 * KrylovFactoryFixture.hh
 *
 *  Created on: 13 Aug 2015
 *      Author: wolfgang
 */

#ifndef UNITTEST_KEMFIELD_LINEARALGEBRA_SOLVERS_INCLUDE_KRYLOVFACTORYFIXTURE_HH_
#define UNITTEST_KEMFIELD_LINEARALGEBRA_SOLVERS_INCLUDE_KRYLOVFACTORYFIXTURE_HH_

#include "KEMFieldTest.hh"
#include "KElectrostaticBasis.hh"
#include "KKrylovSolverFactory.hh"
#include "KSimpleSquareMatrix.hh"

class KrylovFactoryFixture : public KEMFieldTest
{
  public:
    KrylovFactoryFixture();
    virtual ~KrylovFactoryFixture();

  protected:
    typedef KEMField::KElectrostaticBasis::ValueType ElectricType;
    typedef KEMField::KSmartPointer<KEMField::KIterativeKrylovSolver<ElectricType>> ElectricSolverPtr;

    typedef KEMField::KSimpleIterativeKrylovSolver<ElectricType, KEMField::KGeneralizedMinimalResidual> ElectricGMRES;

    typedef KEMField::KPreconditionedIterativeKrylovSolver<ElectricType,
                                                           KEMField::KPreconditionedGeneralizedMinimalResidual>
        ElectricPGMRES;

    typedef KEMField::KSimpleIterativeKrylovSolver<ElectricType, KEMField::KBiconjugateGradientStabilized>
        ElectricBiCGSTAB;

    typedef KEMField::KPreconditionedIterativeKrylovSolver<ElectricType,
                                                           KEMField::KPreconditionedBiconjugateGradientStabilized>
        ElectricPBiCGSTAB;

    KEMField::KKrylovSolverConfiguration fConfig;

    KEMField::KSmartPointer<KEMField::KSimpleSquareMatrix<ElectricType>> fA;
    KEMField::KSmartPointer<KEMField::KSimpleSquareMatrix<ElectricType>> fP;
};

#endif /* UNITTEST_KEMFIELD_LINEARALGEBRA_SOLVERS_INCLUDE_KRYLOVFACTORYFIXTURE_HH_ */
