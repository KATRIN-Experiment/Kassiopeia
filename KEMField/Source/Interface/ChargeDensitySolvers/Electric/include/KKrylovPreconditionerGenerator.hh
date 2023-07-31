/*
 * KKrylovPreconditioner.hh
 *
 *  Created on: 18 Aug 2015
 *      Author: wolfgang
 */

#ifndef KKRYLOVPRECONDITIONERGENERATOR_HH_
#define KKRYLOVPRECONDITIONERGENERATOR_HH_


#include "KBoundaryMatrixGenerator.hh"
#include "KElectrostaticBasis.hh"
#include "KKrylovSolverConfiguration.hh"


namespace KEMField
{

class KKrylovPreconditionerGenerator : public KBoundaryMatrixGenerator<KElectrostaticBasis::ValueType>
{
  public:
    using ValueType = KElectrostaticBasis::ValueType;
    using MatrixGenerator = KBoundaryMatrixGenerator<ValueType>;

    KKrylovPreconditionerGenerator();
    ~KKrylovPreconditionerGenerator() override;

    std::shared_ptr<KSquareMatrix<ValueType>> Build(const KSurfaceContainer& container) const override;

    void SetMatrixGenerator(const std::shared_ptr<MatrixGenerator>& matrixGen);
    std::shared_ptr<const MatrixGenerator> GetMatrixGenerator() const;
    void SetPreconditionerGenerator(const std::shared_ptr<MatrixGenerator>& preconGen);

    void SetIterationsBetweenRestart(unsigned int iterationsBetweenRestart)
    {
        fKrylovConfig.SetIterationsBetweenRestart(iterationsBetweenRestart);
    }

    void SetMaxIterations(unsigned int maxIterations)
    {
        fKrylovConfig.SetMaxIterations(maxIterations);
    }

    void SetSolverName(const std::string& solverName)
    {
        fKrylovConfig.SetSolverName(solverName);
    }

    void SetStepsBetweenCheckpoints(unsigned int stepsBetweenCheckpoints)
    {
        fKrylovConfig.SetStepsBetweenCheckpoints(stepsBetweenCheckpoints);
    }

    void SetStepsBetweenTimeChecks(unsigned int stepsBetweenTimeChecks)
    {
        fKrylovConfig.SetStepsBetweenTimeChecks(stepsBetweenTimeChecks);
    }

    void SetTimeLimitSeconds(double timeLimitSeconds)
    {
        fKrylovConfig.SetTimeLimitSeconds(timeLimitSeconds);
    }

    void SetTolerance(double tolerance)
    {
        fKrylovConfig.SetTolerance(tolerance);
    }

    void SetUseCheckpoints(bool useCheckpoints)
    {
        fKrylovConfig.SetUseCheckpoints(useCheckpoints);
    }

    void SetUseDisplay(bool useDisplay)
    {
        fKrylovConfig.SetUseDisplay(useDisplay);
    }

    void SetUsePlot(bool usePlot)
    {
        fKrylovConfig.SetUsePlot(usePlot);
    }

    void SetUseTimer(bool useTimer)
    {
        fKrylovConfig.SetUseTimer(useTimer);
    }

  private:
    std::shared_ptr<MatrixGenerator> fMatrixGenerator;
    std::shared_ptr<MatrixGenerator> fPreconditionerGenerator;

    KKrylovSolverConfiguration fKrylovConfig;
};

} /* namespace KEMField */

#endif /* KKRYLOVPRECONDITIONERGENERATOR_HH_ */
