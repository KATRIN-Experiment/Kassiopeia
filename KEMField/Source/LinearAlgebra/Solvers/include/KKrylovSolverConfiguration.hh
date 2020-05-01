/*
 * KKrylovSolverConfiguration.hh
 *
 *  Created on: 12 Aug 2015
 *      Author: wolfgang
 */

#ifndef KKRYLOVSOLVERCONFIGURATION_HH_
#define KKRYLOVSOLVERCONFIGURATION_HH_

#include <string>

namespace KEMField
{

class KKrylovSolverConfiguration
{
  public:
    KKrylovSolverConfiguration();
    virtual ~KKrylovSolverConfiguration();

    unsigned int GetIterationsBetweenRestart() const
    {
        return fIterationsBetweenRestart;
    }

    void SetIterationsBetweenRestart(unsigned int iterationsBetweenRestart)
    {
        fIterationsBetweenRestart = iterationsBetweenRestart;
    }

    unsigned int GetMaxIterations() const
    {
        return fMaxIterations;
    }

    void SetMaxIterations(unsigned int maxIterations)
    {
        fMaxIterations = maxIterations;
    }

    const std::string& GetSolverName() const
    {
        return fSolverName;
    }

    void SetSolverName(const std::string& solverName)
    {
        fSolverName = solverName;
    }

    unsigned int GetStepsBetweenCheckpoints() const
    {
        return fStepsBetweenCheckpoints;
    }

    void SetStepsBetweenCheckpoints(unsigned int stepsBetweenCheckpoints)
    {
        fStepsBetweenCheckpoints = stepsBetweenCheckpoints;
    }

    unsigned int GetStepsBetweenTimeChecks() const
    {
        return fStepsBetweenTimeChecks;
    }

    void SetStepsBetweenTimeChecks(unsigned int stepsBetweenTimeChecks)
    {
        fStepsBetweenTimeChecks = stepsBetweenTimeChecks;
    }

    double GetTimeLimitSeconds() const
    {
        return fTimeLimitSeconds;
    }

    void SetTimeLimitSeconds(double timeLimitSeconds)
    {
        fTimeLimitSeconds = timeLimitSeconds;
    }

    double GetTolerance() const
    {
        return fTolerance;
    }

    void SetTolerance(double tolerance)
    {
        fTolerance = tolerance;
    }

    bool IsUseCheckpoints() const
    {
        return fUseCheckpoints;
    }

    void SetUseCheckpoints(bool useCheckpoints)
    {
        fUseCheckpoints = useCheckpoints;
    }

    bool IsUseDisplay() const
    {
        return fUseDisplay;
    }

    void SetUseDisplay(bool useDisplay)
    {
        fUseDisplay = useDisplay;
    }

    std::string GetDisplayName() const
    {
        return fDisplayName;
    }

    void SetDisplayName(std::string displayName)
    {
        fDisplayName = displayName;
    }

    bool IsUsePlot() const
    {
        return fUsePlot;
    }

    void SetUsePlot(bool usePlot)
    {
        fUsePlot = usePlot;
    }

    bool IsUseTimer() const
    {
        return fUseTimer;
    }

    void SetUseTimer(bool useTimer)
    {
        fUseTimer = useTimer;
    }

  private:
    std::string fSolverName;

    double fTolerance;
    unsigned int fMaxIterations;
    unsigned int fIterationsBetweenRestart;

    bool fUseCheckpoints;
    unsigned int fStepsBetweenCheckpoints;

    bool fUseDisplay;
    std::string fDisplayName;
    bool fUsePlot;

    bool fUseTimer;
    double fTimeLimitSeconds;
    unsigned int fStepsBetweenTimeChecks;
};

} /* namespace KEMField */

#endif /* KKRYLOVSOLVERCONFIGURATION_HH_ */
