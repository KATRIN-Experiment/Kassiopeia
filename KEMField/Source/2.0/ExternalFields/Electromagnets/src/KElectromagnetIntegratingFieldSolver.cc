#include "KElectromagnetIntegratingFieldSolver.hh"

namespace KEMField
{
  KEMThreeVector KIntegratingFieldSolver<KElectromagnetIntegrator>::VectorPotential(const KPosition& P) const
  {
    VectorPotentialAction action(fContainer,fIntegrator,P);

    KElectromagnetAction<>::ActOnElectromagnets(action);

    return action.GetVectorPotential();
  }

  KEMThreeVector KIntegratingFieldSolver<KElectromagnetIntegrator>::MagneticField(const KPosition& P) const
  {
    MagneticFieldAction action(fContainer,fIntegrator,P);

    KElectromagnetAction<>::ActOnElectromagnets(action);

    return action.GetMagneticField();
  }

  KGradient KIntegratingFieldSolver<KElectromagnetIntegrator>::MagneticFieldGradient(const KPosition& P) const
  {
    KGradient g;
    double epsilon = 1.e-6;
    for (unsigned int i=0;i<3;i++)
    {
      KPosition Pplus = P;
      Pplus[i] += epsilon;
      KPosition Pminus = P;
      Pminus[i] -= epsilon;
      KEMThreeVector Bplus = MagneticField(Pplus);
      KEMThreeVector Bminus = MagneticField(Pminus);
      for (unsigned int j=0;j<3;j++)
	g[j + 3*i] = (Bplus[j]-Bminus[j])/(2.*epsilon);
    }
    return g;
  }
}
