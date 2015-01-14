#include "KGBox.hh"

namespace KGeoBag
{
  KGBox::KGBox() : KGArea()
  {
    fMeshCount[0] = fMeshCount[1] = fMeshCount[2] = 0;
    fMeshPower[0] = fMeshPower[1] = fMeshPower[2] = 1.;
  }

  KGBox::KGBox(double x0,double x1,double y0,double y1,double z0,double z1)
    : KGArea()
  {
    fMeshCount[0] = fMeshCount[1] = fMeshCount[2] = 0;
    fMeshPower[0] = fMeshPower[1] = fMeshPower[2] = 1.;

    fP0[0] = x0;
    fP1[0] = x1;
    if (x1 < x0)
    {
      fP0[0] = x1; fP1[0] = x0;
    }

    fP0[1] = y0;
    fP1[1] = y1;
    if (y1 < y0)
    {
      fP0[1] = y1; fP1[1] = y0;
    }

    fP0[2] = z0;
    fP1[2] = z1;
    if (z1 < z0)
    {
      fP0[2] = z1; fP1[2] = z0;
    }
  }

  KGBox::KGBox(const KThreeVector& p0,
	       const KThreeVector& p1) : KGArea()
  {
    fMeshCount[0] = fMeshCount[1] = fMeshCount[2] = 0;
    fMeshPower[0] = fMeshPower[1] = fMeshPower[2] = 1.;

    fP0[0] = p0[0];
    fP1[0] = p1[0];
    if (p1[0] < p0[0])
    {
      fP0[0] = p1[0]; fP1[0] = p0[0];
    }

    fP0[1] = p0[1];
    fP1[1] = p1[1];
    if (p1[1] < p0[1])
    {
      fP0[1] = p1[1]; fP1[1] = p0[1];
    }

    fP0[2] = p0[2];
    fP1[2] = p1[2];
    if (p1[2] < p0[2])
    {
      fP0[2] = p1[2]; fP1[2] = p0[2];
    }

  }

  void KGBox::AreaAccept(KGVisitor* aVisitor)
  {
    KGBox::Visitor* tBoxVisitor = dynamic_cast<KGBox::Visitor*>(aVisitor);
    if(tBoxVisitor != NULL)
    {
      tBoxVisitor->VisitBox(this);
    }
    return;
  }
  bool KGBox::AreaAbove(const KThreeVector& P) const
  {
    if ((P[0]-fP0[0])*(P[0]-fP1[0]) > 0. ||
	(P[1]-fP0[1])*(P[1]-fP1[1]) > 0. ||
	(P[2]-fP0[2])*(P[2]-fP1[2]) > 0.)
      return false;

    return true;
  }
  KThreeVector KGBox::AreaPoint(const KThreeVector& P) const
  {
    KThreeVector p = P;

    if (p[0] < fP0[0]) p[0] = fP0[0];
    else if (p[0] > fP1[0]) p[0] = fP1[0];
    if (p[1] < fP0[1]) p[1] = fP0[1];
    else if (p[1] > fP1[1]) p[1] = fP1[1];
    if (p[2] < fP0[2]) p[2] = fP0[2];
    else if (p[2] > fP1[2]) p[2] = fP1[2];

    return p;
  }
  KThreeVector KGBox::AreaNormal(const KThreeVector& P) const
  {
    KThreeVector dir = P - (fP0+fP1)*.5;
    for (int i=0;i<3;i++)
      dir[i]/=(fP1[i]-fP0[i]);

    int max = 0;
    if (dir[1]>dir[max])
      max = 1;
    if (dir[2]>dir[max])
      max = 2;

    double magnitude = 1.;
    if ((dir[max]>0. && P[max]<fP1[max]) || P[max]<fP0[max])
      magnitude = -1.;

    KThreeVector normal(0.,0.,0.);
    normal[max] = magnitude;

    return normal;
  }

}
