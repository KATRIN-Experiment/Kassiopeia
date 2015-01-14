#ifndef KZONALHARMONICPARAMETERS_DEF
#define KZONALHARMONICPARAMETERS_DEF



namespace KEMField
{
  class KZonalHarmonicParameters
  {
  public:
    KZonalHarmonicParameters() :
      fNBifurcations(5),
      fConvergenceRatio(0.99),
      fProximityToSourcePoint(1.e-14),
      fConvergenceParameter(1.e-15),
      fNCentralCoefficients(500),
      fCentralFractionalSpacing(true),
      fCentralFractionalDistance(.2),
      fCentralDeltaZ(.01),
      fCentralZ1(0.),
      fCentralZ2(0.),
      fNRemoteCoefficients(500),
      fNRemoteSourcePoints(3),
      fRemoteZ1(0.),
      fRemoteZ2(0.) {}

    virtual ~KZonalHarmonicParameters() {}

    static std::string Name() { return "ZonalHarmonicParameters"; }

    void SetNBifurcations(unsigned int i) { fNBifurcations = i; }
    void SetConvergenceRatio(double d) { fConvergenceRatio = d; }
    void SetProximityToSourcePoint(double d) { fProximityToSourcePoint = d; }
    void SetConvergenceParameter(double d) { fConvergenceParameter = d; }
    void SetNCentralCoefficients(unsigned int i) { fNCentralCoefficients = i; }
    void SetCentralFractionalSpacing(bool b) { fCentralFractionalSpacing = b; }
    void SetCentralFractionalDistance(double d) { fCentralFractionalDistance = d; }
    void SetCentralDeltaZ(double d) { fCentralDeltaZ = d; }
    void SetCentralZ1(double d) { fCentralZ1 = d; }
    void SetCentralZ2(double d) { fCentralZ2 = d; }
    void SetNRemoteCoefficients(unsigned int i) { fNRemoteCoefficients = i; }
    void SetNRemoteSourcePoints(unsigned int i) { fNRemoteSourcePoints = i; }
    void SetRemoteZ1(double d) { fRemoteZ1 = d; }
    void SetRemoteZ2(double d) { fRemoteZ2 = d; }

    unsigned int GetNBifurcations() const { return fNBifurcations; }
    double GetConvergenceRatio() const { return fConvergenceRatio; }
    double GetProximityToSourcePoint() const { return fProximityToSourcePoint; }
    double GetConvergenceParameter() const { return fConvergenceParameter; }
    unsigned int GetNCentralCoefficients() const { return fNCentralCoefficients; }
    bool GetCentralFractionalSpacing() const { return fCentralFractionalSpacing; }
    double GetCentralFractionalDistance() const { return fCentralFractionalDistance; }
    double GetCentralDeltaZ() const { return fCentralDeltaZ; }
    double GetCentralZ1() const { return fCentralZ1; }
    double GetCentralZ2() const { return fCentralZ2; }
    unsigned int GetNRemoteCoefficients() const { return fNRemoteCoefficients; }
    unsigned int GetNRemoteSourcePoints() const { return fNRemoteSourcePoints; }
    double GetRemoteZ1() const { return fRemoteZ1; }
    double GetRemoteZ2() const { return fRemoteZ2; }

  protected:
    unsigned int fNBifurcations;
    double fConvergenceRatio;
    double fProximityToSourcePoint;
    double fConvergenceParameter;
    unsigned int fNCentralCoefficients;
    bool fCentralFractionalSpacing;
    double fCentralFractionalDistance;
    double fCentralDeltaZ;
    double fCentralZ1;
    double fCentralZ2;
    unsigned int fNRemoteCoefficients;
    unsigned int fNRemoteSourcePoints;
    double fRemoteZ1;
    double fRemoteZ2;

  };

  template <typename Stream>
  Stream& operator>>(Stream& s,KZonalHarmonicParameters& p)
  {
    s.PreStreamInAction(p);

    unsigned int i;
    double d;
    bool b;

    s >> i; p.SetNBifurcations(i);
    s >> d; p.SetConvergenceRatio(d);
    s >> d; p.SetProximityToSourcePoint(d);
    s >> d; p.SetConvergenceParameter(d);
    s >> i; p.SetNCentralCoefficients(i);
    s >> b; p.SetCentralFractionalSpacing(b);
    s >> d; p.SetCentralFractionalDistance(d);
    s >> d; p.SetCentralDeltaZ(d);
    s >> d; p.SetCentralZ1(d);
    s >> d; p.SetCentralZ2(d);
    s >> i; p.SetNRemoteCoefficients(i);
    s >> i; p.SetNRemoteSourcePoints(i);
    s >> d; p.SetRemoteZ1(d);
    s >> d; p.SetRemoteZ2(d);

    s.PostStreamInAction(p);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KZonalHarmonicParameters& p)
  {
    s.PreStreamOutAction(p);

    s << p.GetNBifurcations();
    s << p.GetConvergenceRatio();
    s << p.GetProximityToSourcePoint();
    s << p.GetConvergenceParameter();
    s << p.GetNCentralCoefficients();
    s << p.GetCentralFractionalSpacing();
    s << p.GetCentralFractionalDistance();
    s << p.GetCentralDeltaZ();
    s << p.GetCentralZ1();
    s << p.GetCentralZ2();
    s << p.GetNRemoteCoefficients();
    s << p.GetNRemoteSourcePoints();
    s << p.GetRemoteZ1();
    s << p.GetRemoteZ2();

    s.PostStreamOutAction(p);
    return s;
  }

} // end namespace KEMField

#endif /* KZONALHARMONICPARAMETERS_DEF */
