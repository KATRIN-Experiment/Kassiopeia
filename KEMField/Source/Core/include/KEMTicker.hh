#ifndef KEMTICKER_DEF
#define KEMTICKER_DEF

#include "KEMCout.hh"

namespace KEMField
{
class KTicker
{
  public:
    KTicker() : fGoal(0), fCounter(0), fNoGoal(false) {}
    ~KTicker() {}

    void StartTicker(double goal = 0.);
    void Tick(double progress = 0.) const;
    void EndTicker() const;

  private:
    double fGoal;
    mutable unsigned int fCounter;
    bool fNoGoal;
};
}  // namespace KEMField

#endif /* KEMTICKER_DEF */
