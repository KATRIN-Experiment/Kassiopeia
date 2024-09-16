#ifndef Kassiopeia_KSTimed_h_
#define Kassiopeia_KSTimed_h_

#include <chrono>

namespace Kassiopeia
{

class KSTimed {
  private:
    std::chrono::steady_clock::time_point fTimeStart;
    double fProcessingDuration = -1;

  public:
    void StartTiming();
    void EndTiming();

    const double& GetProcessingDuration() const {return fProcessingDuration;};
};

}

#endif
