#ifndef KAPPLICATIONRUNNER_H_
#define KAPPLICATIONRUNNER_H_

#include "KApplication.h"

#include <vector>

using std::vector;

namespace katrin
{

class KApplicationRunner : public KApplication
{
  public:
    KApplicationRunner();
    KApplicationRunner(const KApplicationRunner& aCopy);
    ~KApplicationRunner() override;

    bool Execute() override;
    void AddApplication(KApplication* tApplication);

  protected:
    std::vector<KApplication*> fApplications;
};

}  // namespace katrin


#endif