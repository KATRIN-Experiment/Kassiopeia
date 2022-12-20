/**
 * @file KGlobals.hh
 * @brief Global variable definitions.
 * @date Created on: 26.11.2021
 * @author Benedikt Bieringer <benedikt.b@wwu.de>
 */

#ifndef KGLOBALS_H_
#define KGLOBALS_H_

#include "KSingleton.h"

namespace katrin
{

class KGlobals : public KSingleton<KGlobals>
{
  public:
    friend class KSingleton<KGlobals>;

  private:
    KGlobals();
    ~KGlobals() override;

  public:
    void SetBatchMode(bool);
    bool IsBatchMode();

    void SetVerbosityLevel(int);
    int VerbosityLevel();

  private:
    bool fAccessed;
    bool fBatchMode;
    int fVerbosityLevel;
};

}  // namespace katrin

#endif /* KGLOBALS_H_ */
