#ifndef KGPATHAWARE_HH_
#define KGPATHAWARE_HH_

#include "KTagged.h"

#include <string>

namespace KGeoBag
{

class KGPathAware : public katrin::KTagged
{
  public:
    KGPathAware();
    KGPathAware(const KGPathAware& aCopy);    
    ~KGPathAware() override;

    KGPathAware& operator=(const KGPathAware& other);
    
    virtual std::string GetPath() const {return "";};
};

}  // namespace KGeoBag

#endif
