#ifndef KNAMED_H_
#define KNAMED_H_

#include "Printable.h"

#include <ostream>
#include <string>

namespace katrin
{

class KNamed : public Kommon::Printable
{
  public:
    KNamed();
    KNamed(const KNamed& aNamed);
    KNamed& operator=(const KNamed& other) = default;
    virtual ~KNamed() = default;
    bool HasName(const std::string& aName) const;
    const std::string& GetName() const;
    void SetName(std::string aName);
    void Print(std::ostream& output) const override;

  private:
    std::string fName;
};

}  // namespace katrin

#endif
