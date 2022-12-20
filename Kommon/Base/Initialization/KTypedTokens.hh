#ifndef Kommon_KTypedTokens_hh_
#define Kommon_KTypedTokens_hh_

#include "KToken.hh"

namespace katrin
{

template<class XType> class KTypedToken : public KToken
{
  public:
    KTypedToken();
    KTypedToken(const KTypedToken& aToken);
    ~KTypedToken() override;

    KToken* Clone() override;
};

template<class XType> inline KTypedToken<XType>::KTypedToken() : KToken() {}
template<class XType> inline KTypedToken<XType>::KTypedToken(const KTypedToken& aToken) : KToken(aToken) {}
template<class XType> KTypedToken<XType>::~KTypedToken() = default;

template<class XType> inline KToken* KTypedToken<XType>::Clone()
{
    return new KTypedToken<XType>(*this);
}

class KBeginParsing;
typedef KTypedToken<KBeginParsing> KBeginParsingToken;

class KEndParsing;
using KEndParsingToken = KTypedToken<KEndParsing>;

class KBeginFile;
using KBeginFileToken = KTypedToken<KBeginFile>;

class KEndFile;
using KEndFileToken = KTypedToken<KEndFile>;

class KBeginElement;
using KBeginElementToken = KTypedToken<KBeginElement>;

class KBeginAttribute;
using KBeginAttributeToken = KTypedToken<KBeginAttribute>;

class KAttributeData;
using KAttributeDataToken = KTypedToken<KAttributeData>;

class KEndAttribute;
using KEndAttributeToken = KTypedToken<KEndAttribute>;

class KMidElement;
using KMidElementToken = KTypedToken<KMidElement>;

class KElementData;
using KElementDataToken = KTypedToken<KElementData>;

class KEndElement;
using KEndElementToken = KTypedToken<KEndElement>;

class KComment;
using KCommentToken = KTypedToken<KComment>;

class KError;
using KErrorToken = KTypedToken<KError>;

}  // namespace katrin

#endif
