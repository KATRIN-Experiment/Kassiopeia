#ifndef KFIELD_H_
#define KFIELD_H_

#define K_REFS(xTYPE, xVARIABLE)                                                                                       \
  public:                                                                                                              \
    const xTYPE& Get##xVARIABLE() const                                                                                \
    {                                                                                                                  \
        return f##xVARIABLE;                                                                                           \
    }                                                                                                                  \
    xTYPE& xVARIABLE()                                                                                                 \
    {                                                                                                                  \
        return f##xVARIABLE;                                                                                           \
    }                                                                                                                  \
                                                                                                                       \
  protected:                                                                                                           \
    xTYPE f##xVARIABLE;

#define K_GET(xTYPE, xVARIABLE)                                                                                        \
  public:                                                                                                              \
    const xTYPE& Get##xVARIABLE() const                                                                                \
    {                                                                                                                  \
        return f##xVARIABLE;                                                                                           \
    }                                                                                                                  \
                                                                                                                       \
  protected:                                                                                                           \
    xTYPE f##xVARIABLE;

#define K_GET_PTR(xTYPE, xVARIABLE)                                                                                    \
  public:                                                                                                              \
    xTYPE* Get##xVARIABLE() const                                                                                      \
    {                                                                                                                  \
        return f##xVARIABLE;                                                                                           \
    }                                                                                                                  \
                                                                                                                       \
  protected:                                                                                                           \
    xTYPE* f##xVARIABLE;

#define K_SET(xTYPE, xVARIABLE)                                                                                        \
  public:                                                                                                              \
    void Set##xVARIABLE(const xTYPE& aVariable)                                                                        \
    {                                                                                                                  \
        f##xVARIABLE = aVariable;                                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
                                                                                                                       \
  protected:                                                                                                           \
    xTYPE f##xVARIABLE;

#define K_SET_PTR(xTYPE, xVARIABLE)                                                                                    \
  public:                                                                                                              \
    void Set##xVARIABLE(xTYPE* aVariable)                                                                              \
    {                                                                                                                  \
        f##xVARIABLE = aVariable;                                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
                                                                                                                       \
  protected:                                                                                                           \
    xTYPE* f##xVARIABLE;

#define K_SET_GET(xTYPE, xVARIABLE)                                                                                    \
  public:                                                                                                              \
    void Set##xVARIABLE(const xTYPE& aVariable)                                                                        \
    {                                                                                                                  \
        f##xVARIABLE = aVariable;                                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    const xTYPE& Get##xVARIABLE() const                                                                                \
    {                                                                                                                  \
        return f##xVARIABLE;                                                                                           \
    }                                                                                                                  \
                                                                                                                       \
  protected:                                                                                                           \
    xTYPE f##xVARIABLE;

#define K_SET_GET_PTR(xTYPE, xVARIABLE)                                                                                \
  public:                                                                                                              \
    void Set##xVARIABLE(xTYPE* aVariable)                                                                              \
    {                                                                                                                  \
        f##xVARIABLE = aVariable;                                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    xTYPE* Get##xVARIABLE() const                                                                                      \
    {                                                                                                                  \
        return f##xVARIABLE;                                                                                           \
    }                                                                                                                  \
                                                                                                                       \
  protected:                                                                                                           \
    xTYPE* f##xVARIABLE;

#endif
