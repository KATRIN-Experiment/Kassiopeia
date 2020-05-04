#ifndef _katrin_KVTKWindow_h_
#define _katrin_KVTKWindow_h_

#include "KWindow.h"
#include "vtkActor.h"
#include "vtkAxesActor.h"
#include "vtkCornerAnnotation.h"
#include "vtkOrientationMarkerWidget.h"
#include "vtkPolyData.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkSmartPointer.h"
#include "vtkXMLPolyDataWriter.h"

#include <set>
#include <string>
#include <vector>

namespace katrin
{

class KVTKPainter;

class KVTKWindow : public KWindow
{

  public:
    KVTKWindow();
    virtual ~KVTKWindow();

    //********
    //settings
    //********

  public:
    void SetWriteMode(bool aMode);
    bool GetWriteMode() const;

    void SetDisplayMode(bool aMode);
    bool GetDisplayMode() const;

    void SetHelpMode(bool aMode);
    bool GetHelpMode() const;

    void SetAxisMode(bool aMode);
    bool GetAxisMode() const;

    void SetDataMode(bool aMode);
    bool GetDataMode() const;

    void SetParallelProjectionMode(bool aMode);
    bool GetParallelProjectionMode() const;

    void SetFrameTitle(const std::string& aFrameTitle);
    const std::string& GetFrameTitle() const;

    void SetFrameSizeX(const unsigned int& anXPixelCount);
    const unsigned int& GetFrameSizeX() const;
    void SetFrameSizeY(const unsigned int& aYPixelCount);
    const unsigned int& GetFrameSizeY() const;

    void SetFrameColorRed(const float& aRed);
    const float& GetFrameColorRed() const;
    void SetFrameColorGreen(const float& aGreen);
    const float& GetFrameColorGreen() const;
    void SetFrameColorBlue(const float& aBlue);
    const float& GetFrameColorBlue() const;

    void SetEyeAngle(const double& eye);
    const double& GetEyeAngle() const;

    void SetViewAngle(const double& fov);
    const double& GetViewAngle() const;

    void SetMultiSamples(const unsigned int& samples);
    const unsigned int& GetMultiSamples() const;

    void SetDepthPeelingLevel(const unsigned int& level);
    const unsigned int& GetDepthPeelingLevel() const;

    vtkSmartPointer<vtkRenderWindow> GetRenderWindow() const;
    vtkSmartPointer<vtkRenderer> GetRenderer() const;
    vtkSmartPointer<vtkXMLPolyDataWriter> GetWriter() const;

  private:
    bool fWriteToggle;
    bool fDisplayToggle;
    bool fHelpToggle;
    bool fDataToggle;
    bool fAxisToggle;
    bool fParallelProjectionToggle;

    std::string fFrameTitle;
    unsigned int fFrameXPixels;
    unsigned int fFrameYPixels;
    float fFrameRed;
    float fFrameGreen;
    float fFrameBlue;

    double fEyeAngle;
    double fViewAngle;
    unsigned int fMultiSamples;
    unsigned int fDepthPeelingLevel;

  public:
    void Render();
    void Display();
    void Write();

    void AddPainter(KPainter* aPainter);
    void RemovePainter(KPainter* aPainter);

    void AddActor(vtkSmartPointer<vtkActor> anActor);
    void RemoveActor(vtkSmartPointer<vtkActor> anActor);

    void AddPoly(vtkSmartPointer<vtkPolyData> aPoly);
    void RemovePoly(vtkSmartPointer<vtkPolyData> aPoly);

  private:
    typedef std::set<KVTKPainter*> PainterSet;
    typedef PainterSet::iterator PainterIt;
    PainterSet fPainters;

    typedef std::vector<vtkSmartPointer<vtkActor>> ActorVector;
    typedef ActorVector::iterator ActorIt;
    ActorVector fActors;

    typedef std::vector<vtkSmartPointer<vtkPolyData>> PolyVector;
    typedef PolyVector::iterator PolyIt;
    PolyVector fPolys;

    //********
    //VTK data
    //********

  private:
    vtkSmartPointer<vtkXMLPolyDataWriter> fWriter;
    vtkSmartPointer<vtkRenderer> fRenderer;
    vtkSmartPointer<vtkRenderWindow> fRenderWindow;
    vtkSmartPointer<vtkRenderWindowInteractor> fRenderInteractor;

    vtkSmartPointer<vtkCornerAnnotation> fHelpActor;
    void UpdateHelp();

    vtkSmartPointer<vtkCornerAnnotation> fDataActor;
    void UpdateData();

    vtkSmartPointer<vtkOrientationMarkerWidget> fOrientationWidget;
    vtkSmartPointer<vtkAxesActor> fAxesActor;

    void Screenshot();

    static void OnKeyPress(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData);
    static void OnEnd(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData);
};

inline void KVTKWindow::SetDisplayMode(bool aMode)
{
    fDisplayToggle = aMode;
    return;
}
inline bool KVTKWindow::GetDisplayMode() const
{
    return fDisplayToggle;
}

inline void KVTKWindow::SetWriteMode(bool aMode)
{
    fWriteToggle = aMode;
    return;
}
inline bool KVTKWindow::GetWriteMode() const
{
    return fWriteToggle;
}

inline void KVTKWindow::SetAxisMode(bool aMode)
{
    fAxisToggle = aMode;
    return;
}
inline bool KVTKWindow::GetAxisMode() const
{
    return fAxisToggle;
}

inline void KVTKWindow::SetHelpMode(bool aMode)
{
    fHelpToggle = aMode;
    return;
}
inline bool KVTKWindow::GetHelpMode() const
{
    return fHelpToggle;
}

inline void KVTKWindow::SetDataMode(bool aMode)
{
    fDataToggle = aMode;
    return;
}
inline bool KVTKWindow::GetDataMode() const
{
    return fDataToggle;
}

inline void KVTKWindow::SetParallelProjectionMode(bool aMode)
{
    fParallelProjectionToggle = aMode;
    return;
}
inline bool KVTKWindow::GetParallelProjectionMode() const
{
    return fParallelProjectionToggle;
}

inline void KVTKWindow::SetFrameTitle(const std::string& aTitle)
{
    fFrameTitle = aTitle;
    return;
}
inline const std::string& KVTKWindow::GetFrameTitle() const
{
    return fFrameTitle;
}

inline void KVTKWindow::SetFrameSizeX(const unsigned int& anXPixelCount)
{
    fFrameXPixels = anXPixelCount;
    return;
}
inline void KVTKWindow::SetFrameSizeY(const unsigned int& aYPixelCount)
{
    fFrameYPixels = aYPixelCount;
    return;
}

inline void KVTKWindow::SetFrameColorRed(const float& aRed)
{
    fFrameRed = aRed;
    return;
}
inline const float& KVTKWindow::GetFrameColorRed() const
{
    return fFrameRed;
}
inline void KVTKWindow::SetFrameColorGreen(const float& aGreen)
{
    fFrameGreen = aGreen;
    return;
}
inline const float& KVTKWindow::GetFrameColorGreen() const
{
    return fFrameGreen;
}
inline void KVTKWindow::SetFrameColorBlue(const float& aBlue)
{
    fFrameBlue = aBlue;
    return;
}
inline const float& KVTKWindow::GetFrameColorBlue() const
{
    return fFrameBlue;
}

inline void KVTKWindow::SetEyeAngle(const double& eye)
{
    fEyeAngle = eye;
    return;
}
inline const double& KVTKWindow::GetEyeAngle() const
{
    return fEyeAngle;
}

inline void KVTKWindow::SetViewAngle(const double& fov)
{
    fViewAngle = fov;
    return;
}
inline const double& KVTKWindow::GetViewAngle() const
{
    return fViewAngle;
}

inline void KVTKWindow::SetMultiSamples(const unsigned int& samples)
{
    fMultiSamples = samples;
    return;
}
inline const unsigned int& KVTKWindow::GetMultiSamples() const
{
    return fMultiSamples;
}

inline void KVTKWindow::SetDepthPeelingLevel(const unsigned int& level)
{
    fDepthPeelingLevel = level;
    return;
}
inline const unsigned int& KVTKWindow::GetDepthPeelingLevel() const
{
    return fDepthPeelingLevel;
}

inline vtkSmartPointer<vtkRenderWindow> KVTKWindow::GetRenderWindow() const
{
    return fRenderWindow;
}

inline vtkSmartPointer<vtkRenderer> KVTKWindow::GetRenderer() const
{
    return fRenderer;
}
inline vtkSmartPointer<vtkXMLPolyDataWriter> KVTKWindow::GetWriter() const
{
    return fWriter;
}

}  // namespace katrin

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////                                                   /////
/////  BBBB   U   U  IIIII  L      DDDD   EEEEE  RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB   U   U    I    L      D   D  EE     RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB    UUU   IIIII  LLLLL  DDDD   EEEEE  R   R  /////
/////                                                   /////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////


#endif
