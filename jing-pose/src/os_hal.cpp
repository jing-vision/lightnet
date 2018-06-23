#if WIN32
#include <windows.h>
#else
#include <X11/Xlib.h>
#endif

void getScreenResolution(int &width, int &height)
{
#if WIN32
    width = (int)GetSystemMetrics(SM_CXSCREEN);
    height = (int)GetSystemMetrics(SM_CYSCREEN);
#else
    Display *disp = XOpenDisplay(NULL);
    Screen *scrn = DefaultScreenOfDisplay(disp);
    width = scrn->width;
    height = scrn->height;
#endif
}
