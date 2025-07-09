# SW-ObjectClassification-CAM
* Experiment 1: is an analysis of the performances of Grad-CAM , Grad-CAM++ and LayerCAM in a web-application working fully on the client side

* Experiment 2: is an implementation of object recognition with a heavyweight model and a lightweight model, both based on FasterRCNN, running on the client browser



## Requirements
- A webbrowser (This application has been tested only on Chrome  and Firefox)
- Internet is required to download some javascript modules and libraries: e.g the lastest version of tensorflow


## Execution
Run the following commands to open a Chrome Browser which avoid CORS Error

* MacOS (in Terminal) :
    ```
    open -na Google\ Chrome --args --user-data-dir=/tmp/temporary-chrome-profile-dir --disable-web-security --disable-site-isolation-trials
    ```

* Windows (from "Run" dialog [Windows+R] or start menu in Windows 8+) : 
    ```
    chrome.exe --user-data-dir=%TMP%\temporary-chrome-profile-dir --disable-web-security --disable-site-isolation-trials
    ```
<br/>
Then open <b>index.html</b>  in the web browser