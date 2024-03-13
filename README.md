# Regional-ecological-time-series-monitoring-based-on-LandsatTrendr
*Use long-term series remote sensing image data obtained by LLR and LT for time series clustering and classification based on deep learning.*

## STEP 1: Get Your Date

**Using [`ee-LandsatLinkr`](https://github.com/gee-community/ee-LandsatLinkr) tools in GEE platform or python scrips developed by *Justin Braaten* & *Annie Taylor*.**

- ### OPTION 1: Gather your data with `google colab`

  Click me to get it!⚓ [![Click here to use it](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gee-community/ee-LandsatLinkr/blob/main/colab_template.ipynb)

- ### OPTION 2: Gather your data in [`GEE`](https://code.earthengine.google.com/2ec3c28efc3ecf15504979a9698a8b0d?noload=true)
<p align="center">
  <img width="800" height="400" src="https://i.postimg.cc/ZYQphMtL/01.png">
</p>

*then you can get these asset in your `GEE` account*
<p align="center">
    <img src="https://i.postimg.cc/bvKjQnZb/02.jpg">
</p>

*after this you can get data after execuating `LandasatLinkr`*
<p align="center">
    <img width='800' height='500' src="https://i.postimg.cc/d1QB1HBV/03.png">
</p>

*now you need to storage data in `Google Drive` before you download it to local(becouse of gee doesn't support to dowanload to local directly)*
<p align="center">
    <img width='800' height='400' src="https://i.postimg.cc/ZnRNN00G/04.png">
</p>

---
> [!NOTE]  
> These steps will cost lots of time, keep patient🛏️.
---

*and check data in ENVI*
<p align="center">
    <img width='800' height='400' src="https://i.postimg.cc/vZxGBnLJ/05.png">
</p>

Congratulations!㊗️ As now you have got the original data!🎆

## STEP 2: Fit Change Curve(e.g. NDVI index)
---
- **use IDL to execuate [`LandsatTrendr`](https://github.com/jdbcode/LLR-LandTrendr)**
  - we develop a GUI to help users to use it.
  <p align="center">
    <img src="https://i.postimg.cc/hPTkFLLv/13.png">
    <img src="https://i.postimg.cc/pT8fVJPc/17.png">
  </p>

  - **now see what we get**
  <p align="center">
    <img src="https://i.postimg.cc/28T5WL6H/14.png">
  </p>

